"""Tests for the frozen path banks (Hedging_backtest.bank_write/_BankLoader) and
the run_hedging.py runners.

Path-bank tests use a Black-Scholes stub provider (no checkpoints, fast). The
runner/band tests build ONE tiny trained best.pt (train.main, 8 steps) and copy
it into every <arm>/s<seed>/best.pt the run needs — the copies are identical, so
every arm produces the same Greeks, which is all the plumbing (10 seeds, per-cell
pnl npz, ledger resume, magnitude-0.5-only band selection) needs to be exercised.

Runnable via pytest.
"""
from __future__ import annotations

import glob
import os
import shutil

import numpy as np
import pytest

import Hedging_backtest as hb
import run_hedging as rh
import train
from make_labels import generate_labels
from pinn_provider import _ARM_DIR
from test_hedging_backtest import BSProvider

CONTRACT = "heston_benchmark_v6.yaml"
PINN_CFG = "pinn_config.yaml"
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
CONFIRMATORY_SEEDS = list(range(42, 52))          # global_seed 42 + 0..9


# ---------------------------------------------------------------------------
# tiny resolved cfg (production sizes overridden)
# ---------------------------------------------------------------------------

def _cfg(n_paths=32, freq=52, n_boot=20):
    cfg = hb.resolve_config(CONTRACT, "hedging_config.yaml")
    eng = cfg["engine"]
    eng["simulation"]["n_paths"] = n_paths
    eng["rebalancing"]["frequency_per_year"] = freq
    eng["risk"]["bootstrap_B"] = n_boot
    return cfg


# ===========================================================================
# TASK 1 — frozen path banks
# ===========================================================================

def _stub_providers(cfg):
    r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
    return {"oracle": BSProvider(r, q), "standard_pinn": BSProvider(r, q)}


def _tiny_bank_cfg():
    cfg = _cfg()
    cfg["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]
    cfg["derived"]["seeds"] = cfg["derived"]["seeds"][:1]
    return cfg


def test_bank_loaded_run_equals_on_the_fly_bitforbit(tmp_path):
    cfg = _tiny_bank_cfg()
    provs = _stub_providers(cfg)
    rows_live = hb.run_headline(cfg, provs, None)

    bank = tmp_path / "bank"
    info = hb.bank_write(cfg, str(bank))
    # perturbation (3 dir x 2 mag) + bates (10) + merton (1) = 17 cells at one seed
    assert info["n_cells"] == 17
    assert (bank / "bank_manifest.json").exists()

    cfg2 = _tiny_bank_cfg()
    cfg2["engine"]["simulation"]["paths_dir"] = str(bank)
    rows_bank = hb.run_headline(cfg2, provs, None)

    assert len(rows_live) == len(rows_bank)
    # The PATHS are bit-for-bit (asserted directly below); the METRIC columns are
    # asserted to round-off, because they are accumulated through the pricing stack,
    # which on this platform reproduces to ~1 ULP rather than exactly on repeated
    # identical calls (docs/CODE_AUDIT_2026-08-20.md measured 1.8e-15 on the CF
    # provider). The invariant is that a frozen bank yields the SAME RUN as an
    # on-the-fly simulation; a stale or mis-keyed bank moves these columns by O(0.1),
    # nine orders above the tolerance.
    for a, b in zip(rows_live, rows_bank):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], float) and isinstance(b[k], float):
                assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), (k, a[k], b[k])
            else:
                assert a[k] == b[k], (k, a[k], b[k])

    # the paths themselves ARE bit-for-bit — that is what a frozen bank guarantees
    # and it involves no floating-point accumulation, only RNG + arithmetic
    cfg3 = _tiny_bank_cfg()
    for seed, tag, dgp_kind, dgp in hb._iter_sim_cells(cfg3):
        loader = hb._BankLoader(str(bank), cfg3)
        n_steps = int(round(float(cfg3["engine"]["horizon"]["T_prime"])
                            * cfg3["engine"]["rebalancing"]["frequency_per_year"]))
        t_b, S_b, v_b = loader.load(hb._cell_slug(tag, seed), dgp_kind, dgp,
                                    cfg3["engine"]["simulation"]["n_paths"], n_steps)
        if dgp_kind == "merton":
            t_l, S_l, v_l = hb.simulate_merton(
                dgp, 100.0, float(cfg3["engine"]["horizon"]["T_prime"]), n_steps,
                cfg3["engine"]["simulation"]["n_paths"], seed,
                cfg3["benchmark"]["grid"]["r"], cfg3["benchmark"]["grid"]["q"])
        else:
            t_l, S_l, v_l = hb.simulate_heston_qe(
                dgp, 100.0, float(cfg3["engine"]["horizon"]["T_prime"]), n_steps,
                cfg3["engine"]["simulation"]["n_paths"], seed,
                cfg3["engine"]["simulation"]["psi_c"])
        assert np.array_equal(S_b, S_l) and np.array_equal(v_b, v_l), tag
        assert np.array_equal(t_b, t_l), tag
        break        # one cell is enough; bank_write/_BankLoader share one code path


def test_bank_tampered_sha_raises(tmp_path):
    cfg = _tiny_bank_cfg()
    provs = _stub_providers(cfg)
    bank = tmp_path / "bank"
    hb.bank_write(cfg, str(bank))
    victim = sorted(glob.glob(str(bank / "*.npz")))[0]
    with open(victim, "ab") as fh:                 # flip the file's bytes
        fh.write(b"\x00tamper")
    cfg["engine"]["simulation"]["paths_dir"] = str(bank)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        hb.run_headline(cfg, provs, None)


def test_bank_dgp_mismatch_raises(tmp_path):
    """A bank built at one shift must refuse to serve a differently-shifted sweep
    (same slugs, wrong paths): the per-cell DGP check hard-errors."""
    cfg = _tiny_bank_cfg()
    provs = _stub_providers(cfg)
    bank = tmp_path / "bank"
    hb.bank_write(cfg, str(bank))
    # change the engine shift so the same cells now request different DGP params
    cfg["engine"]["misspecification"]["directions"]["xi_up"]["shift_at_m1"] = 0.10
    cfg["engine"]["simulation"]["paths_dir"] = str(bank)
    with pytest.raises(ValueError, match="DGP mismatch"):
        hb.run_headline(cfg, provs, None)


def test_bank_missing_cell_raises(tmp_path):
    """paths_dir set but a requested cell absent from the bank -> hard error, no
    silent resimulation."""
    cfg = _tiny_bank_cfg()
    provs = _stub_providers(cfg)
    bank = tmp_path / "bank"
    hb.bank_write(cfg, str(bank))
    # ask for a magnitude never banked
    cfg["engine"]["misspecification"]["magnitudes"] = [0.5]
    cfg["engine"]["simulation"]["paths_dir"] = str(bank)
    with pytest.raises(KeyError):
        hb.run_headline(cfg, provs, None)


def test_bank_write_refuses_frozen_dir(tmp_path):
    cfg = _tiny_bank_cfg()
    with pytest.raises(AssertionError, match="frozen"):
        hb.bank_write(cfg, str(tmp_path / "data" / "frozen" / "paths"))


# ===========================================================================
# fixtures: one tiny trained best.pt copied across every arm/seed dir
# ===========================================================================

@pytest.fixture(scope="module")
def _npz(tmp_path_factory):
    out = tmp_path_factory.mktemp("labels")
    res = generate_labels(CONTRACT, PINN_CFG, 8, 0, str(out), n_skt=4,
                          mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


@pytest.fixture(scope="module")
def ckpt_root(_npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpts")
    src = tmp_path_factory.mktemp("src_ckpt")
    train.main(["--arm", "rung3_delta_gamma_vega", "--seed", "0", "--data", _npz,
                "--pinn-cfg", PINN_CFG, "--contract", CONTRACT,
                "--lambdas", str(src / "absent.yaml"), "--out", str(src),
                "--steps", "8"])
    best = src / "best.pt"
    assert best.exists()
    for arm in rh._CONFIRMATORY_ARMS:
        arm_dir = _ARM_DIR.get(arm, arm)
        for seed in CONFIRMATORY_SEEDS:
            dst = root / arm_dir / f"s{seed}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(best, dst / "best.pt")
    return str(root)


# ===========================================================================
# TASK 2 — confirmatory runner + ledger resume
# ===========================================================================

@pytest.fixture(scope="module")
def confirmatory_run(ckpt_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("confirm_out")
    res = rh.run_confirmatory(_cfg(), ckpt_root, str(out))
    return res, str(out)


def test_confirmatory_uses_exactly_10_seeds(confirmatory_run):
    res, out = confirmatory_run
    assert res["seeds"] == CONFIRMATORY_SEEDS
    assert len(res["seeds"]) == 10
    # combined perturbation x {0.0, 1.0} magnitudes x 10 seeds = 20 cells
    assert res["n_ran"] == 20
    # ledger records every (cell, seed); 10 distinct seeds
    import csv
    with open(res["ledger"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    assert sorted({int(r["seed"]) for r in rows}) == CONFIRMATORY_SEEDS


def test_confirmatory_persists_pnl_for_every_cell_tc_method(confirmatory_run):
    res, out = confirmatory_run
    pnl_dir = res["pnl_dir"]
    npz_files = sorted(glob.glob(os.path.join(pnl_dir, "*.npz")))
    assert len(npz_files) == 20                    # one per (cell, seed)
    tiers = [0.0, 0.01, 0.02]
    methods = ["oracle", "standard_pinn", "standard_pinn_smoothed",
               "rung1", "rung2", "rung3"]
    for f in npz_files:
        keys = set(np.load(f).files)
        for m in methods:
            for tc in tiers:
                assert f"{m}__tc{tc}" in keys, (os.path.basename(f), m, tc)


def test_resolved_config_describes_the_program_not_the_last_cell(confirmatory_run):
    """R1: the engine writes resolved_config.yaml into the shared run root on
    EVERY (cell, seed) call, and _run_program calls it with a TRIMMED config —
    one direction, one magnitude, one seed. What survived was the last cell's
    trim, so the provenance record of a 2-magnitude, 10-seed confirmatory run
    claimed to be a single-seed, single-magnitude run. Nothing numerical moves;
    for a pre-registered study the provenance record is load-bearing."""
    import yaml

    res, out = confirmatory_run
    path = os.path.join(res["run_root"], "resolved_config.yaml")
    assert os.path.exists(path)
    cfg = yaml.safe_load(open(path))

    assert cfg["derived"]["seeds"] == CONFIRMATORY_SEEDS          # all 10, not the last
    assert cfg["engine"]["misspecification"]["magnitudes"] == [0.0, 1.0]
    # the program's own trim IS still recorded (combined only, no cross-model)
    mis = cfg["benchmark"]["hedging_simulation"]["misspecification"]
    assert list(mis["perturbations"]) == ["combined"]
    assert mis["cross_model"] == []
    # and it round-trips into the same cell set the run actually executed
    n_cells = sum(1 for _ in hb._iter_sim_cells(cfg))
    assert n_cells == res["n_ran"] + res["n_skipped"] == 20


def test_ledger_resume_skips_completed_cells(confirmatory_run, ckpt_root):
    _res, out = confirmatory_run                   # first run already completed
    # re-run against the SAME out_dir: every (cell, seed) is in the ledger -> skipped
    again = rh.run_confirmatory(_cfg(), ckpt_root, out)
    assert again["n_ran"] == 0
    assert again["n_skipped"] == 20
    # the ledger did not grow (no duplicate rows appended for done cells)
    import csv
    with open(again["ledger"]) as fh:
        assert len(list(csv.DictReader(fh))) == 20


# ===========================================================================
# TASK 3 — no-trade band width on magnitude-0.5 validation only
# ===========================================================================

def test_select_band_width_only_reads_magnitude_0p5(ckpt_root, tmp_path):
    cfg = _cfg()
    # include the confirmatory magnitude 1.0 to prove it is NEVER consumed
    cfg["engine"]["misspecification"]["magnitudes"] = [0.0, 0.5, 1.0]
    cfg["derived"]["seeds"] = [42, 43]
    sel = rh.select_band_width(cfg, ckpt_root, widths=(0.01, 0.02),
                               out_dir=str(tmp_path))
    assert sel["magnitudes_consumed"] == [0.5]     # never 0.0 or 1.0
    assert sel["no_trade_band"] in (0.01, 0.02)
    assert set(sel["table"]) == {0.01, 0.02}
    assert os.path.exists(sel["path"])


def test_select_band_width_rebuilds_providers_per_seed(ckpt_root, tmp_path, monkeypatch):
    """The selection's seed diversity must be real: providers are rebuilt ONCE
    per engine seed (checkpoints are per-seed), not built once and reused so every
    seed hedges seed 0's network."""
    cfg = _cfg()
    cfg["engine"]["misspecification"]["magnitudes"] = [0.5]
    cfg["derived"]["seeds"] = [42, 43, 44]
    seeds_seen: list[int] = []
    real = rh.build_providers

    def _spy(*args, **kwargs):
        seeds_seen.append(args[3] if len(args) > 3 else kwargs["seed"])
        return real(*args, **kwargs)

    monkeypatch.setattr(rh, "build_providers", _spy)
    rh.select_band_width(cfg, ckpt_root, widths=(0.01, 0.02), out_dir=str(tmp_path))
    # exactly one build per seed (not per width, not once total), every seed covered
    assert seeds_seen == [42, 43, 44]


def test_selected_band_is_applied_by_runner(tmp_path):
    """_apply_selected_band overrides the hedging_config default when a
    band_selected.yaml is present, and leaves it untouched otherwise."""
    cfg = _cfg()
    default = cfg["engine"]["smoothing_baseline"]["no_trade_band"]
    rh._apply_selected_band(cfg, None)             # absent -> unchanged (0.02)
    assert cfg["engine"]["smoothing_baseline"]["no_trade_band"] == default

    band = tmp_path / "band_selected.yaml"
    import yaml
    band.write_text(yaml.safe_dump({"no_trade_band": 0.005}))
    rh._apply_selected_band(cfg, str(band))
    assert cfg["engine"]["smoothing_baseline"]["no_trade_band"] == 0.005
