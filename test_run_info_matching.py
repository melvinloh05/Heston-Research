"""Tests for the A10 information-matching saturation driver (run_info_matching.py).

CPU, tiny sizes. The label artifact is the same shrunk-leg fixture the other training
suites use (make_labels.generate_labels, N_POINTS=8, N_SKT=4). We check the four things
the driver promises: (1) the frozen-row subsample is deterministic AND nested (never
resamples labels); (2) the declared plateau rule fires correctly on flat / monotone /
noisy synthetic curves; (3) the 5N cap assertion is enforced (reused from train_pinn);
(4) the whole pipeline wires end to end at steps<=60 — CSVs, PNG and the contract-worded
paragraph, plus the capacity-control (2x width) rung.
"""
from __future__ import annotations

import csv
import json
import warnings

import numpy as np
import pytest
import torch
import yaml

import run_info_matching as rim
from make_labels import generate_labels
from pinn_provider import build_providers
from SobolevPINN import load_arm
from train_pinn import ArmDataset, saturation_sweep_configs

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
N_POINTS, N_SKT, SEED = 8, 4, 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}


@pytest.fixture(scope="module")
def npz(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("im_labels")
    res = generate_labels(CONTRACT, CFG, N_POINTS, SEED, str(out),
                          n_skt=N_SKT, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


# ---------------------------------------------------------------------------
# (1) subsample determinism + nestedness (never resamples labels)
# ---------------------------------------------------------------------------

def test_subsample_is_deterministic_and_nested(npz):
    base = load_arm(CFG, rim.SATURATED_ARM)
    full = ArmDataset(npz, base, "train", seed=0)
    assert full.n_rows >= 4                                  # enough rows to actually subsample

    n1 = max(2, full.n_rows // 3)
    n2 = max(n1 + 1, (2 * full.n_rows) // 3)
    a = rim.subsample_train(full, n1, seed=0)
    b = rim.subsample_train(full, n1, seed=0)
    assert a.n_rows == n1 and b.n_rows == n1
    for k in a.data:                                        # same (dataset, n, seed) -> bit-equal
        assert torch.equal(a.data[k], b.data[k]), k

    # NESTED: the smaller budget's rows are a subset of the larger's (same seed) -> growing
    # the budget ADDS rows, it does not swap them. Matched on the (S,K,tau) input signature.
    big = rim.subsample_train(full, n2, seed=0)
    sig_small = {tuple(np.round(r, 8)) for r in a.data["x"].numpy()}
    sig_big = {tuple(np.round(r, 8)) for r in big.data["x"].numpy()}
    assert sig_small <= sig_big

    # different seed -> generally a different subset (not a resample of NEW labels: every
    # retained row still comes from the frozen split)
    c = rim.subsample_train(full, n1, seed=7)
    all_sig = {tuple(np.round(r, 8)) for r in full.data["x"].numpy()}
    assert {tuple(np.round(r, 8)) for r in c.data["x"].numpy()} <= all_sig

    # capped at the frozen row count — no invented rows past what was frozen
    capped = rim.subsample_train(full, 10 * full.n_rows, seed=0)
    assert capped.n_rows == full.n_rows


# ---------------------------------------------------------------------------
# (2) the plateau rule on synthetic curves
# ---------------------------------------------------------------------------

def test_plateau_flat_curve_stops_at_first_rung():
    p = rim.plateau_multiplier([1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0, 1.0])
    assert p["plateau_multiplier"] == 2 and p["plateau_reached"]
    assert np.isnan(p["rel_improvement"][0])                # first rung has no predecessor


def test_plateau_monotone_curve_never_plateaus_pins_at_cap():
    # every rung halves the error (50% gain >> 2%) -> plateau never reached -> pinned at cap
    p = rim.plateau_multiplier([1, 2, 3, 4, 5], [1.0, 0.5, 0.25, 0.125, 0.0625])
    assert p["plateau_multiplier"] == 5 and not p["plateau_reached"]


def test_plateau_noisy_curve_first_subthreshold_rung():
    # big gain at m=2, then a <2% gain at m=3 (the plateau), even though m=5 improves again
    p = rim.plateau_multiplier([1, 2, 3, 4, 5], [1.0, 0.70, 0.695, 0.690, 0.60])
    assert p["plateau_multiplier"] == 3 and p["plateau_reached"]


def test_plateau_worsening_rung_counts_as_plateau():
    # a rung that gets WORSE (negative improvement) is < tol -> trips the plateau
    p = rim.plateau_multiplier([1, 2, 3], [1.0, 1.2, 0.5])
    assert p["plateau_multiplier"] == 2 and p["plateau_reached"]


# ---------------------------------------------------------------------------
# (3) the 5N cap assertion (reused from train_pinn.saturation_sweep_configs)
# ---------------------------------------------------------------------------

def test_cap_5N_is_enforced():
    base = load_arm(CFG, rim.SATURATED_ARM)
    saturation_sweep_configs(base, [1, 2, 5], cap_multiplier=5)          # at the cap: fine
    with pytest.raises(AssertionError, match="cap"):
        saturation_sweep_configs(base, [1, 2, 6], cap_multiplier=5)      # 6N > 5N: refused


# ---------------------------------------------------------------------------
# (4) end-to-end wiring smoke at steps <= 60
# ---------------------------------------------------------------------------

def test_wiring_smoke_writes_all_outputs(npz, tmp_path):
    out = tmp_path / "im_run"
    res = rim.run_saturation_sweep(
        npz, seeds=[0, 1], out_dir=str(out), pinn_cfg=CFG, contract=CONTRACT,
        steps=20, multipliers=[1, 2, 3], base_n_price_points=6,
        train_overrides={"n_pde": 64, "batch": 64, "val_every": 10})

    for name in ("saturation_curve_per_seed.csv", "saturation_curve.csv",
                 "saturation_curve.png", "saturation_summary.txt"):
        assert (out / name).exists(), name

    # per-seed rows: 3 multipliers x 2 seeds (width 1.0) + plateau rung x 2 seeds (width 2.0)
    with open(out / "saturation_curve_per_seed.csv", newline="") as fh:
        ps = list(csv.DictReader(fh))
    assert len(ps) == 3 * 2 + 2
    widths = {float(r["width_mult"]) for r in ps}
    assert widths == {1.0, 2.0}                             # capacity-control rung present
    # every run logs the contract-mandated accounting fields
    for r in ps:
        for f in ("config_hash", "wall_clock_s", "param_count", "derivative_evals",
                  "peak_memory_bytes"):
            assert r[f] not in ("", None), f
        # subsample bit on the tiny artifact: budget grows with the multiplier (base_n=6)
        assert int(r["n_train_rows"]) <= int(r["n_price_points"])

    # curve CSV: the plateau annotation is present and consistent with the returned dict
    with open(out / "saturation_curve.csv", newline="") as fh:
        agg = list(csv.DictReader(fh))
    pm = res["plateau"]["plateau_multiplier"]
    plateau_rows = [r for r in agg if r["is_plateau"] == "True"]
    assert plateau_rows and all(int(r["multiplier"]) == pm for r in plateau_rows)
    assert any(float(r["width_mult"]) == 2.0 for r in agg)  # 2x-width point in the curve

    # the paragraph carries the contract's EXACT wording, read from the source of truth
    wording = yaml.safe_load(open(CONTRACT))["information_matching"]["wording"]
    summary = (out / "saturation_summary.txt").read_text()
    assert wording in summary and summary == res["paragraph"] + "\n"


def test_row_cap_cannot_masquerade_as_a_plateau(npz, tmp_path):
    """I1: when the frozen artifact holds fewer rows than the ladder requests, the top
    rungs train on BIT-IDENTICAL data and the Gamma curve is flat BY CONSTRUCTION.

    A plateau at or below a capped rung bounds what the LABEL ARTIFACT contains, not
    what the architecture extracts from prices — the sweep must refuse to report it as
    a reached plateau."""
    out = tmp_path / "capped_run"
    base_n = 10_000                                          # >> the fixture's train rows
    with pytest.warns(RuntimeWarning, match="capped"):
        res = rim.run_saturation_sweep(
            npz, seeds=[0], out_dir=str(out), pinn_cfg=CFG, contract=CONTRACT,
            steps=16, multipliers=[1, 2], base_n_price_points=base_n,
            train_overrides={"n_pde": 64, "batch": 64, "val_every": 8})

    plat = res["plateau"]
    assert plat["plateau_reached"] is False                  # NOT a demonstrated plateau
    assert plat["plateau_capped"] is True
    assert plat["capped_rungs"] == [1, 2]

    # every capped rung is visible in the artifacts, per seed and on the curve
    with open(out / "saturation_curve_per_seed.csv", newline="") as fh:
        ps = list(csv.DictReader(fh))
    assert ps and all(r["subsample_capped"] == "True" for r in ps)
    assert all(int(r["n_train_rows"]) < int(r["n_train_rows_requested"]) for r in ps)
    with open(out / "saturation_curve.csv", newline="") as fh:
        agg = list(csv.DictReader(fh))
    assert any(r["subsample_capped"] == "True" for r in agg)

    # ... and the sidecar the downstream sweep reads says so too
    rec = json.loads((out / "info_matched_baseline" / "s0" /
                      rim.BUDGET_SIDECAR).read_text())
    assert rec["data_capped_multipliers"] == [1, 2]
    assert "CAPPED" in rec["note"]


def test_preflight_refuses_before_a_single_training_step(npz, tmp_path, monkeypatch):
    """N5/ITEM 7: the row-cap check is decided by `train_ds.n_rows` and the m*N
    ladder, BOTH known before the first optimizer step. On a full-size run the
    post-hoc check spends hours of GPU before announcing the ladder was unfillable.

    train_model is monkeypatched to explode, so if anything trains, the test fails
    with THAT error instead — i.e. the warning below can only have been emitted
    pre-flight."""
    def _boom(*a, **k):
        raise AssertionError("a training step ran despite an unfillable ladder")

    monkeypatch.setattr(rim, "train_model", _boom)
    out = tmp_path / "preflight_warn"
    with pytest.warns(RuntimeWarning, match="PRE-FLIGHT"):
        with pytest.raises(AssertionError, match="a training step ran"):
            rim.run_saturation_sweep(
                npz, seeds=[0], out_dir=str(out), pinn_cfg=CFG, contract=CONTRACT,
                steps=16, multipliers=[1, 2], base_n_price_points=10_000,
                train_overrides={"n_pde": 64, "batch": 64, "val_every": 8})

    # strict mode refuses outright, still before any training
    out2 = tmp_path / "preflight_raise"
    with pytest.raises(ValueError, match="rows"):
        rim.run_saturation_sweep(
            npz, seeds=[0], out_dir=str(out2), pinn_cfg=CFG, contract=CONTRACT,
            steps=16, multipliers=[1, 2], base_n_price_points=10_000,
            preflight="raise",
            train_overrides={"n_pde": 64, "batch": 64, "val_every": 8})


def test_preflight_is_silent_on_a_fillable_ladder(npz, tmp_path, monkeypatch):
    """The control: a ladder the artifact CAN fill must not warn or refuse, in
    either mode — the check has to discriminate, not always fire."""
    calls = []
    monkeypatch.setattr(rim, "train_model",
                        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
                            AssertionError("stop after the pre-flight")))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError, match="stop after the pre-flight"):
            rim.run_saturation_sweep(
                npz, seeds=[0], out_dir=str(tmp_path / "ok"), pinn_cfg=CFG,
                contract=CONTRACT, steps=16, multipliers=[1, 2],
                base_n_price_points=4, preflight="raise",
                train_overrides={"n_pde": 64, "batch": 64, "val_every": 8})
    assert calls, "the fillable ladder must reach training"
    assert not [w for w in caught if "PRE-FLIGHT" in str(w.message)]


def test_uncapped_ladder_reports_no_capping(npz, tmp_path):
    """The control: a ladder the artifact CAN fill reports capped=False everywhere, so
    the flag discriminates rather than always firing."""
    out = tmp_path / "uncapped_run"
    res = rim.run_saturation_sweep(
        npz, seeds=[0], out_dir=str(out), pinn_cfg=CFG, contract=CONTRACT,
        steps=16, multipliers=[1, 2], base_n_price_points=4,
        train_overrides={"n_pde": 64, "batch": 64, "val_every": 8})
    assert res["plateau"]["capped_rungs"] == []
    assert res["plateau"]["plateau_capped"] is False
    with open(out / "saturation_curve_per_seed.csv", newline="") as fh:
        ps = list(csv.DictReader(fh))
    assert ps and all(r["subsample_capped"] == "False" for r in ps)


def test_cli_smoke(npz, tmp_path):
    out = tmp_path / "cli_run"
    res = rim.main(["--data", npz, "--out-dir", str(out), "--pinn-cfg", CFG,
                    "--contract", CONTRACT, "--seeds", "0", "--steps", "16",
                    "--multipliers", "1,2", "--base-n", "6"])
    assert (out / "saturation_curve.csv").exists()
    assert res["seeds"] == [0] and res["multipliers"] == [1, 2]
    assert res["plateau"]["plateau_multiplier"] in (1, 2)


# ---------------------------------------------------------------------------
# (5) the plateau-m checkpoint the full sweep hedges — layout, schema, loadability
# ---------------------------------------------------------------------------

def test_persists_plateau_checkpoint_loadable_by_full_sweep(npz, tmp_path):
    out = tmp_path / "im_run"
    ckpt = tmp_path / "grid"                                 # the shared grid root the sweep reads
    base_n = 6
    res = rim.run_saturation_sweep(
        npz, seeds=[0, 1], out_dir=str(out), ckpt_root=str(ckpt), pinn_cfg=CFG,
        contract=CONTRACT, steps=20, multipliers=[1, 2, 3], base_n_price_points=base_n,
        train_overrides={"n_pde": 64, "batch": 64, "val_every": 10})
    pm = res["plateau"]["plateau_multiplier"]

    for seed in (0, 1):
        # EXACT layout pinn_provider.build_providers reads: <root>/info_matched_baseline/s<seed>/
        ck = ckpt / "info_matched_baseline" / f"s{seed}" / "best.pt"
        sc = ckpt / "info_matched_baseline" / f"s{seed}" / "info_matched_budget.json"
        assert ck.exists() and sc.exists(), seed
        blob = torch.load(str(ck), map_location="cpu", weights_only=False)
        # schema matches train.py's best.pt (state_dict + cfg + scales + sha/DGP audit fields)
        assert set(blob) >= {"state_dict", "cfg", "scales", "config_hash", "seed", "arm"}
        assert blob["arm"] == "info_matched_baseline" and blob["seed"] == seed
        # the PLATEAU-m budget at WIDTH 1.0 — never N, never the 2x capacity-control rung
        assert blob["cfg"]["n_price_points"] == pm * base_n
        assert blob["cfg"]["width_mult"] == 1.0

        rec = json.loads(sc.read_text())                    # strict JSON (non-finite -> null)
        assert rec["selected_multiplier"] == pm
        assert rec["plateau_reached"] == res["plateau"]["plateau_reached"]
        assert rec["n_price_points"] == pm * base_n
        assert rec["config_hash"] == blob["config_hash"]
        assert len(rec["saturation_curve"]) == 3            # one entry per swept multiplier

    # the real integration gate: load through the SAME entry point run_full_sweep uses
    cfg = yaml.safe_load(open(CONTRACT))
    provs = build_providers(cfg, str(ckpt), ["info_matched_baseline"], seed=0,
                            r=cfg["grid"]["r"], q=cfg["grid"]["q"], include_oracle=False)
    g = provs["info_matched_baseline"].evaluate(
        np.array([100.0, 101.0]), np.array([0.04, 0.05]), tau=0.17, K=100.0)
    assert set(g) == {"price", "delta", "gamma", "vega", "vanna"}
    assert all(np.isfinite(g[k]).all() for k in g)


def test_sidecar_pins_at_cap_when_plateau_not_reached():
    """When the plateau never fires, the sidecar records the CAP budget loudly — never N."""
    plat = {"plateau_multiplier": 5, "plateau_index": 4, "plateau_reached": False,
            "rel_improvement": [float("nan"), 0.5, 0.5, 0.5, 0.5]}
    row = {"n_train_rows": 20, "fd_equivalent_budget": 100, "raw_scalar_budget": 80}
    rec = rim._budget_sidecar(seed=0, pm=5, N=6, plat=plat, sat_curve=[],
                              plateau_row=row, tol=rim.PLATEAU_TOL, cap=5,
                              accounting="fd_equivalent", chash="abc", dsha="def")
    assert rec["plateau_reached"] is False and rec["plateau_pinned_at_cap"] is True
    assert rec["selected_multiplier"] == 5 and rec["n_price_points"] == 30   # 5N, not N
    assert "NOT the N budget" in rec["note"]


def test_checkpoint_defaults_under_out_dir(npz, tmp_path):
    """ckpt_root omitted -> checkpoints land under out_dir (no silent skip of persistence)."""
    out = tmp_path / "im_run"
    res = rim.run_saturation_sweep(
        npz, seeds=[0], out_dir=str(out), pinn_cfg=CFG, contract=CONTRACT, steps=16,
        multipliers=[1, 2], base_n_price_points=6,
        train_overrides={"n_pde": 64, "batch": 64, "val_every": 10})
    assert (out / "info_matched_baseline" / "s0" / "best.pt").exists()
    assert res["ckpt_root"] == str(out) and 0 in res["checkpoints"]
