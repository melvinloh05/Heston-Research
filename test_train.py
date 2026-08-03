"""Smoke tests for the training entry points (train_pinn TASKS 1-4 + train.py CLI).

CPU, tiny sizes. The label artifact is built in-module exactly as test_make_labels does
(N_POINTS=8, N_SKT=4, SEED=0, shrunk legs) — accuracy is the oracle selftest's job; here
we check plumbing: split-restricted batching, the full-split loss-scale freeze, PDE
collocation sampling, run-to-run determinism, early stopping, the matched-epochs +
pilot + lambda-selection paths, and that nothing above regresses the existing suites.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

import train
import train_pinn as tp
from make_labels import generate_labels
from SobolevPINN import SobolevPINN, load_arm
from train_pinn import (ArmDataset, HESTON_PARAM_NAMES, LockedTestSet, TrainConfig,
                        config_hash, pde_sampling_spec, sample_pde_points,
                        select_lambdas, train_model)

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
N_POINTS, N_SKT, SEED = 8, 4, 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
SCALE_TERMS = ("price", "delta", "gamma", "vega", "vanna", "bc", "pde")


@pytest.fixture(scope="module")
def npz(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("train_labels")
    res = generate_labels(CONTRACT, CFG, N_POINTS, SEED, str(out),
                          n_skt=N_SKT, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


def _tiny_tcfg(**kw) -> TrainConfig:
    base = dict(steps=40, batch=128, n_pde=128, val_every=20, patience=10)
    base.update(kw)
    return TrainConfig(**base)


def _spec():
    return pde_sampling_spec(CONTRACT, CFG)


# ---------------------------------------------------------------------------
# TASK 1 — split-restricted batching + full-split loss-scale freeze
# ---------------------------------------------------------------------------

def test_arm_dataset_minibatch_carries_build_arm_labels_keys(npz):
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    ds = ArmDataset(npz, cfg, "train", seed=0)
    assert ds.n_rows > 0
    mb = next(ds.iter_minibatches(4, seed=0, epoch=0))
    # gamma_ref is REQUIRED (loss-scale invariant); keys match build_arm_labels exactly
    assert "gamma_ref" in mb
    assert set(mb) == {"x", "price", "delta", "vega", "vanna", "gamma_ref", "gamma"}
    assert mb["x"].shape[1] == len(cfg.inputs) and mb["x"].shape[0] <= 4


def test_split_key_is_read_and_leakage_safe(npz, tmp_path):
    # inject a by-parameter-point split into a copy and confirm ArmDataset honors it
    d = dict(np.load(npz, allow_pickle=False))
    n_points, n_skt = d["mask_any"].shape
    split = np.zeros(n_points, dtype=np.int8)
    split[::2] = 1                                      # alternate points -> val
    d["split"] = split
    path = str(tmp_path / "with_split.npz")
    np.savez(path, **d)
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    tr = ArmDataset(path, cfg, "train", seed=0)
    va = ArmDataset(path, cfg, "val", seed=0)
    assert tr.n_rows > 0 and va.n_rows > 0
    assert tr.n_rows + va.n_rows == int((~d["mask_any"].ravel()).sum())   # partition of kept rows
    # leakage-safety: the parameter point of every retained row lands in the declared split
    keep = ~d["mask_any"].ravel()
    point_of_row = np.flatnonzero(keep) // n_skt
    assert (split[point_of_row[split[point_of_row] == 0]] == 0).all()
    assert (split[point_of_row[split[point_of_row] == 1]] == 1).all()


def test_missing_split_key_falls_back_and_warns(npz):
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    with pytest.warns(RuntimeWarning, match="no 'split' key"):
        ArmDataset(npz, cfg, "train", seed=0, warn=True)


def test_full_split_scale_freeze_is_arm_and_order_invariant(npz):
    """The freeze runs ONCE on the whole training split (canonical order) BEFORE the loop,
    so per-term scales are bit-identical (a) across arms and (b) across minibatch orders."""
    # (a) arm invariance: freeze directly on the full split for two different arms
    a = ArmDataset(npz, load_arm(CFG, "rung1_delta"), "train", seed=3)
    b = ArmDataset(npz, load_arm(CFG, "rung3_delta_gamma_vega"), "train", seed=3)
    tp.set_seed(0)
    ma = SobolevPINN(load_arm(CFG, "rung1_delta"))
    ma._freeze_loss_scales(a.data)
    tp.set_seed(0)
    mb = SobolevPINN(load_arm(CFG, "rung3_delta_gamma_vega"))
    mb._freeze_loss_scales(b.data)
    for term in SCALE_TERMS:
        assert torch.equal(getattr(ma, f"loss_scale_{term}"),
                           getattr(mb, f"loss_scale_{term}")), term

    # (b) order invariance: two full training runs whose ONLY difference is the seeded
    # minibatch order must carry bit-identical loss-scale buffers (weights DO differ).
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    tr = ArmDataset(npz, cfg, "train", seed=3)
    va = ArmDataset(npz, cfg, "val", seed=3)
    tcfg = _tiny_tcfg(steps=24, batch=6)               # batch < n_rows -> real reshuffling
    ranges, anchors, fmin, exc = _spec()

    def run(seed):
        _, best, _, _ = train_model(cfg, tr, va, tcfg, seed, pde_ranges=ranges,
                                    pde_anchors=anchors, feller_min=fmin,
                                    excise_rel_radius=exc)
        return best

    s7, s8 = run(7), run(8)
    for term in SCALE_TERMS:
        assert torch.equal(s7[f"loss_scale_{term}"], s8[f"loss_scale_{term}"]), term
    weight_keys = [k for k in s7 if k.startswith("net.")]
    assert any(not torch.equal(s7[k], s8[k]) for k in weight_keys)   # orders really differed


# ---------------------------------------------------------------------------
# TASK 1 — PDE collocation sampling
# ---------------------------------------------------------------------------

def test_pde_points_satisfy_feller_and_excision_and_loss_has_pde(npz):
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    ranges, anchors, fmin, exc = _spec()
    x = sample_pde_points(3000, 123, ranges, cfg.inputs, anchors=anchors,
                          feller_min=fmin, excise_rel_radius=exc)
    assert x.shape == (3000, len(cfg.inputs))           # columns follow cfg.inputs
    i = {n: k for k, n in enumerate(cfg.inputs)}
    k, t, xi = x[:, i["kappa"]], x[:, i["theta"]], x[:, i["xi"]]
    assert (2.0 * k * t / xi ** 2 >= fmin).all()        # Feller filter
    # anchor excision — same distance check the excision test uses (parameter dims)
    pnames = list(HESTON_PARAM_NAMES)
    lo = np.array([ranges[n][0] for n in pnames])
    hi = np.array([ranges[n][1] for n in pnames])
    un = (np.stack([x[:, i[n]] for n in pnames], 1) - lo) / (hi - lo)
    r_norm = exc * np.sqrt(len(pnames))
    for anc in anchors:
        au = (np.array([anc[n] for n in pnames]) - lo) / (hi - lo)
        assert np.linalg.norm(un - au, axis=1).min() >= r_norm
    for ax in ("S", "K", "tau"):                        # grid coords uniform in range
        lo2, hi2 = ranges[ax]
        assert (x[:, i[ax]] >= lo2).all() and (x[:, i[ax]] <= hi2).all()

    # the loss dict gains a "pde" term when x_pde is plumbed through
    tp.set_seed(0)
    model = SobolevPINN(cfg)
    tr = ArmDataset(npz, cfg, "train", seed=0)
    model._freeze_loss_scales(tr.data)
    mb = next(tr.iter_minibatches(128, 0, 0))
    mb["x_pde"] = torch.as_tensor(x[:64], dtype=torch.float32)
    assert "pde" in model.loss(mb)


# ---------------------------------------------------------------------------
# TASK 2 — determinism + accounting
# ---------------------------------------------------------------------------

def test_two_runs_same_seed_bit_equal_best_diff_seed_differs(npz):
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    tr = ArmDataset(npz, cfg, "train", seed=1)
    va = ArmDataset(npz, cfg, "val", seed=1)
    tcfg = _tiny_tcfg()
    ranges, anchors, fmin, exc = _spec()

    def run(seed):
        _, best, _, log = train_model(cfg, tr, va, tcfg, seed, pde_ranges=ranges,
                                      pde_anchors=anchors, feller_min=fmin,
                                      excise_rel_radius=exc)
        return best, log

    a, la = run(7)
    b, _ = run(7)
    c, _ = run(8)
    for kk in a:
        assert torch.equal(a[kk], b[kk]), kk                # same seed -> bit-equal
    assert any(not torch.equal(a[kk], c[kk]) for kk in a)   # different seed -> differs
    # compute accounting present and sensible
    acct = la["compute"]
    for f in ("wall_clock_s", "param_count", "derivative_evals", "peak_memory_bytes"):
        assert acct[f] > 0, f
    assert acct["n_pde_per_epoch"] == tcfg.n_pde


def test_config_hash_depends_on_cfg_and_seed():
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    h = config_hash(cfg, 42, "datasha", "lamsha")
    assert h == config_hash(cfg, 42, "datasha", "lamsha")         # stable
    assert h != config_hash(cfg, 43, "datasha", "lamsha")         # seed matters
    assert h != config_hash(cfg, 42, "OTHER", "lamsha")           # data manifest matters
    from dataclasses import replace
    assert h != config_hash(replace(cfg, lambda_gamma=2.0), 42, "datasha", "lamsha")


# ---------------------------------------------------------------------------
# TASK 2/3 — early stopping + checkpoints
# ---------------------------------------------------------------------------

def test_early_stop_before_max_steps_best_is_best_val(npz, monkeypatch):
    cfg = load_arm(CFG, "rung3_delta_gamma_vega")
    tr = ArmDataset(npz, cfg, "train", seed=2)
    va = ArmDataset(npz, cfg, "val", seed=2)
    counter = {"n": 0}

    def non_improving(model, val_batch):               # each val is worse than the last
        counter["n"] += 1
        return float(counter["n"]), {"total": float(counter["n"])}

    monkeypatch.setattr(tp, "_eval_val", non_improving)
    tcfg = _tiny_tcfg(steps=1000, val_every=5, patience=1, n_pde=64)
    ranges, anchors, fmin, exc = _spec()
    _, _, _, log = train_model(cfg, tr, va, tcfg, 2, pde_ranges=ranges, pde_anchors=anchors,
                               feller_min=fmin, excise_rel_radius=exc, early_stop=True)
    assert log["compute"]["stopped_early"]
    assert log["compute"]["steps"] < 1000                        # stopped before max
    assert log["checkpoints"]["best"]["step"] == 5               # first val was the best


def test_early_stopped_run_has_no_matched_epochs_checkpoint(npz, tmp_path, monkeypatch):
    """T2: train.py defaults to early_stop = not --matched-epochs, so the DEFAULT
    run early-stops. Filing that checkpoint under 'matched_epochs' would let the
    contract's report_both table compare arms that stopped at different steps as
    though they had a matched budget. last.pt is 'last'; 'matched_epochs' exists
    only when the step budget was actually reached."""
    counter = {"n": 0}

    def non_improving(model, val_batch):               # forces the early stop
        counter["n"] += 1
        return float(counter["n"]), {"total": float(counter["n"])}

    monkeypatch.setattr(tp, "_eval_val", non_improving)
    # the contract cadence (val_every 500, patience 10) needs 5500 steps to trip;
    # shrink ONLY the cadence, so the default early_stop path is what is tested
    from dataclasses import replace
    from_dict = tp.TrainConfig.from_dict
    monkeypatch.setattr(tp.TrainConfig, "from_dict",
                        staticmethod(lambda d: replace(from_dict(d), batch=128,
                                                       n_pde=128, val_every=5,
                                                       patience=1)))
    out = tmp_path / "early"
    runlog = train.main(["--arm", "rung3_delta_gamma_vega", "--seed", "5", "--data", npz,
                         "--pinn-cfg", CFG, "--contract", CONTRACT,
                         "--lambdas", str(tmp_path / "absent.yaml"),
                         "--out", str(out), "--steps", "1000"])   # DEFAULT: early stop on
    assert runlog["compute"]["stopped_early"]
    ck = runlog["checkpoints"]
    assert "matched_epochs" not in ck, (
        "an early-stopped checkpoint was filed as matched_epochs; the report_both "
        "table would compare unequal budgets as if matched")
    assert ck["last"]["path"] == "last.pt"
    assert ck["last"]["reached_max_steps"] is False
    assert (out / "last.pt").exists()                  # the FILE is still written
    assert runlog["matched_epochs_mode"] is False


def test_matched_epochs_checkpoint_exists_alongside_best(npz, tmp_path):
    out = tmp_path / "run"
    runlog = train.main(["--arm", "rung3_delta_gamma_vega", "--seed", "5", "--data", npz,
                         "--pinn-cfg", CFG, "--contract", CONTRACT,
                         "--lambdas", str(tmp_path / "absent.yaml"),
                         "--out", str(out), "--steps", "30", "--matched-epochs"])
    for f in ("best.pt", "last.pt", "runlog.json", "loss_curves.csv"):
        assert (out / f).exists(), f
    ck = runlog["checkpoints"]
    assert set(ck) == {"best", "last", "matched_epochs"}         # runlog has BOTH
    assert ck["matched_epochs"]["path"] == "last.pt"
    assert ck["matched_epochs"]["step"] == 30 and ck["matched_epochs"]["reached_max_steps"]
    # T2: 'matched_epochs' is the SAME checkpoint as 'last', present only because
    # the step budget was reached — not a second file.
    assert ck["last"] == ck["matched_epochs"]
    # best.pt carries cfg + frozen scales for reload
    saved = torch.load(out / "best.pt", weights_only=False)
    assert saved["config_hash"] == runlog["config_hash"]
    assert any(k.startswith("loss_scale") for k in saved["scales"])


def test_pilot_prints_finite_sigma_gamma(npz, tmp_path, capsys):
    out = tmp_path / "pilot"
    runlog = train.main(["--arm", "rung3_delta_gamma_vega", "--seed", "4", "--data", npz,
                         "--pinn-cfg", CFG, "--contract", CONTRACT,
                         "--lambdas", str(tmp_path / "absent.yaml"),
                         "--out", str(out), "--pilot", "--steps", "20"])
    printed = capsys.readouterr().out
    assert "sigma_gamma_pilot BEFORE fix" in printed
    assert "sigma_gamma_pilot AFTER  fix" in printed
    assert np.isfinite(runlog["sigma_gamma_pilot"])
    assert np.isfinite(runlog["sigma_gamma_pilot_relative"])
    assert runlog["sigma_gamma_pilot_source"] == "gamma_ref"


class _FakeGammaModel:
    """Stub with a fixed greeks_eval output, for isolating _pilot_gamma_rmse's truth-source
    priority from any actual training/fitting."""

    def __init__(self, gamma_pred: torch.Tensor) -> None:
        self._gamma_pred = gamma_pred

    def greeks_eval(self, x, need=()):
        return {"gamma": self._gamma_pred}


def test_pilot_gamma_rmse_prefers_gamma_ref_over_corrupted_arm_label():
    # gamma_ref is the true-consensus label; "gamma" simulates a sigma_* dose arm's corrupted
    # label, deliberately far from both gamma_ref and the model prediction.
    gamma_ref = torch.tensor([1.0, 2.0, 3.0, 4.0])
    corrupted = gamma_ref + 100.0
    pred = gamma_ref.clone()                          # model matches the TRUE consensus exactly
    val_ds = type("Stub", (), {})()
    val_ds.data = {"x": torch.zeros(4, 1), "gamma_ref": gamma_ref, "gamma": corrupted}
    model = _FakeGammaModel(pred)

    rmse, rel, source = train._pilot_gamma_rmse(model, val_ds, "cpu")
    assert source == "gamma_ref"
    assert rmse == pytest.approx(0.0)
    assert rel == pytest.approx(0.0)

    # legacy (buggy) priority reproduces the inflated sigma_gamma for before/after comparison
    rmse_legacy, rel_legacy, source_legacy = train._pilot_gamma_rmse(
        model, val_ds, "cpu", prefer_ref=False)
    assert source_legacy == "gamma (arm label)"
    assert rmse_legacy == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TASK 4 — validation-only lambda selection
# ---------------------------------------------------------------------------

def test_select_lambdas_writes_yaml_and_guards_locked_test_set(tmp_path):
    guard = LockedTestSet({"anchor_grids": 1})

    def score(lp, lg, lv):                              # convex bowl -> unique argmin
        return (lp - 0.1) ** 2 + (lg - 1.0) ** 2 + (lv - 1.0) ** 2

    out = tmp_path / "lambdas_selected.yaml"
    res = select_lambdas([0.0, 0.1], [1.0, 10.0], [1.0, 10.0], fit_and_val_score=score,
                         test_set=guard, out_path=str(out), config_hash="abc123")
    assert out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert (loaded["lambda_pde"], loaded["lambda_gamma"], loaded["lambda_vega"]) == (0.1, 1.0, 1.0)
    assert res["lambda_delta"] == 1.0 and res["config_hash"] == "abc123"
    assert len(res["scores_table"]) == 2 * 2 * 2         # full candidate grid
    assert guard.locked                                  # never unlocked during selection

    # a fit that unlocks the guard mid-search must raise (test set touched)
    g2 = LockedTestSet({"anchor_grids": 1})

    def leaky(lp, lg, lv):
        g2.unlock()                                      # simulate touching the held-out set
        return 0.0

    with pytest.raises(RuntimeError, match="touched"):
        select_lambdas([1.0], [1.0], [1.0], fit_and_val_score=leaky, test_set=g2)
    # an already-unlocked guard is refused at entry
    g3 = LockedTestSet({"x": 1})
    g3.unlock()
    with pytest.raises(RuntimeError, match="unlocked"):
        select_lambdas([1.0], [1.0], [1.0], fit_and_val_score=score, test_set=g3)


def test_select_lambdas_is_staged_and_records_its_source_arms(npz, tmp_path):
    """Q3: lambda_pde is selected on the BASELINE arm (for which the PDE residual is the
    only structural signal), then (lambda_gamma, lambda_vega) on rung3 at that fixed
    lambda_pde. Both source arms are recorded IN the artifact, and the one shared
    lambda_pde reaches the baseline and the treatment arm identically."""
    lam_sel = yaml.safe_load(open(CONTRACT))["lambda_selection"]
    out = tmp_path / "lam_staged.yaml"
    res = train.main(["--select-lambdas", "--seed", "0", "--data", npz,
                      "--pinn-cfg", CFG, "--contract", CONTRACT, "--out", str(out),
                      "--cand-pde", "0.1,1.0", "--cand-gamma", "1.0",
                      "--cand-vega", "0.3,1.0", "--steps", "6"])
    lam = yaml.safe_load(out.read_text())

    # provenance lives in the artifact, and matches the contract's pre-registered sourcing
    assert lam["sources"]["lambda_pde"] == lam_sel["lambda_pde"]["source_arm"]
    assert lam["sources"]["lambda_gamma"] == lam_sel["lambda_gamma"]["source_arm"]
    assert lam["sources"]["lambda_vega"] == lam_sel["lambda_vega"]["source_arm"]
    assert lam["sources"]["lambda_pde"] != lam["sources"]["lambda_gamma"]

    # STAGED, not joint: 2 lambda_pde fits on the baseline, then a 1x2 grid on rung3
    assert len(lam["scores_table_pde"]) == 2
    assert all(set(r) == {"lambda_pde", "score"} for r in lam["scores_table_pde"])
    assert len(lam["scores_table"]) == 2
    assert all(r["lambda_pde"] == lam["lambda_pde"] for r in lam["scores_table"])
    assert res["lambda_pde"] in (0.1, 1.0) and res["lambda_delta"] == 1.0

    # the shared lambda_pde reaches BOTH arms identically (_apply_lambdas unchanged)
    std = train._apply_lambdas(load_arm(CFG, lam_sel["lambda_pde"]["source_arm"]), lam)
    r3 = train._apply_lambdas(load_arm(CFG, lam_sel["lambda_gamma"]["source_arm"]), lam)
    assert std.lambda_pde == lam["lambda_pde"] == r3.lambda_pde
    assert r3.lambda_gamma == lam["lambda_gamma"] and r3.lambda_vega == lam["lambda_vega"]


def test_select_lambdas_cli_smoke_writes_yaml(npz, tmp_path):
    out = tmp_path / "lam.yaml"
    res = train.main(["--select-lambdas", "--arm", "rung3_delta_gamma_vega", "--seed", "0",
                      "--data", npz, "--pinn-cfg", CFG, "--contract", CONTRACT,
                      "--out", str(out), "--cand-pde", "1.0", "--cand-gamma", "1.0",
                      "--cand-vega", "1.0", "--steps", "6"])
    assert out.exists()
    assert res["lambda_pde"] == 1.0 and len(res["scores_table"]) == 1
    assert np.isfinite(res["scores_table"][0]["score"])
