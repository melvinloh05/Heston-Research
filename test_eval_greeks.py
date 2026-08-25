"""Tests for eval_greeks.py: the OOD-parameter Greek metric layer.

Fast + self-contained. Anchor grids are FAKE npz (controlled consensus / uncertainty /
mask fields in the make_datasets.generate_anchor_grids layout) so the metric machinery —
mask exclusion, the wing absolute-only rule, regime-param injection, and the comparison /
aggregation algebra — is tested in isolation from oracle accuracy. Tiny TRAINED
checkpoints (train.py smoke settings) supply real finite model predictions.
"""
from __future__ import annotations

import csv
import warnings

import numpy as np
import pytest
import torch

import train
from eval_greeks import (AGG_COLS, GREEKS, PERSEED_COLS, THRESHOLD_COLS,
                         aggregate_seeds, eval_arm_on_regime,
                         improvement_to_oracle_noise, price_parity,
                         reduction_vs_baseline, run_greek_eval)
from make_labels import generate_labels
from pinn_provider import PINNProvider
from train_pinn import HESTON_PARAM_NAMES

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
N_POINTS, N_SKT, SEED0 = 8, 4, 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
R, Q = 0.02, 0.0
TRAIN_REGIME = {"kappa": 2.0, "theta": 0.04, "xi": 0.30, "rho": -0.50}

# (engine name, config/dir arm name) for the checkpoints we build
ARM_SET = [("standard_pinn", "standard_pinn"), ("rung2", "rung2_delta_gamma"),
           ("rung3", "rung3_delta_gamma_vega")]
ENGINE_ARMS = [e for e, _ in ARM_SET]
SEEDS = [0, 1]

# midpoint of the contract's hedging box, read from the contract (never a literal) so
# every fake anchor grid carries a tau node inside it
def _hedge_tau_mid() -> float:
    import yaml as _yaml

    from eval_greeks import hedge_slice_spec
    sp = hedge_slice_spec(_yaml.safe_load(open(CONTRACT)))
    return 0.5 * (sp["tau_lo"] + sp["tau_hi"])


_HEDGE_TAU_MID = _hedge_tau_mid()

# regime params in HESTON_PARAM_NAMES order (kappa, theta, xi, rho, v0)
REGIME_PARAMS = {
    "baseline": [2.0, 0.04, 0.30, -0.50, 0.04],
    "high_variance": [2.0, 0.09, 0.30, -0.50, 0.09],
    "feller_violating_volvol": [2.0, 0.04, 0.60, -0.50, 0.04],
    "strong_neg_corr": [2.0, 0.04, 0.30, -0.90, 0.04],
    "near_feller": [1.5, 0.04, 0.34, -0.70, 0.02],
}


# ---------------------------------------------------------------------------
# fake anchor grid builder
# ---------------------------------------------------------------------------

def _fake_grid(path, params, *, seed=0, mask_points=(), poison=False,
               nS=4, nK=3, nT=3, param_names=HESTON_PARAM_NAMES):
    """Minimal {regime}_grid.npz: full contract-grid axes, injected params, controlled
    consensus/uncertainty/mask. `mask_points` = list of (i,j,k) grid cells set in mask_any;
    `poison` sets those cells' consensus to NaN (to prove masked points are never read).
    `param_names` is written alongside `params` (Q6) and can be permuted to prove the
    reader binds by NAME."""
    S_ax = np.linspace(50.0, 150.0, nS)
    K_ax = np.linspace(60.0, 140.0, nK)
    # The tau axis carries one node INSIDE the contract's hedging box so the "hedge"
    # slice (CODE_AUDIT_2026-08-20 action 1) is non-empty on every fixture grid; a
    # coarse linspace over [0.04, 1.0] straddles the box without ever landing in it.
    T_ax = np.unique(np.concatenate([np.linspace(0.04, 1.0, nT), [_HEDGE_TAU_MID]]))
    shape = (nS, nK, T_ax.size)
    rng = np.random.default_rng(seed)
    mask_any = np.zeros(shape, bool)
    for pt in mask_points:
        mask_any[pt] = True
    arr = {"S_axis": S_ax, "K_axis": K_ax, "tau_axis": T_ax,
           "params": np.asarray(params, float),
           "param_names": np.array(param_names), "mask_any": mask_any,
           "r": np.float64(R), "q": np.float64(Q), "seed": np.int64(seed)}
    for g in GREEKS:
        cons = rng.uniform(0.5, 1.5, shape)
        if poison:
            cons[mask_any] = np.nan
        arr[f"consensus_{g}"] = cons
        arr[f"uncertainty_{g}"] = rng.uniform(0.01, 0.05, shape)
        arr[f"mask_{g}"] = mask_any.copy()
    np.savez(path, **arr)
    return str(path)


# ---------------------------------------------------------------------------
# fixtures: checkpoints (train.py smoke) + provider + fake anchors dir
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npz(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("eg_labels")
    res = generate_labels(CONTRACT, CFG, N_POINTS, SEED0, str(out),
                          n_skt=N_SKT, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


@pytest.fixture(scope="module")
def ckpt_root(npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("eg_ckpts")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _engine, arm in ARM_SET:
            for s in SEEDS:
                train.main(["--arm", arm, "--seed", str(s), "--data", npz,
                            "--pinn-cfg", CFG, "--contract", CONTRACT,
                            "--lambdas", str(root / "absent.yaml"),
                            "--out", str(root / arm / f"s{s}"), "--steps", "60"])
    return root


@pytest.fixture(scope="module")
def provider(ckpt_root):
    path = ckpt_root / "rung3_delta_gamma_vega" / "s0" / "best.pt"
    return PINNProvider(str(path), TRAIN_REGIME, R, Q)


@pytest.fixture
def anchors_dir(tmp_path):
    for name, params in REGIME_PARAMS.items():
        _fake_grid(tmp_path / f"{name}_grid.npz", params)
    return tmp_path


# ---------------------------------------------------------------------------
# eval_arm_on_regime: masking, wing rule, regime injection
# ---------------------------------------------------------------------------

def test_masked_points_excluded_from_every_metric(provider, tmp_path):
    mp = [(0, 0, 0), (2, 1, 1)]
    clean = _fake_grid(tmp_path / "a_clean.npz", REGIME_PARAMS["baseline"],
                       seed=5, mask_points=mp, poison=False)
    poison = _fake_grid(tmp_path / "a_poison.npz", REGIME_PARAMS["baseline"],
                        seed=5, mask_points=mp, poison=True)   # masked cells -> NaN consensus
    mc = eval_arm_on_regime(provider, clean, "baseline")
    mp_ = eval_arm_on_regime(provider, poison, "baseline")
    # derived from the grid, not a literal: _fake_grid's tau axis carries an extra
    # node inside the contract's hedging box, so a hardcoded cell count would rot
    n_cells = int(np.load(clean)["mask_any"].size)
    for g in GREEKS:
        assert mc[g]["n_unmasked"] == n_cells - len(mp)
        for k in ("rmse", "rel_rmse", "p50", "p90", "p95", "p99", "oracle_unc_rms"):
            assert np.isfinite(mc[g][k]), (g, k)                # no NaN leaked from masked pts
            # poisoning a MASKED point changes nothing (it is never read)
            assert mc[g][k] == pytest.approx(mp_[g][k], rel=0, abs=0), (g, k)


def test_wing_gamma_is_absolute_only(provider, anchors_dir):
    npz_path = str(anchors_dir / "baseline_grid.npz")
    from eval_greeks import _slice_masks
    wing = _slice_masks(npz_path)["wing"]

    full = eval_arm_on_regime(provider, npz_path, "baseline")
    wing_slice = eval_arm_on_regime(provider, npz_path, "baseline", restrict=wing)

    # full grid: gamma rel_rmse is computed on the body (finite); wing slice: it is BLANK
    assert np.isfinite(full["gamma"]["rel_rmse"])
    assert np.isfinite(full["gamma"]["rmse"])
    assert not np.isfinite(wing_slice["gamma"]["rel_rmse"])     # absolute-only on the wing
    assert np.isfinite(wing_slice["gamma"]["rmse"])             # absolute metric still valid
    # non-gamma greeks keep a finite relative metric on the wing
    for g in ("price", "delta", "vega"):
        assert np.isfinite(wing_slice[g]["rel_rmse"]), g


def test_regime_params_are_injected(provider, tmp_path):
    # same checkpoint + same consensus, different regime params -> different predictions
    a = _fake_grid(tmp_path / "reg_a.npz", REGIME_PARAMS["baseline"], seed=11)
    b = _fake_grid(tmp_path / "reg_b.npz", REGIME_PARAMS["strong_neg_corr"], seed=11)
    ma = eval_arm_on_regime(provider, a, "baseline")
    mb = eval_arm_on_regime(provider, b, "strong_neg_corr")
    assert any(ma[g]["rmse"] != mb[g]["rmse"] for g in GREEKS)  # columns actually vary


def test_anchor_grid_params_are_bound_by_name(provider, tmp_path):
    """Q6: the anchor grid stored `params` as a bare 5-vector and eval_greeks
    zipped it against its own imported HESTON_PARAM_NAMES. Safe only while that
    import holds — and the grids are FROZEN artifacts that outlive it. The label
    artifact already carries an explicit param_names; the grids now match, and
    the binding is by name."""
    base = REGIME_PARAMS["near_feller"]
    ref = _fake_grid(tmp_path / "ref.npz", base, seed=3)
    m_ref = eval_arm_on_regime(provider, ref, "near_feller")

    # SAME regime, params and names permuted together -> identical metrics
    perm = [3, 0, 4, 1, 2]
    shuffled = _fake_grid(tmp_path / "perm.npz", [base[i] for i in perm], seed=3,
                          param_names=[HESTON_PARAM_NAMES[i] for i in perm])
    m_perm = eval_arm_on_regime(provider, shuffled, "near_feller")
    for g in GREEKS:
        assert m_perm[g]["rmse"] == pytest.approx(m_ref[g]["rmse"]), g

    # the writer emits it, so producer and consumer agree without a shared import
    import make_datasets
    assert "param_names" in make_datasets._ANCHOR_GRID_KEYS

    # a grid whose names are not the five Heston parameters is REFUSED, not
    # silently zipped into the wrong slots
    bad = _fake_grid(tmp_path / "bad.npz", base, seed=3,
                     param_names=["kappa", "theta", "xi", "rho", "sigma0"])
    with pytest.raises(ValueError, match="param_names"):
        eval_arm_on_regime(provider, bad, "near_feller")


# ---------------------------------------------------------------------------
# comparison + aggregation algebra (pure functions, hand-made values)
# ---------------------------------------------------------------------------

def test_reduction_improvement_parity_algebra():
    assert reduction_vs_baseline(0.8, 1.0) == pytest.approx(0.2)
    assert improvement_to_oracle_noise(0.8, 1.0, 0.5) == pytest.approx(0.4)
    assert price_parity(0.11, 0.10) == pytest.approx(0.1)
    # degenerate denominators -> NaN, never a crash
    assert not np.isfinite(reduction_vs_baseline(0.5, 0.0))
    assert not np.isfinite(improvement_to_oracle_noise(0.5, 1.0, 0.0))
    assert not np.isfinite(price_parity(0.1, 0.0))


def test_aggregate_seeds_matches_numpy():
    m, s = aggregate_seeds([1.0, 2.0, 3.0])
    assert (m, s) == pytest.approx((2.0, float(np.std([1.0, 2.0, 3.0], ddof=1))))
    assert aggregate_seeds([4.0]) == (4.0, 0.0)                 # single seed -> std 0
    m2, s2 = aggregate_seeds([np.nan, 2.0, 4.0])               # NaN entries dropped
    assert (m2, s2) == pytest.approx((3.0, float(np.std([2.0, 4.0], ddof=1))))
    assert all(np.isnan(x) for x in aggregate_seeds([np.nan, np.nan]))


# ---------------------------------------------------------------------------
# run_greek_eval: schemas, coverage, seed aggregation, threshold pre-check
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result(ckpt_root, tmp_path_factory):
    ad = tmp_path_factory.mktemp("eg_anchors")
    for name, params in REGIME_PARAMS.items():
        _fake_grid(ad / f"{name}_grid.npz", params, seed=hash(name) % 1000)
    out = tmp_path_factory.mktemp("eg_out")
    return run_greek_eval(CONTRACT, ckpt_root, ENGINE_ARMS, SEEDS, str(ad), str(out))


def _read_csv(path):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def test_csv_schemas_stable(result):
    for key, cols in (("ood_param_greeks", PERSEED_COLS),
                      ("sanity_in_domain", PERSEED_COLS),
                      ("ood_param_greeks_agg", AGG_COLS),
                      ("sanity_in_domain_agg", AGG_COLS),
                      ("threshold_precheck", THRESHOLD_COLS)):
        fields, _ = _read_csv(result["paths"][key])
        assert fields == cols, key


def test_primary_covers_regimes_arms_seeds_greeks(result):
    rows = result["primary"]
    assert result["primary_regimes"] == ["near_feller", "strong_neg_corr"]
    got = {(r["regime"], r["arm"], r["seed"], r["greek"]) for r in rows}
    expect = {(rg, arm, s, g) for rg in result["primary_regimes"]
              for arm in ENGINE_ARMS for s in SEEDS for g in GREEKS}
    assert got == expect
    # standard_pinn is its own baseline -> zero reduction / parity everywhere
    for r in rows:
        if r["arm"] == "standard_pinn":
            assert r["reduction_vs_standard_pinn"] == pytest.approx(0.0, abs=1e-12)
            assert r["price_parity"] == pytest.approx(0.0, abs=1e-12)


def test_sanity_has_wing_and_tau_slices(result):
    slices = {r["slice"] for r in result["sanity"]}
    assert slices == {"full", "wing", "tau"}
    assert {r["regime"] for r in result["sanity"]} == set(result["sanity_regimes"])
    # wing gamma rel_rmse is blank in the written CSV
    fields, rows = _read_csv(result["paths"]["sanity_in_domain"])
    wing_gamma = [r for r in rows if r["slice"] == "wing" and r["greek"] == "gamma"]
    assert wing_gamma and all(r["rel_rmse"] == "" for r in wing_gamma)


def test_seed_aggregation_matches_hand_computation(result):
    per, agg = result["primary"], result["primary_agg"]
    # pick one (regime, arm, greek) full-slice cell and recompute mean/std over seeds
    key = ("strong_neg_corr", "rung3", "gamma")
    vals = [r["rmse"] for r in per
            if (r["regime"], r["arm"], r["greek"]) == key and r["slice"] == "full"]
    assert len(vals) == len(SEEDS)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    arow = next(r for r in agg if (r["regime"], r["arm"], r["greek"], r["slice"])
                == (*key, "full"))
    assert arow["n_seeds"] == len(SEEDS)
    assert arow["rmse_mean"] == pytest.approx(mean)
    assert arow["rmse_std"] == pytest.approx(std)


def test_threshold_precheck_present_for_ladder_arms(result):
    rows = result["threshold"]
    got = {(r["regime"], r["arm"]) for r in rows}
    assert got == {(rg, arm) for rg in result["primary_regimes"]
                   for arm in ("rung2", "rung3")}
    for r in rows:                                              # booleans + a pass verdict
        assert isinstance(r["pass"], bool)
        # columns renamed off the contract's numbers in fix batch 3 ITEM 8(a):
        # gamma_ge_0.15 -> gamma_ge_min, vega_ge_0.15 -> vega_ge_min,
        # price_parity_within_0.10 -> price_parity_within_tol
        assert r["pass"] == (r["gamma_ge_min"] and r["vega_ge_min"]
                             and r["price_parity_within_tol"])


# ---------------------------------------------------------------------------
# hedging-box slice (CODE_AUDIT_2026-08-20 action 1)
# ---------------------------------------------------------------------------

def test_hedge_slice_spec_is_contract_derived():
    """Every bound of the hedging box comes from the contract, none is a literal."""
    import yaml as _yaml

    from eval_greeks import WING_HI, WING_LO, hedge_slice_spec
    cfg = _yaml.safe_load(open(CONTRACT))
    sp = hedge_slice_spec(cfg)
    hs = cfg["hedging_simulation"]
    assert sp["K"] == float(hs["instrument"]["K"])
    assert sp["tau_hi"] == pytest.approx(float(hs["instrument"]["tau0"]))
    assert sp["tau_lo"] == pytest.approx(float(hs["instrument"]["tau0"])
                                         - float(hs["horizon"]["T_prime"]))
    assert (sp["moneyness_lo"], sp["moneyness_hi"]) == (WING_LO, WING_HI)
    # the box is the maturities the hedge actually walks, inception -> liquidation
    assert 0.0 < sp["tau_lo"] < sp["tau_hi"]


def test_hedge_slice_mask_selects_exactly_the_hedging_box(tmp_path):
    """The mask is K == hedged strike, tau in [tau0-T', tau0], body moneyness only."""
    import yaml as _yaml

    from eval_greeks import _slice_masks, hedge_slice_spec
    cfg = _yaml.safe_load(open(CONTRACT))
    sp = hedge_slice_spec(cfg)
    # a grid that straddles the box on every axis
    S_ax = np.array([50.0, 80.0, 100.0, 120.0, 150.0])
    K_ax = np.array([60.0, sp["K"], 140.0])
    T_ax = np.array([0.04, sp["tau_lo"], 0.5 * (sp["tau_lo"] + sp["tau_hi"]),
                     sp["tau_hi"], 1.0])
    p = tmp_path / "g.npz"
    shape = (S_ax.size, K_ax.size, T_ax.size)
    arr = {"S_axis": S_ax, "K_axis": K_ax, "tau_axis": T_ax,
           "params": np.asarray(REGIME_PARAMS["baseline"], float),
           "param_names": np.array(HESTON_PARAM_NAMES),
           "mask_any": np.zeros(shape, bool),
           "r": np.float64(R), "q": np.float64(Q), "seed": np.int64(0)}
    for g in GREEKS:
        arr[f"consensus_{g}"] = np.ones(shape)
        arr[f"uncertainty_{g}"] = np.full(shape, 0.01)
        arr[f"mask_{g}"] = np.zeros(shape, bool)
    np.savez(p, **arr)

    m = _slice_masks(str(p), cfg)["hedge"].reshape(shape)
    Sg, Kg, Tg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")
    want = ((Kg == sp["K"]) & (Tg >= sp["tau_lo"] - 1e-12) & (Tg <= sp["tau_hi"] + 1e-12)
            & (Sg / Kg >= sp["moneyness_lo"]) & (Sg / Kg <= sp["moneyness_hi"]))
    assert m.any(), "hedging box must be non-empty on a grid that contains it"
    np.testing.assert_array_equal(m, want)
    # the three exclusions actually bite
    assert not m[:, 0, :].any() and not m[:, 2, :].any()      # wrong strike
    assert not m[:, :, 0].any() and not m[:, :, -1].any()     # tau outside the hedge's life
    assert not m[0, 1, :].any()                               # S/K = 0.5 is a wing point


def test_hedge_slice_snaps_strike_to_nearest_grid_node(tmp_path):
    """A grid whose K axis misses the hedged strike still yields a non-empty box."""
    import yaml as _yaml

    from eval_greeks import _slice_masks, hedge_slice_spec
    cfg = _yaml.safe_load(open(CONTRACT))
    sp = hedge_slice_spec(cfg)
    K_ax = np.array([sp["K"] - 3.0, sp["K"] + 7.0])          # nearest node is K-3
    S_ax = np.array([90.0, 100.0])
    T_ax = np.array([0.5 * (sp["tau_lo"] + sp["tau_hi"])])
    p = tmp_path / "g2.npz"
    shape = (S_ax.size, K_ax.size, T_ax.size)
    arr = {"S_axis": S_ax, "K_axis": K_ax, "tau_axis": T_ax,
           "params": np.asarray(REGIME_PARAMS["baseline"], float),
           "param_names": np.array(HESTON_PARAM_NAMES),
           "mask_any": np.zeros(shape, bool),
           "r": np.float64(R), "q": np.float64(Q), "seed": np.int64(0)}
    for g in GREEKS:
        arr[f"consensus_{g}"] = np.ones(shape)
        arr[f"uncertainty_{g}"] = np.full(shape, 0.01)
        arr[f"mask_{g}"] = np.zeros(shape, bool)
    np.savez(p, **arr)
    m = _slice_masks(str(p), cfg)["hedge"].reshape(shape)
    assert m.any()
    assert m[:, 1, :].sum() == 0                              # only the nearest node
    assert m[:, 0, :].any()


def test_hedge_slice_cannot_move_a_registered_verdict(result):
    """The registered OOD threshold reads slice == 'full' ONLY.

    The hedging box is reported alongside it; adding the slice must leave every
    threshold row bit-identical, or the audit's reporting fix would have silently
    become a re-scoring of a pre-registered endpoint.
    """
    from eval_greeks import _threshold_rows
    thresholds = {"ood_gamma_reduction_min": 0.15, "ood_vega_reduction_min": 0.15,
                  "price_parity_within": 0.10}
    agg = result["primary_agg"]
    assert any(r["slice"] == "hedge" for r in agg), "hedge slice missing from primary agg"
    with_box = _threshold_rows(agg, result["primary_regimes"], ENGINE_ARMS, thresholds)
    without = _threshold_rows([r for r in agg if r["slice"] != "hedge"],
                              result["primary_regimes"], ENGINE_ARMS, thresholds)
    assert with_box == without


def test_hedge_slice_is_reported_for_primary_regimes(result):
    """Both regions are emitted for every primary regime/arm/seed/greek."""
    slices = {r["slice"] for r in result["primary"]}
    assert slices == {"full", "hedge"}
    got = {(r["regime"], r["arm"], r["seed"], r["greek"])
           for r in result["primary"] if r["slice"] == "hedge"}
    assert got == {(rg, arm, s, g) for rg in result["primary_regimes"]
                   for arm in ENGINE_ARMS for s in SEEDS for g in GREEKS}
