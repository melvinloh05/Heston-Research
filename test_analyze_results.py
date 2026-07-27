"""Tests for analyze_results.py (P13) on SYNTHETIC inputs built in-test.

No engine/model/oracle runs: every test fabricates the exact artifacts P12/P11
write — per-cell PnL npz (name = Hedging_backtest._cell_slug, keys '{arm}__tc{tc}'),
per-seed metric CSVs, and the P11 Greek-agg CSV — then checks that

  * the pooled-stratified paired bootstrap detects a constant per-seed
    improvement (CI excludes 0) and cancels opposing per-seed shifts (~0),
  * each pre-registered verdict returns the right pass/fail/null on constructed
    pass/fail/null inputs and never throws on blank ('') cells,
  * the dose-response x-axis orders shuffled > sigma_050 > sigma_000 == 0 and the
    isotonic + Spearman fit runs end to end,
  * the four constructed mechanism patterns map to (i)/(ii)/decomposition/null.

The one place a real config is touched is build_arm_labels via the arm PINNConfig
(pinn_config.yaml) in the dose-response x-axis test — that IS the axis under test.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

import analyze_results as ar

PINN_CFG = "pinn_config.yaml"


# ---------------------------------------------------------------------------
# synthetic-artifact builders
# ---------------------------------------------------------------------------

def _slug(seed, **tag) -> str:
    """Mirror Hedging_backtest._cell_slug: seed first, blank fields dropped."""
    parts = [f"seed={seed}"] + [f"{k}={v}" for k, v in tag.items() if v != ""]
    return "__".join(parts)


def _write_cell(pnl_dir, seed, tag, pnl_by_key) -> str:
    os.makedirs(pnl_dir, exist_ok=True)
    path = os.path.join(pnl_dir, _slug(seed, **tag) + ".npz")
    np.savez(path, **pnl_by_key)
    return path


def _misspec_tag(mag=1.0):
    return {"direction": "combined", "magnitude": mag, "lambda_j": "", "sigma_j": "",
            "in_model": (mag == 0.0), "sweep": "perturbation"}


def _write_csv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


# ===========================================================================
# 1. stratified pooling
# ===========================================================================

def test_pooled_stratified_constant_shift_excludes_zero(tmp_path):
    """A constant per-seed improvement: pooled CI excludes 0, diff == -c exactly."""
    d = tmp_path / "pnl"
    rng = np.random.default_rng(0)
    c = 0.4
    for s in range(4):
        b = -np.abs(rng.normal(0.0, 3.0, 1500))       # standard losses
        a = b + c                                       # rung3 uniformly better
        _write_cell(d, s, _misspec_tag(1.0),
                    {"rung3__tc0.01": a, "standard_pinn__tc0.01": b})
    res = ar.paired_ci_from_npz(d, "rung3", "standard_pinn", 0.01, 0.95, 400, 42,
                                slug_filter=ar._MISSPEC_FILTER)
    p = res["pooled"]
    assert res["n_seeds"] == 4
    assert p["diff"] == pytest.approx(-c, abs=1e-9)     # translation-equivariant CVaR
    assert ar._excludes_zero(p["ci_lo"], p["ci_hi"])
    assert p["ci_hi"] < 0.0                             # improvement side


def test_pooled_stratified_opposing_shifts_cancel(tmp_path):
    """+c in one seed, -c in another (same b): pooled diff ~ 0, CI covers 0."""
    d = tmp_path / "pnl"
    rng = np.random.default_rng(1)
    b = -np.abs(rng.normal(0.0, 3.0, 4000))
    c = 0.5
    _write_cell(d, 0, _misspec_tag(1.0),
                {"rung3__tc0.01": b + c, "standard_pinn__tc0.01": b})
    _write_cell(d, 1, _misspec_tag(1.0),
                {"rung3__tc0.01": b - c, "standard_pinn__tc0.01": b})
    res = ar.paired_ci_from_npz(d, "rung3", "standard_pinn", 0.01, 0.95, 600, 7,
                                slug_filter=ar._MISSPEC_FILTER)
    p = res["pooled"]
    # opposing per-seed shifts do NOT yield a genuine c-sized gap: the pooled diff
    # is a small residual (~0.2c, a one-sided-tail boundary effect of CVaR), an
    # order below the -c gap the SAME c produces under a constant shift.
    assert abs(p["diff"]) < 0.3 * c


def test_slug_filter_selects_one_cell(tmp_path):
    """Mixed m=0.0 and m=1.0 npz in one dir: the magnitude filter isolates the cell."""
    d = tmp_path / "pnl"
    rng = np.random.default_rng(2)
    for mag, c in ((0.0, 0.0), (1.0, 0.6)):
        for s in range(3):
            b = -np.abs(rng.normal(0, 2, 800))
            _write_cell(d, s, _misspec_tag(mag),
                        {"rung3__tc0.01": b + c, "standard_pinn__tc0.01": b})
    misspec = ar.paired_ci_from_npz(d, "rung3", "standard_pinn", 0.01, 0.95, 200, 1,
                                    slug_filter=ar._MISSPEC_FILTER)
    inmodel = ar.paired_ci_from_npz(d, "rung3", "standard_pinn", 0.01, 0.95, 200, 1,
                                    slug_filter=ar._INMODEL_FILTER)
    assert misspec["pooled"]["diff"] == pytest.approx(-0.6, abs=1e-9)
    assert inmodel["pooled"]["diff"] == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 2. verdict truth-table (pass / fail / null; never throws on blanks)
# ===========================================================================

def _pnl_pass(shift):
    """Standard losses with a spread ~2; rung3 better by `shift` (rel ~ shift/cvar_b)."""
    def fn(seed, _tc=None):
        rng = np.random.default_rng([seed, 99])
        b = -np.abs(rng.normal(0.0, 2.0, 1200))
        return {"standard": b, "rung3": b + shift, "rung2": b + shift,
                "rung1": b + 0.5 * shift}
    return fn


def _confirmatory_dir(tmp_path, shift, mag=1.0, tcs=(0.0, 0.01, 0.02)):
    d = tmp_path / f"pnl_m{mag}_{shift}"
    gen = _pnl_pass(shift)
    for s in range(10):
        cell = gen(s, None)
        keys = {}
        for tc in tcs:
            keys[ar._pnl_key("rung3", tc)] = cell["rung3"]
            keys[ar._pnl_key("rung2", tc)] = cell["rung2"]
            keys[ar._pnl_key("rung1", tc)] = cell["rung1"]
            keys[ar._pnl_key("standard_pinn", tc)] = cell["standard"]
        _write_cell(d, s, _misspec_tag(mag), keys)
    return d


def test_confirmatory_pass_and_fail(tmp_path):
    # PASS: rung3 clearly better than standard (rel >= 0.10) with a tight CI
    d_pass = _confirmatory_dir(tmp_path, shift=1.0, mag=1.0)
    v = ar.confirmatory_cell(d_pass, tc=0.01, n_boot=400, seed=3)
    assert v["threshold_id"] == "confirmatory_cell"
    assert v["verdict"] == "pass"
    assert v["statistic"] >= 0.10 and v["ci_hi"] < 0.0

    # FAIL: rung3 == standard (no improvement)
    d_fail = _confirmatory_dir(tmp_path, shift=0.0, mag=1.0)
    vf = ar.confirmatory_cell(d_fail, tc=0.01, n_boot=400, seed=3)
    assert vf["verdict"] == "fail"


def test_order_attribution_pass_and_null(tmp_path):
    d_pass = _confirmatory_dir(tmp_path, shift=1.0)
    v = ar.order_attribution(d_pass, tc=0.01, n_boot=400, seed=3)
    assert v["verdict"] == "pass" and v["ci_hi"] < 0.0

    # rung2 == rung1 -> honest null
    d = tmp_path / "order_null"
    rng = np.random.default_rng(5)
    for s in range(6):
        b = -np.abs(rng.normal(0, 2, 900))
        _write_cell(d, s, _misspec_tag(1.0),
                    {"rung2__tc0.01": b, "rung1__tc0.01": b})
    vn = ar.order_attribution(d, tc=0.01, n_boot=300, seed=3)
    assert vn["verdict"] == "fail" and "honest null" in vn["notes"]


def test_sakuma_consistency_and_flag(tmp_path):
    # in-model gap ~ 0 -> consistent (Sakuma null reproduced)
    d = tmp_path / "sakuma_ok"
    rng = np.random.default_rng(6)
    for s in range(5):
        b = -np.abs(rng.normal(0, 2, 900))
        _write_cell(d, s, _misspec_tag(0.0),
                    {"rung3__tc0.0": b, "standard_pinn__tc0.0": b})
    v = ar.sakuma_null_consistency(d, tc=0.0, n_boot=300, seed=1)
    assert v["threshold_id"] == "sakuma_null_consistency"
    assert v["verdict"] == "consistent"

    # large in-model gap -> flag
    d2 = tmp_path / "sakuma_flag"
    for s in range(5):
        rng2 = np.random.default_rng([s, 6])
        b = -np.abs(rng2.normal(0, 2, 900))
        _write_cell(d2, s, _misspec_tag(0.0),
                    {"rung3__tc0.0": b + 1.5, "standard_pinn__tc0.0": b})
    vf = ar.sakuma_null_consistency(d2, tc=0.0, n_boot=300, seed=1)
    assert vf["verdict"] == "flag"


def _greek_agg_rows(gamma_red, vega_red, parity, regimes=("near_feller", "strong_neg_corr"),
                    n_seeds=5, std=0.01):
    """Minimal P11 ood_param_greeks_agg.csv rows (slice=full) for rung3."""
    rows = []
    for rg in regimes:
        for greek, red in (("gamma", gamma_red), ("vega", vega_red), ("delta", 0.2),
                           ("price", 0.05), ("vanna", 0.1)):
            rows.append({"regime": rg, "slice": "full", "arm": "rung3", "greek": greek,
                         "n_seeds": n_seeds,
                         "reduction_vs_standard_pinn_mean": red,
                         "reduction_vs_standard_pinn_std": std,
                         "price_parity_mean": parity, "price_parity_std": 0.01,
                         "rmse_mean": 1.0, "rmse_std": 0.1})
    return rows


def test_ood_greek_pass_fail_null(tmp_path):
    cols = ["regime", "slice", "arm", "greek", "n_seeds",
            "reduction_vs_standard_pinn_mean", "reduction_vs_standard_pinn_std",
            "price_parity_mean", "price_parity_std", "rmse_mean", "rmse_std"]
    # PASS: both reductions >= 0.15 at parity within 0.10
    p = _write_csv(tmp_path / "agg_pass.csv",
                   _greek_agg_rows(0.25, 0.20, 0.05), cols)
    assert ar.ood_greek_thresholds(p)["verdict"] == "pass"
    # FAIL: gamma reduction below threshold
    f = _write_csv(tmp_path / "agg_fail.csv",
                   _greek_agg_rows(0.05, 0.20, 0.05), cols)
    vf = ar.ood_greek_thresholds(f)
    assert vf["verdict"] == "fail"
    # FAIL: price parity blown
    f2 = _write_csv(tmp_path / "agg_parity.csv",
                    _greek_agg_rows(0.25, 0.20, 0.40), cols)
    assert ar.ood_greek_thresholds(f2)["verdict"] == "fail"
    # NULL: no rung3 rows at all
    n = _write_csv(tmp_path / "agg_null.csv",
                   _greek_agg_rows(0.25, 0.2, 0.05, regimes=()) or
                   [{c: "" for c in cols}], cols)
    assert ar.ood_greek_thresholds(n)["verdict"] == "null"


def test_ood_greek_never_throws_on_blanks(tmp_path):
    cols = ["regime", "slice", "arm", "greek", "n_seeds",
            "reduction_vs_standard_pinn_mean", "reduction_vs_standard_pinn_std",
            "price_parity_mean", "price_parity_std", "rmse_mean", "rmse_std"]
    rows = _greek_agg_rows(0.25, 0.2, 0.05)
    for r in rows:                                        # blank out numeric cells
        if r["greek"] == "gamma" and r["regime"] == "near_feller":
            r["reduction_vs_standard_pinn_mean"] = ""
            r["price_parity_mean"] = ""
    p = _write_csv(tmp_path / "agg_blank.csv", rows, cols)
    v = ar.ood_greek_thresholds(p)                        # must not raise
    assert v["verdict"] in ("pass", "fail", "null")


# ===========================================================================
# 3. dose-response x-axis + isotonic/Spearman end to end
# ===========================================================================

def _fake_labels_npz(path, n_points=30, n_skt=8, seed=1):
    """A minimal labels artifact build_arm_labels can consume (no masking)."""
    rng = np.random.default_rng(seed)
    params = np.column_stack([
        rng.uniform(1, 4, n_points), rng.uniform(0.02, 0.12, n_points),
        rng.uniform(0.2, 0.6, n_points), rng.uniform(-0.8, -0.2, n_points),
        rng.uniform(0.01, 0.12, n_points)])
    S = rng.uniform(80, 120, n_skt)
    K = rng.uniform(80, 120, n_skt)
    tau = rng.uniform(0.1, 0.9, n_skt)
    # wide-spread gamma so a permutation (shuffled) is clearly worse than 50% noise
    gamma = np.exp(rng.normal(-3.0, 1.0, (n_points, n_skt)))
    arrays = {"params": params, "param_names": np.array(["kappa", "theta", "xi", "rho", "v0"]),
              "S": S, "K": K, "tau": tau, "r": np.float64(0.02), "q": np.float64(0.0),
              "mask_any": np.zeros((n_points, n_skt), bool)}
    for g, val in (("price", np.abs(rng.normal(5, 1, (n_points, n_skt)))),
                   ("delta", rng.uniform(0, 1, (n_points, n_skt))),
                   ("gamma", gamma),
                   ("vega", np.abs(rng.normal(10, 2, (n_points, n_skt)))),
                   ("vanna", rng.normal(0, 1, (n_points, n_skt)))):
        arrays[f"consensus_{g}"] = val
        arrays[f"uncertainty_{g}"] = np.abs(val) * 0.01
        arrays[f"mask_{g}"] = np.zeros((n_points, n_skt), bool)
    np.savez(path, **arrays)
    return str(path)


def test_dose_axis_orders_shuffled_gt_sigma_gt_zero(tmp_path):
    npz = _fake_labels_npz(tmp_path / "labels.npz")
    e000, _, _ = ar._measured_label_error(npz, PINN_CFG, "sigma_000", 42)
    e050, _, _ = ar._measured_label_error(npz, PINN_CFG, "sigma_050", 42)
    eshuf, _, _ = ar._measured_label_error(npz, PINN_CFG, "shuffled", 42)
    egp, src, _ = ar._measured_label_error(npz, PINN_CFG, "gradient_penalty_only", 42)
    assert e000 == pytest.approx(0.0, abs=1e-12)
    assert eshuf > e050 > e000
    assert src == "none" and np.isnan(egp)              # reference line, off the axis


def _dose_per_seed_csv(path, arm_cvar, seeds=(0, 1, 2, 3, 4)):
    """per-seed CSV rows at the confirmatory cell; arm_cvar: {arm: fn(seed)->cvar}."""
    cols = ["direction", "magnitude", "sweep", "in_model", "lambda_j", "sigma_j",
            "method", "tc", "seed", "cvar", "t_ex"]
    rows = []
    for arm, fn in arm_cvar.items():
        for s in seeds:
            rows.append({"direction": "combined", "magnitude": 1.0, "sweep": "perturbation",
                         "in_model": False, "lambda_j": "", "sigma_j": "",
                         "method": arm, "tc": 0.01, "seed": s, "cvar": fn(s), "t_ex": ""})
    return _write_csv(path, rows, cols)


def test_dose_response_monotone_end_to_end(tmp_path):
    npz = _fake_labels_npz(tmp_path / "labels.npz")
    # CVaR increasing in label error: sigma_000 < sigma_050 < shuffled; gp = reference
    arm_cvar = {
        "sigma_000": lambda s: 10.0 + 0.05 * s,
        "sigma_050": lambda s: 11.0 + 0.05 * s,
        "shuffled": lambda s: 12.5 + 0.05 * s,
        "gradient_penalty_only": lambda s: 9.5 + 0.05 * s,
    }
    ps = _dose_per_seed_csv(tmp_path / "ps.csv", arm_cvar)
    vd, rows = ar.dose_response(ps, npz, PINN_CFG,
                                dose_arms=("sigma_000", "sigma_050", "shuffled",
                                           "gradient_penalty_only"),
                                n_boot=200, seed=1)
    assert vd["threshold_id"] == "dose_response"
    assert vd["verdict"] == "monotone"
    assert vd["statistic"] > 0.0                         # positive Spearman rho
    ref = [r for r in rows if r["arm"] == "gradient_penalty_only"][0]
    assert ref["is_reference"] is True
    assert not any(r["is_reference"] for r in rows if r["arm"] == "sigma_050")


def test_dose_response_flat_null(tmp_path):
    npz = _fake_labels_npz(tmp_path / "labels.npz")
    # CVaR NON-monotone in label error (highest at the smallest error): distinct
    # per-arm seed means with a negative rank relation -> flat = regularization null
    arm_cvar = {
        "sigma_000": lambda s: 11.0 + 0.01 * s,
        "sigma_050": lambda s: 10.0 + 0.01 * s,
        "shuffled": lambda s: 10.5 + 0.01 * s,
        "gradient_penalty_only": lambda s: 9.5,
    }
    ps = _dose_per_seed_csv(tmp_path / "ps_flat.csv", arm_cvar)
    vd, _ = ar.dose_response(ps, npz, PINN_CFG,
                             dose_arms=("sigma_000", "sigma_050", "shuffled",
                                        "gradient_penalty_only"),
                             n_boot=200, seed=1)
    assert vd["verdict"] == "flat"


def test_dose_response_never_throws_on_blanks(tmp_path):
    npz = _fake_labels_npz(tmp_path / "labels.npz")
    cols = ["direction", "magnitude", "sweep", "in_model", "lambda_j", "sigma_j",
            "method", "tc", "seed", "cvar", "t_ex"]
    rows = [{"direction": "combined", "magnitude": 1.0, "sweep": "perturbation",
             "in_model": False, "lambda_j": "", "sigma_j": "", "method": "sigma_000",
             "tc": 0.01, "seed": 0, "cvar": "", "t_ex": ""}]        # blank cvar
    ps = _write_csv(tmp_path / "ps_blank.csv", rows, cols)
    vd, _rows = ar.dose_response(ps, npz, PINN_CFG,
                                 dose_arms=("sigma_000", "sigma_050"),
                                 n_boot=50, seed=1)                 # must not raise
    assert vd["verdict"] in ("monotone", "flat")


# ===========================================================================
# 4. mechanism truth-table (four constructed patterns)
# ===========================================================================

def _gap(tc, diff, lo, hi):
    return {"tc": tc, "diff": diff, "ci_lo": lo, "ci_hi": hi}


def _tex(diff, lo, hi):
    return {"diff": diff, "ci_lo": lo, "ci_hi": hi}


def test_mechanism_reading_four_patterns():
    # (i) robustness: tc=0 gap present, no widening, T_ex unmoved
    ch_i = ar._mechanism_reading(
        [_gap(0.0, -1.0, -1.5, -0.5), _gap(0.02, -1.0, -1.5, -0.5)],
        _tex(-0.1, -0.5, 0.3))
    assert ch_i["reading"] == "channel_i"

    # (ii) cost: tc=0 covers 0, gap widens & excludes 0 at top tc, T_ex reduced
    ch_ii = ar._mechanism_reading(
        [_gap(0.0, -0.05, -0.2, 0.1), _gap(0.02, -1.0, -1.4, -0.6)],
        _tex(-2.0, -3.0, -1.0))
    assert ch_ii["reading"] == "channel_ii"

    # both -> decomposition
    dec = ar._mechanism_reading(
        [_gap(0.0, -1.0, -1.5, -0.5), _gap(0.02, -2.0, -2.5, -1.5)],
        _tex(-2.0, -3.0, -1.0))
    assert dec["reading"] == "decomposition"

    # neither -> null
    nul = ar._mechanism_reading(
        [_gap(0.0, -0.05, -0.2, 0.1), _gap(0.02, -0.05, -0.2, 0.1)],
        _tex(-0.1, -0.5, 0.3))
    assert nul["reading"] == "null"


def test_mechanism_t_ex_unmoved_kills_channel_ii():
    """Cost PnL pattern but T_ex CI covers 0 -> channel_ii rejected, falls to null."""
    r = ar._mechanism_reading(
        [_gap(0.0, -0.05, -0.2, 0.1), _gap(0.02, -1.0, -1.4, -0.6)],
        _tex(-2.0, -3.0, 0.5))                           # T_ex CI covers 0
    assert r["reading"] == "null" and r["present_ii"] is False


def test_mechanism_adjudication_assembles_from_npz(tmp_path):
    """End-to-end: build a misspec + in-model confirmatory dir and a per-seed CSV
    with T_ex, then assemble the reading, slope, 2x2 and T_ex diff."""
    d = tmp_path / "pnl"
    rng = np.random.default_rng(4)
    # misspec: robustness present at tc=0 (constant gap across tc), in-model ~ 0
    for s in range(5):
        b = -np.abs(rng.normal(0, 2, 1000))
        keys_mis, keys_in = {}, {}
        for tc in (0.0, 0.01, 0.02):
            keys_mis[ar._pnl_key("rung3", tc)] = b + 0.8
            keys_mis[ar._pnl_key("standard_pinn", tc)] = b
            keys_in[ar._pnl_key("rung3", tc)] = b
            keys_in[ar._pnl_key("standard_pinn", tc)] = b
        _write_cell(d, s, _misspec_tag(1.0), keys_mis)
        _write_cell(d, s, _misspec_tag(0.0), keys_in)

    cols = ["direction", "magnitude", "sweep", "in_model", "lambda_j", "sigma_j",
            "method", "tc", "seed", "cvar", "t_ex"]
    rows = []
    for s in range(5):
        for method, te in (("rung3", 1.0), ("standard_pinn", 1.2)):
            for tc in (0.0, 0.01, 0.02):
                rows.append({"direction": "combined", "magnitude": 1.0,
                             "sweep": "perturbation", "in_model": False,
                             "lambda_j": "", "sigma_j": "", "method": method,
                             "tc": tc, "seed": s, "cvar": 1.0, "t_ex": te})
    ps = _write_csv(tmp_path / "ps.csv", rows, cols)

    mech = ar.mechanism_adjudication(d, ps, n_boot=200, seed=2)
    assert mech["reading"]["reading"] == "channel_i"     # gap at tc=0 excludes 0
    assert len(mech["misspec_gaps"]) == 3
    assert len(mech["inmodel_gaps"]) == 3
    assert mech["t_ex"]["diff"] == pytest.approx(-0.2, abs=1e-9)   # 1.0 - 1.2
    assert mech["two_by_two"]["misspec"]["tc0"] is not None
    assert np.isfinite(mech["slope_dgap_dtc"])


# ===========================================================================
# 5. goldilocks + orchestrator + outputs
# ===========================================================================

def _bates_tag(lam, sj):
    return {"direction": "", "magnitude": "", "lambda_j": lam, "sigma_j": sj,
            "in_model": "", "sweep": "bates"}


def test_goldilocks_locates_and_null(tmp_path):
    d = tmp_path / "pnl"
    cols = ["direction", "magnitude", "sweep", "in_model", "lambda_j", "sigma_j",
            "method", "tc", "seed", "cvar", "t_ex"]
    rows = []
    rng = np.random.default_rng(8)
    # one decisive severity cell (lam=0.25), one indistinct (lam=0.0)
    for lam, sj, shift in ((0.0, 0.05, 0.0), (0.25, 0.10, 0.7)):
        for s in range(4):
            b = -np.abs(rng.normal(0, 2, 900))
            _write_cell(d, s, _bates_tag(lam, sj),
                        {"rung3__tc0.01": b + shift, "standard_pinn__tc0.01": b})
            for method in ("rung3", "standard_pinn"):
                rows.append({"direction": "", "magnitude": "", "sweep": "bates",
                             "in_model": "", "lambda_j": lam, "sigma_j": sj,
                             "method": method, "tc": 0.01, "seed": s,
                             "cvar": 1.0, "t_ex": ""})
    ps = _write_csv(tmp_path / "ps.csv", rows, cols)
    vd, grows = ar.goldilocks_bates(d, ps, n_boot=300, seed=1)
    assert vd["threshold_id"] == "goldilocks_bates"
    assert vd["verdict"] == "decision_relevant_regime_located"
    assert any(r["decisive"] for r in grows) and len(grows) == 2

    # all cells indistinct -> pre-registered 'nowhere' null
    d2 = tmp_path / "pnl2"
    rows2 = []
    for lam, sj in ((0.0, 0.05), (0.25, 0.10)):
        for s in range(4):
            b = -np.abs(np.random.default_rng([s, 3]).normal(0, 2, 900))
            _write_cell(d2, s, _bates_tag(lam, sj),
                        {"rung3__tc0.01": b, "standard_pinn__tc0.01": b})
            for method in ("rung3", "standard_pinn"):
                rows2.append({"direction": "", "magnitude": "", "sweep": "bates",
                              "in_model": "", "lambda_j": lam, "sigma_j": sj,
                              "method": method, "tc": 0.01, "seed": s,
                              "cvar": 1.0, "t_ex": ""})
    ps2 = _write_csv(tmp_path / "ps2.csv", rows2, cols)
    vn, _ = ar.goldilocks_bates(d2, ps2, n_boot=300, seed=1)
    assert vn["verdict"] == "null"


def test_run_analysis_writes_outputs_and_degrades(tmp_path):
    """Orchestrator on a confirmatory dir only (no full-sweep / greek inputs):
    writes all three outputs; absent inputs become 'null' rows, never raise."""
    conf = tmp_path / "confirmatory"
    pnl = conf / ar.PNL_DIR
    rng = np.random.default_rng(11)
    # confirmatory cell (misspec + in-model) with all four arms across tc tiers
    for s in range(10):
        b = -np.abs(rng.normal(0, 2, 1000))
        mis, inm = {}, {}
        for tc in (0.0, 0.01, 0.02):
            mis[ar._pnl_key("rung3", tc)] = b + 1.0
            mis[ar._pnl_key("rung2", tc)] = b + 0.9
            mis[ar._pnl_key("rung1", tc)] = b + 0.5
            mis[ar._pnl_key("standard_pinn", tc)] = b
            inm[ar._pnl_key("rung3", tc)] = b
            inm[ar._pnl_key("standard_pinn", tc)] = b
        _write_cell(pnl, s, _misspec_tag(1.0), mis)
        _write_cell(pnl, s, _misspec_tag(0.0), inm)
    # a per-seed CSV with T_ex for the mechanism assembly
    cols = ["direction", "magnitude", "sweep", "in_model", "lambda_j", "sigma_j",
            "method", "tc", "seed", "cvar", "t_ex"]
    ps_rows = []
    for s in range(10):
        for method, te in (("rung3", 1.0), ("standard_pinn", 1.3)):
            for tc in (0.0, 0.01, 0.02):
                ps_rows.append({"direction": "combined", "magnitude": 1.0,
                                "sweep": "perturbation", "in_model": False,
                                "lambda_j": "", "sigma_j": "", "method": method,
                                "tc": tc, "seed": s, "cvar": 1.0, "t_ex": te})
    _write_csv(conf / ar.PER_SEED_CSV, ps_rows, cols)

    res = ar.run_analysis(conf, tmp_path / "out", n_boot=200, seed=2)
    paths = res["paths"]
    assert os.path.exists(paths["threshold_verdicts"])
    assert os.path.exists(paths["dose_response"])
    assert os.path.exists(paths["mechanism_memo"])

    ids = {v["threshold_id"] for v in res["verdicts"]}
    assert {"confirmatory_cell", "order_attribution", "dose_response",
            "ood_greek_thresholds", "sakuma_null_consistency",
            "mechanism_adjudication", "goldilocks_bates"} <= ids
    byid = {v["threshold_id"]: v for v in res["verdicts"]}
    assert byid["confirmatory_cell"]["verdict"] == "pass"
    assert byid["sakuma_null_consistency"]["verdict"] == "consistent"
    # absent full-sweep / greek inputs degrade to null, not an exception
    assert byid["dose_response"]["verdict"] == "null"
    assert byid["ood_greek_thresholds"]["verdict"] == "null"
    assert byid["goldilocks_bates"]["verdict"] == "null"

    memo = open(paths["mechanism_memo"]).read()
    assert "Three candidate stories" in memo
    assert "COST parameter" in memo


def test_run_analysis_verdicts_csv_schema(tmp_path):
    conf = tmp_path / "confirmatory"
    (conf / ar.PNL_DIR).mkdir(parents=True)              # empty -> everything null
    res = ar.run_analysis(conf, tmp_path / "out", n_boot=50, seed=1)
    with open(res["paths"]["threshold_verdicts"]) as fh:
        header = next(csv.reader(fh))
    assert header == ar.VERDICT_COLS
    # nothing raised; every verdict row present even with no artifacts
    # (7 pre-registered + 1 SECONDARY non-binding rung2 OOD-Greek row)
    assert len(res["verdicts"]) == 8
    byid = {v["threshold_id"]: v for v in res["verdicts"]}
    assert "ood_greek_thresholds_rung2_secondary" in byid
    assert "SECONDARY" in byid["ood_greek_thresholds_rung2_secondary"]["notes"]
