"""Contract-constant parity (audit findings C1 / A1 / G1).

Every pre-registered numeric a verdict, gate or Greek-eval module acts on must BE
the contract's value, read from `heston_benchmark_v6.yaml` (+ the engine
supplement), not a Python literal re-typed alongside it. The audit found 19 such
literals across 5 modules with 0 drifted — this file is the guard that keeps the
count at 0 the next time a human edits the contract (the one file they are
*expected* to edit).

Three layers of assertion:
  1. `Hedging_backtest.contract_thresholds` returns exactly what the YAML says;
  2. each consumer resolves its defaults from that helper (no stale constants);
  3. each threshold is actually CONSUMED — the verdict flips at the contract's
     value, and an injected threshold changes the outcome.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest
import yaml

import Hedging_backtest as hb
import analyze_results as ar
import eval_greeks as eg
import gate_headroom as gh
import run_info_matching as rim

_DIR = Path(__file__).resolve().parent
CONTRACT = _DIR / "heston_benchmark_v6.yaml"
ENGINE = _DIR / "hedging_config.yaml"

BM = yaml.safe_load(CONTRACT.read_text())
ENG = yaml.safe_load(ENGINE.read_text())
CFG = hb.resolve_config(str(CONTRACT), str(ENGINE))
TH = hb.contract_thresholds(CFG)

_AT = BM["acceptance_thresholds"]
_HS = BM["hedging_simulation"]

# (key in contract_thresholds, value navigated independently out of the YAML)
_EXPECTED = [
    ("global_seed", BM["meta"]["global_seed"]),
    ("seeds_min", BM["meta"]["seeds_min"]),
    ("seeds_confirmatory_cell", BM["meta"]["seeds_confirmatory_cell"]),
    ("cvar_level", BM["metrics"]["cvar_convention"]["level"]),
    ("tc_tiers", tuple(float(t) for t in _HS["transaction_costs"]["tiers"])),
    ("confirmatory_tc", _HS["confirmatory_cell"]["tc_tier"]),
    ("confirmatory_direction", _HS["confirmatory_cell"]["perturbation"]),
    ("confirmatory_regime", _HS["confirmatory_cell"]["regime"]),
    ("ood_regimes", tuple(BM["splits"]["heldout_greek_and_hedging"])),
    ("ood_gamma_reduction_min", _AT["ood_gamma_rmse_reduction_min"]),
    ("ood_vega_reduction_min", _AT["ood_vega_rmse_reduction_min"]),
    ("price_parity_within", _AT["price_parity_within"]),
    ("sakuma_null_rel_tol", _AT["sakuma_null_rel_tol"]),
    ("dose_bootstrap_tail_prob_max", _AT["dose_response"]["bootstrap_tail_prob_max"]),
    ("gate_spread_threshold_rel", BM["oracle_headroom_gate"]["spread_threshold_rel"]),
    ("moneyness_wing_bounds",
     tuple(float(x) for x in BM["splits"]["moneyness_wing_holdout"]["moneyness_bounds"])),
    ("plateau_tol", BM["information_matching"]["plateau_tol"]),
    ("n_boot", ENG["risk"]["bootstrap_B"]),
    ("baseline_arm", ENG["baseline_provider_name"]),
    ("oracle_arm", ENG["oracle_provider_name"]),
]


@pytest.mark.parametrize("key,expected", _EXPECTED, ids=[k for k, _ in _EXPECTED])
def test_contract_thresholds_match_the_yaml(key, expected):
    assert TH[key] == expected


def test_cvar_level_is_one_number_across_contract_and_engine():
    """The engine supplement mirrors the contract's CVaR level; the helper serves
    the CONTRACT's (metrics.cvar_convention.level) and the two must not diverge."""
    assert TH["cvar_level"] == ENG["risk"]["cvar_level"]


def test_contract_thresholds_from_a_bare_contract_dict():
    """eval_greeks loads only the contract; the engine-half keys are then absent
    rather than silently defaulted."""
    bare = hb.contract_thresholds(BM)
    assert bare["ood_gamma_reduction_min"] == TH["ood_gamma_reduction_min"]
    assert "n_boot" not in bare and "baseline_arm" not in bare


# ---------------------------------------------------------------------------
# analyze_results — no module-level threshold literals; defaults come from the
# contract; the verdict flips AT the contract value
# ---------------------------------------------------------------------------

def test_analyze_results_defaults_are_the_contract():
    assert ar._default_thresholds() == TH
    for stale in ("DEFAULT_LEVEL", "DEFAULT_BOOT", "DEFAULT_SEED"):
        assert not hasattr(ar, stale), f"{stale} is a re-typed contract literal (C1)"


def test_analyze_results_confirmatory_cell_filter_is_the_contract_cell():
    assert ar._MISSPEC_FILTER["direction"] == TH["confirmatory_direction"]
    assert ar._MISSPEC_FILTER["magnitude"] == TH["confirmatory_magnitude"]
    assert ar._INMODEL_FILTER["direction"] == TH["confirmatory_direction"]
    assert ar._INMODEL_FILTER["magnitude"] == 0.0


def _write_agg_csv(path, gamma_red, vega_red, parity, arm="rung3"):
    cols = ["regime", "slice", "arm", "greek", "n_seeds",
            "reduction_vs_standard_pinn_mean", "reduction_vs_standard_pinn_std",
            "price_parity_mean", "price_parity_std"]
    rows = []
    for rg in TH["ood_regimes"]:
        for greek, red in (("gamma", gamma_red), ("vega", vega_red)):
            rows.append({"regime": rg, "slice": "full", "arm": arm, "greek": greek,
                         "n_seeds": 5, "reduction_vs_standard_pinn_mean": red,
                         "reduction_vs_standard_pinn_std": 0.01,
                         "price_parity_mean": parity, "price_parity_std": 0.01})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_ood_greek_verdict_flips_at_the_contract_reduction(tmp_path):
    gmin, vmin = TH["ood_gamma_reduction_min"], TH["ood_vega_reduction_min"]
    eps = 1e-6
    ok = _write_agg_csv(tmp_path / "ok.csv", gmin, vmin, 0.0)
    below = _write_agg_csv(tmp_path / "lo.csv", gmin - eps, vmin, 0.0)
    assert ar.ood_greek_thresholds(ok)["verdict"] == "pass"
    assert ar.ood_greek_thresholds(below)["verdict"] == "fail"
    # ... and the threshold is a real argument, not decoration
    assert ar.ood_greek_thresholds(below, gamma_red_threshold=0.0)["verdict"] == "pass"


def test_ood_greek_verdict_flips_at_the_contract_price_parity(tmp_path):
    tol = TH["price_parity_within"]
    ok = _write_agg_csv(tmp_path / "ok.csv", 0.9, 0.9, tol)
    out = _write_agg_csv(tmp_path / "out.csv", 0.9, 0.9, tol + 1e-6)
    assert ar.ood_greek_thresholds(ok)["verdict"] == "pass"
    assert ar.ood_greek_thresholds(out)["verdict"] == "fail"


# ---------------------------------------------------------------------------
# eval_greeks — the pre-check thresholds are the contract's and are consumed
# ---------------------------------------------------------------------------

def _eg_agg_rows(gamma_red, vega_red, parity, arm="rung3"):
    rows = []
    for rg in TH["ood_regimes"]:
        for greek, red in (("gamma", gamma_red), ("vega", vega_red)):
            rows.append({"regime": rg, "slice": "full", "arm": arm, "greek": greek,
                         "reduction_vs_standard_pinn_mean": red,
                         "reduction_vs_standard_pinn_std": 0.01,
                         "price_parity_mean": parity, "price_parity_std": 0.01})
    return rows


def test_eval_greeks_precheck_uses_the_contract_thresholds():
    gmin, vmin = TH["ood_gamma_reduction_min"], TH["ood_vega_reduction_min"]
    tol = TH["price_parity_within"]
    at = eg._threshold_rows(_eg_agg_rows(gmin, vmin, tol), TH["ood_regimes"],
                            ["rung3"], TH)
    assert at and all(r["pass"] for r in at)
    below = eg._threshold_rows(_eg_agg_rows(gmin - 1e-6, vmin, tol),
                               TH["ood_regimes"], ["rung3"], TH)
    assert not any(r["pass"] for r in below)
    parity_out = eg._threshold_rows(_eg_agg_rows(gmin, vmin, tol + 1e-6),
                                    TH["ood_regimes"], ["rung3"], TH)
    assert not any(r["pass"] for r in parity_out)


def test_eval_greeks_wing_bounds_match_the_contract():
    assert (eg.WING_LO, eg.WING_HI) == TH["moneyness_wing_bounds"]


# ---------------------------------------------------------------------------
# gate_headroom — the go/no-go threshold is the contract's and is consumed
# ---------------------------------------------------------------------------

def _gate_cfg(n_paths=96, n_boot=25, n_seeds=1):
    cfg = hb.resolve_config(str(CONTRACT), str(ENGINE))
    cfg["engine"]["simulation"]["n_paths"] = n_paths
    cfg["engine"]["risk"]["bootstrap_B"] = n_boot
    cfg["derived"]["seeds"] = cfg["derived"]["seeds"][:n_seeds]
    return cfg


def test_gate_decision_threshold_is_the_contract_value_and_is_consumed():
    """One smoke gate run, two thresholds: an unreachable one flags no decision
    row, a zero one flags every tier. The default equals the contract."""
    cfg = _gate_cfg()
    out = tempfile.mkdtemp()
    res = gh.run_gate(cfg, sigma_rel_list=(0.4,), mode="field", out_dir=out,
                      n_cloud_states=400, n_cloud_paths=96,
                      spread_threshold_rel=1e9)
    assert all(v is None for v in res["decision"].values())
    assert res["spread_threshold_rel"] == 1e9

    res0 = gh.run_gate(cfg, sigma_rel_list=(0.4,), mode="field", out_dir=out,
                       n_cloud_states=400, n_cloud_paths=96,
                       spread_threshold_rel=-1e9)
    assert any(v is not None for v in res0["decision"].values())

    res_default = gh.run_gate(cfg, sigma_rel_list=(0.4,), mode="field",
                              out_dir=out, n_cloud_states=400, n_cloud_paths=96)
    assert res_default["spread_threshold_rel"] == TH["gate_spread_threshold_rel"]
    assert f"{TH['gate_spread_threshold_rel']}" in Path(res_default["report_path"]).read_text()


# ---------------------------------------------------------------------------
# run_info_matching — the plateau tolerance is the contract's
# ---------------------------------------------------------------------------

def test_plateau_tol_matches_the_contract():
    assert rim.PLATEAU_TOL == TH["plateau_tol"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
