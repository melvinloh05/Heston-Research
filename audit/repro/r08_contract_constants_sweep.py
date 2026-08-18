"""R08 — repo-wide sweep: which modules read the contract, and which re-type its
numbers as Python literals.

Part 1: does each module open any YAML at all?
Part 2: every contract-quantity literal I could find in the Python, diffed
        numerically against heston_benchmark_v6.yaml / the engine supplements.
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path("/Users/melvin/Documents/Heston Research")
sys.path.insert(0, str(ROOT))

bm = yaml.safe_load((ROOT / "heston_benchmark_v6.yaml").read_text())
eng = yaml.safe_load((ROOT / "hedging_config.yaml").read_text())
pin = yaml.safe_load((ROOT / "pinn_config.yaml").read_text())

MODULES = ["oracle.py", "greek_labels.py", "make_labels.py", "make_datasets.py",
           "SobolevPINN.py", "ude.py", "train_pinn.py", "train.py",
           "Hedging_backtest.py", "providers.py", "pinn_provider.py",
           "run_hedging.py", "run_info_matching.py", "gate_headroom.py",
           "eval_greeks.py", "analyze_results.py", "exhibits.py"]

print("=" * 78)
print("PART 1 — does the module read a config file?")
print("=" * 78)
print(f"{'module':<24} {'reads contract':>15} {'reads engine yaml':>18} {'yaml import':>12}")
for m in MODULES:
    src = (ROOT / m).read_text()
    print(f"{m:<24} {str('heston_benchmark' in src):>15} "
          f"{str('hedging_config' in src or 'pinn_config' in src):>18} "
          f"{str(bool(re.search(r'^import yaml|^from .* import .*yaml', src, re.M))):>12}")

print()
print("=" * 78)
print("PART 2 — contract quantities re-typed as Python literals")
print("=" * 78)

rows = [
    # (where, quantity, code literal, contract value, contract path)
    ("analyze_results.py:53",  "cvar level",        0.95,  bm["metrics"]["cvar_convention"]["level"],
     "metrics.cvar_convention.level"),
    ("analyze_results.py:54",  "bootstrap B",       2000,  eng["risk"]["bootstrap_B"],
     "hedging_config risk.bootstrap_B"),
    ("analyze_results.py:55",  "global seed",       42,    bm["meta"]["global_seed"],
     "meta.global_seed"),
    ("analyze_results.py:50",  "confirmatory magnitude", 1.0, 1.0,
     "hedging_simulation.misspecification.perturbations.combined (m=1 endpoint)"),
    ("analyze_results.py:304", "confirm rel thresh", 0.10, 0.10,
     "acceptance_thresholds.confirmatory_cell_pass ('>=10% relative')"),
    ("analyze_results.py:301", "confirmatory tc",   0.01,  bm["hedging_simulation"]["confirmatory_cell"]["tc_tier"],
     "hedging_simulation.confirmatory_cell.tc_tier"),
    ("analyze_results.py:556", "ood gamma reduction", 0.15, bm["acceptance_thresholds"]["ood_gamma_rmse_reduction_min"],
     "acceptance_thresholds.ood_gamma_rmse_reduction_min"),
    ("analyze_results.py:556", "ood vega reduction", 0.15, bm["acceptance_thresholds"]["ood_vega_rmse_reduction_min"],
     "acceptance_thresholds.ood_vega_rmse_reduction_min"),
    ("analyze_results.py:556", "price parity tol",  0.10,  bm["acceptance_thresholds"]["price_parity_within"],
     "acceptance_thresholds.price_parity_within"),
    ("analyze_results.py:718", "tc tiers",          (0.0, 0.01, 0.02),
     tuple(bm["hedging_simulation"]["transaction_costs"]["tiers"]),
     "hedging_simulation.transaction_costs.tiers"),
    ("gate_headroom.py:361",   "gate spread thresh", 0.10, 0.10,
     "oracle_headroom_gate.decision_rule ('10% CVaR95 threshold')"),
    ("eval_greeks.py:288",     "ood gamma reduction", 0.15, bm["acceptance_thresholds"]["ood_gamma_rmse_reduction_min"],
     "acceptance_thresholds.ood_gamma_rmse_reduction_min"),
    ("eval_greeks.py:288",     "ood vega reduction", 0.15, bm["acceptance_thresholds"]["ood_vega_rmse_reduction_min"],
     "acceptance_thresholds.ood_vega_rmse_reduction_min"),
    ("eval_greeks.py:289",     "price parity tol",  0.10,  bm["acceptance_thresholds"]["price_parity_within"],
     "acceptance_thresholds.price_parity_within"),
    ("exhibits.py",            "vanna threshold line", 0.15, bm["acceptance_thresholds"]["ood_gamma_rmse_reduction_min"],
     "acceptance_thresholds (axhline 0.15 in E4 vanna inset)"),
    ("train_pinn.py:188",      "info-match cap",    5,     pin["info_matching"]["cap_multiplier"],
     "information_matching.cap (5N)"),
    ("train_pinn.py:341",      "feller_min fallback", 0.40, pin["hypercube_sampling"]["feller_min"],
     "training_parameterization.sampling.feller_constraint (0.40)"),
    ("train_pinn.py:342",      "excision radius fallback", 0.10,
     pin["hypercube_sampling"]["excision"]["rel_radius"],
     "training_parameterization.sampling.exclusions (10% relative radius)"),
    ("train_pinn.py:288",      "80/20 split fallback", 0.20, 0.20,
     "(no contract clause — declared in code/docstring only)"),
    ("run_info_matching.py:62", "PLATEAU_TOL",      0.02,  None,
     "(no contract clause — information_matching says only 'grow until plateau')"),
    ("analyze_results.py:622", "sakuma rel_tol",    0.02,  None,
     "(no contract clause — acceptance_thresholds.in_model_hedging = NOT_PASS_FAIL)"),
    ("analyze_results.py:450", "spearman p_max",    0.05,  None,
     "(no contract clause — dose_response says 'isotonic + rank correlation')"),
    ("eval_greeks.py:45",      "wing bounds",       (0.75, 1.30), None,
     "(no contract clause — moneyness_wing_holdout gives no numeric bounds)"),
    ("Hedging_backtest.py:49", "QE gamma1/gamma2",  (0.5, 0.5), None,
     "(scheme constant, not a contract quantity — Andersen central discretisation)"),
]

print(f"{'site':<26} {'quantity':<24} {'code':>18} {'contract':>18}  status")
n_dup = n_mis = n_nocontract = 0
for site, q, code, want, path in rows:
    if want is None:
        status = "NO CONTRACT VALUE"
        n_nocontract += 1
    elif code == want:
        status = "duplicate (equal)"
        n_dup += 1
    else:
        status = "*** MISMATCH ***"
        n_mis += 1
    print(f"{site:<26} {q:<24} {str(code):>18} {str(want):>18}  {status}")

print()
print(f"  duplicated-but-equal : {n_dup}")
print(f"  MISMATCHED           : {n_mis}")
print(f"  no contract value    : {n_nocontract}")

print()
print("=" * 78)
print("PART 3 — engine-derived quantities that ARE read from the contract (control group)")
print("=" * 78)
import Hedging_backtest as hb  # noqa: E402

cfg = hb.resolve_config()
b, e = cfg["benchmark"], cfg["engine"]
print(f"  tc tiers          <- bm.hedging_simulation.transaction_costs.tiers = "
      f"{b['hedging_simulation']['transaction_costs']['tiers']}")
print(f"  seeds             <- meta.global_seed + 0..seeds_min-1            = "
      f"{cfg['derived']['seeds']}")
print(f"  instrument        <- bm.hedging_simulation.instrument             = "
      f"{b['hedging_simulation']['instrument']}")
print(f"  train_params      <- bm.hedging_simulation.misspecification       = "
      f"{b['hedging_simulation']['misspecification']['train_params']}")
print(f"  oracle/baseline   <- hedging_config provider names                = "
      f"{e['oracle_provider_name']} / {e['baseline_provider_name']}")
print(f"  cvar level        <- hedging_config risk.cvar_level               = {e['risk']['cvar_level']}")
