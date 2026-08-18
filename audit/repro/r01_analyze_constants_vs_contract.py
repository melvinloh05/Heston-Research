"""R01 — every pre-registered threshold in analyze_results.py is a Python literal.

Shows that analyze_results.py opens no contract YAML, and diffs each of its
in-code defaults against the value the contract actually carries.
"""
import inspect
import sys

import yaml

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
import analyze_results as ar  # noqa: E402

bm = yaml.safe_load(open("/Users/melvin/Documents/Heston Research/heston_benchmark_v6.yaml"))
eng = yaml.safe_load(open("/Users/melvin/Documents/Heston Research/hedging_config.yaml"))

src = inspect.getsource(ar)
print("analyze_results.py mentions 'heston_benchmark':", "heston_benchmark" in src)
print("analyze_results.py mentions 'hedging_config'  :", "hedging_config" in src)
print("analyze_results.py calls resolve_config       :", "resolve_config" in src)
print()

def d(fn, name):
    return inspect.signature(fn).parameters[name].default

checks = [
    ("cvar level",            ar.DEFAULT_LEVEL,                       bm["metrics"]["cvar_convention"]["level"]),
    ("bootstrap B",           ar.DEFAULT_BOOT,                        eng["risk"]["bootstrap_B"]),
    ("global seed",           ar.DEFAULT_SEED,                        bm["meta"]["global_seed"]),
    ("confirmatory rel thr",  d(ar.confirmatory_cell, "rel_threshold"), 0.10),   # contract prose ">=10% relative"
    ("confirmatory tc tier",  d(ar.confirmatory_cell, "tc"),          bm["hedging_simulation"]["confirmatory_cell"]["tc_tier"]),
    ("ood greek reduction",   d(ar.ood_greek_thresholds, "red_threshold"), bm["acceptance_thresholds"]["ood_gamma_rmse_reduction_min"]),
    ("ood vega reduction",    d(ar.ood_greek_thresholds, "red_threshold"), bm["acceptance_thresholds"]["ood_vega_rmse_reduction_min"]),
    ("price parity tol",      d(ar.ood_greek_thresholds, "parity_tol"), bm["acceptance_thresholds"]["price_parity_within"]),
    ("tc sweep tiers",        tuple(d(ar.mechanism_adjudication, "tcs")), tuple(bm["hedging_simulation"]["transaction_costs"]["tiers"])),
    ("ood regimes",           tuple(d(ar.ood_greek_thresholds, "regimes")), tuple(bm["splits"]["heldout_greek_and_hedging"])),
    ("baseline arm name",     d(ar.confirmatory_cell, "baseline"),    eng["baseline_provider_name"]),
]
print(f"{'quantity':<24} {'analyze_results.py':>20} {'contract/engine':>20}  match")
for name, got, want in checks:
    print(f"{name:<24} {str(got):>20} {str(want):>20}  {'OK' if got == want else 'MISMATCH'}")

print()
print("--- thresholds with NO contract counterpart at all (invented in Python) ---")
print("sakuma_null_consistency rel_tol =", d(ar.sakuma_null_consistency, "rel_tol"),
      "-> contract acceptance_thresholds.in_model_hedging =",
      bm["acceptance_thresholds"]["in_model_hedging"])
print("dose_response spearman_p_max   =", d(ar.dose_response, "spearman_p_max"),
      "-> contract dose_response text =", repr(bm["acceptance_thresholds"]["dose_response"]))
