"""Read selected paper comparisons from the published per-seed result tables.

Run ``python -m examples.paper_results`` from the repository root.
Uses the Python standard library and performs no training or simulation.
"""

import csv
from pathlib import Path
from statistics import mean, stdev


RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_rows(relative_path):
    with (RESULTS / relative_path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    accuracy = read_rows("eval_greeks_full/ood_param_greeks.csv")
    print("Cross-target transfer: variance-derivative RMSE, five seeds")
    for regime in ("near_feller", "strong_neg_corr"):
        errors = {}
        for arm in ("rung1", "rung2"):
            errors[arm] = mean(
                float(row["rmse"]) for row in accuracy
                if row["regime"] == regime and row["arm"] == arm
                and row["greek"] == "vega" and row["slice"] == "full"
            )
        reduction = 100 * (1 - errors["rung2"] / errors["rung1"])
        print(
            f"  {regime}: delta labels {errors['rung1']:.3f},"
            f" delta + gamma labels {errors['rung2']:.3f} ({reduction:.1f}% lower)"
        )

    print("\nZero-cost hedging under the confirmatory coefficient shift")
    print("Tail loss, mean +/- sample SD over ten seeds; lower is better")
    for path, residual_weight in (
        ("hedging_robustness/confirmatory/headline_delta_only_per_seed.csv", 0),
        ("hedging/confirmatory/headline_delta_only_per_seed.csv", 0.01),
    ):
        rows = read_rows(path)
        for method in ("standard_pinn", "rung3"):
            losses = [
                float(row["cvar"]) for row in rows
                if row["method"] == method and float(row["tc"]) == 0
                and row["in_model"] == "False"
            ]
            print(
                f"  residual weight {residual_weight:g}, {method:13s}:"
                f" {mean(losses):.4f} +/- {stdev(losses):.4f}"
            )
    print("\nThis zero-cost sensitivity analysis is separate from the registered 1% test.")
    print("The registered test fails. These seed summaries do not recreate pathwise CIs.")


if __name__ == "__main__":
    main()
