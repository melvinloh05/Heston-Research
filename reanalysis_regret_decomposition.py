"""T1 -- regret split: approximation error vs control-design error.

Total regret of an arm against the BEST AVAILABLE controller decomposes as

    [arm - exact_delta]      approximation regret : how badly the surrogate
                                                    estimates the delta it is asked for
  + [exact_delta - MV]       policy-design regret : how much is lost by asking for
                                                    delta-only at all, when a
                                                    variance-minimising combination
                                                    of the SAME two derivatives is
                                                    available

Both are read off one frozen zero-cost run, so the split is exact, not modelled.
The point: if policy-design regret dominates, the study's entire supervision axis is
optimising the smaller of two terms, and no amount of derivative accuracy can reach
the controller that a different POLICY reaches with the same information.
"""
from __future__ import annotations

import csv
import json
import os

SRC = "results/mv_delta_full/summary.json"
ARMS = ("rung3", "standard_pinn", "bs_gamma", "standard_pinn_smoothed")


def main() -> None:
    d = json.load(open(SRC))
    rows = []
    for cell in ("misspec", "in-model"):
        exact = d[f"{cell}|oracle"]
        mv = d[f"{cell}|mv_oracle"]
        policy = exact - mv
        for arm in ARMS:
            k = f"{cell}|{arm}"
            if k not in d:
                continue
            approx = d[k] - exact
            total = d[k] - mv
            rows.append({
                "cell": cell, "arm": arm,
                "cvar_arm": d[k], "cvar_exact_delta": exact, "cvar_mv": mv,
                "approximation_regret": approx,
                "policy_design_regret": policy,
                "total_regret_vs_mv": total,
                "policy_over_approx": (policy / approx) if approx > 0 else float("nan"),
                "approx_share_of_total": (approx / total) if total != 0 else float("nan"),
            })

    os.makedirs("results/reanalysis", exist_ok=True)
    with open("results/reanalysis/regret_decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{'cell':<10}{'arm':<24}{'approx':>10}{'policy':>10}{'total':>10}"
          f"{'policy/approx':>15}{'approx share':>14}")
    for r in rows:
        pa = r["policy_over_approx"]
        print(f"{r['cell']:<10}{r['arm']:<24}{r['approximation_regret']:>10.4f}"
              f"{r['policy_design_regret']:>10.4f}{r['total_regret_vs_mv']:>10.4f}"
              f"{pa:>15.1f}" if pa == pa else
              f"{r['cell']:<10}{r['arm']:<24}{r['approximation_regret']:>10.4f}"
              f"{r['policy_design_regret']:>10.4f}{r['total_regret_vs_mv']:>10.4f}"
              f"{'n/a (arm beats exact)':>15}", end="")
        print(f"{r['approx_share_of_total']:>14.3f}")
    print("\nwrote results/reanalysis/regret_decomposition.csv")


if __name__ == "__main__":
    main()
