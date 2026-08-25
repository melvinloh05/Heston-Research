"""Unified metric comparison: global grid vs hedge box vs occupancy-weighted.

Merges R2 (global, box -- computed at the OOD parameter anchors) with R4
(occupancy-weighted -- computed at the confirmatory cell) into the single table the
methodological claim rests on: which way of measuring derivative accuracy actually
predicts realised hedging performance.

HONEST CONFOUND, stated because it changes what the table licenses. The three metrics
differ in TWO ways at once, not one:
  - WEIGHTING: uniform over grid nodes (global, box) vs the visitation measure mu^pi
    (occupancy);
  - PARAMETERS: near_feller / strong_neg_corr (global, box) vs the confirmatory cell's
    perturbed coefficients (occupancy).
So "occupancy beats box" is a claim about the whole estimand -- right states AND right
coefficients -- not about reweighting alone. The paper must say which it is claiming.
The comparison global-vs-box IS clean: same points, same parameters, different support.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

OUT = "results/reanalysis"


def pool_r2(path: str) -> dict:
    """Pool R2 per-seed rows across regimes and seeds, by greek."""
    acc = defaultdict(lambda: {"rho_global": [], "rho_box": [],
                               "cg": 0, "pg": 0, "cb": 0, "pb": 0})
    for r in csv.DictReader(open(path)):
        a = acc[r["greek"]]
        a["rho_global"].append(float(r["rho_global"]))
        a["rho_box"].append(float(r["rho_box"]))
        a["cg"] += int(r["conc_global"]); a["pg"] += int(r["pairs_global"])
        a["cb"] += int(r["conc_box"]);    a["pb"] += int(r["pairs_box"])
    return acc


def main() -> None:
    r2 = pool_r2(f"{OUT}/metric_validation_per_seed.csv")
    occ = {r["greek"]: r for r in csv.DictReader(open(f"{OUT}/occupancy_metric_agg.csv"))}

    rows = []
    for gk in ("delta", "gamma", "vega"):
        if gk not in r2:
            continue
        a = r2[gk]
        rec = {"greek": gk,
               "rho_global": float(np.mean(a["rho_global"])),
               "rho_box": float(np.mean(a["rho_box"])),
               "conc_global": a["cg"] / a["pg"],
               "conc_box": a["cb"] / a["pb"],
               "pairs_global": a["pg"], "pairs_box": a["pb"]}
        if gk in occ:
            rec["rho_occupancy"] = float(occ[gk]["mean_rho"])
            rec["conc_occupancy"] = float(occ[gk]["conc_rate"])
            rec["pairs_occupancy"] = int(occ[gk]["pairs"])
            rec["n_paths_occupancy"] = int(occ[gk]["n_paths"])
        rows.append(rec)

    with open(f"{OUT}/metric_comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("Spearman rho against realised zero-cost misspecified CVaR95 "
          "(7 arms, 5 seeds; higher = metric ranks arms more like the outcome)\n")
    print(f"{'derivative':<12}{'global grid':>14}{'hedge box':>12}{'occupancy':>12}")
    for r in rows:
        print(f"{r['greek']:<12}{r['rho_global']:>14.3f}{r['rho_box']:>12.3f}"
              f"{r.get('rho_occupancy', float('nan')):>12.3f}")
    print(f"\n{'derivative':<12}{'global grid':>14}{'hedge box':>12}{'occupancy':>12}"
          "     (pairwise concordance)")
    for r in rows:
        print(f"{r['greek']:<12}{r['conc_global']:>14.3f}{r['conc_box']:>12.3f}"
              f"{r.get('conc_occupancy', float('nan')):>12.3f}")
    print(f"\nwrote {OUT}/metric_comparison.csv")


if __name__ == "__main__":
    main()
