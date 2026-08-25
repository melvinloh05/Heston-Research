"""R2 -- does region-restricted derivative RMSE predict hedging better than global?

The paper's central methodological claim is that evaluating derivative accuracy on a
parameter grid, rather than on the states the controller actually visits, mis-ranks
models. As a counterexample that rests on one reversal. As a TESTED PROPOSAL it needs
to be a measured property across every arm and seed available, with the departures
named rather than hidden.

Design. For each (seed, regime, derivative), rank the arms by
  (a) rmse on the full parameter grid   -- what the registered rule scores;
  (b) rmse restricted to the hedge box  -- the proposed replacement;
and rank the same arms by zero-cost misspecified CVaR95, the outcome. Report Spearman
rho of each metric against the outcome, plus pairwise concordance (of all arm pairs,
how many does the metric order the same way the outcome does).

Zero cost is the right tier: at 1% the outcome is contaminated by the turnover
subsidy, which is a property of the cost tier, not of derivative accuracy.

Reads only frozen artifacts. Writes results/reanalysis/metric_validation*.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
from collections import defaultdict

from scipy.stats import spearmanr

EVAL = "results/eval_greeks_hedgeslice/ood_param_greeks.csv"
HEDGE = "results/hedging_sweep/full_sweep/headline_delta_only_per_seed.csv"
OUT = "results/reanalysis"

# eval arm label -> hedging `method` label
ARM_MAP = {"standard_pinn": "standard_pinn", "feedforward": "feedforward",
           "sans_pde": "sans_pde", "rung1": "rung1", "rung2": "rung2",
           "rung3": "rung3", "bs_gamma": "bs_gamma"}
DERIVS = ("delta", "gamma", "vega", "vanna")


def load_eval() -> dict:
    """(regime, greek, seed, slice, arm) -> rmse"""
    out = {}
    for r in csv.DictReader(open(EVAL)):
        out[(r["regime"], r["greek"], int(r["seed"]), r["slice"], r["arm"])] = float(r["rmse"])
    return out


def load_outcome() -> dict:
    """(seed, method) -> zero-cost misspecified CVaR95."""
    out = {}
    for r in csv.DictReader(open(HEDGE)):
        if r["tc"] != "0.0" or r["in_model"] != "False":
            continue
        out[(int(r["seed"]), r["method"])] = float(r["cvar"])
    return out


def concordance(metric: dict, outcome: dict, arms: list) -> tuple[int, int, list]:
    ok = tot = 0
    bad = []
    for a, b in itertools.combinations(arms, 2):
        dm, do = metric[a] - metric[b], outcome[a] - outcome[b]
        if dm == 0 or do == 0:
            continue
        tot += 1
        if (dm > 0) == (do > 0):
            ok += 1
        else:
            bad.append((a, b))
    return ok, tot, bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="R2 metric validation")
    ap.add_argument("--exclude", default="", help="comma-separated arms to drop")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    a = ap.parse_args(argv)
    drop = {x for x in a.exclude.split(",") if x}
    tag = a.tag
    ev, oc = load_eval(), load_outcome()
    seeds = sorted({k[2] for k in ev})
    regimes = sorted({k[0] for k in ev})
    arms = [x for x in ARM_MAP
            if x not in drop and all((s, ARM_MAP[x]) in oc for s in seeds)]
    if drop:
        print(f"EXCLUDED: {sorted(drop)}")
    print(f"arms={len(arms)} {arms}\nseeds={seeds}  regimes={regimes}\n")

    rows, disagree = [], defaultdict(int)
    for regime in regimes:
        for greek in DERIVS:
            for seed in seeds:
                have = [a for a in arms
                        if (regime, greek, seed, "full", a) in ev
                        and (regime, greek, seed, "hedge", a) in ev]
                if len(have) < 4:
                    continue
                out_v = {a: oc[(seed, ARM_MAP[a])] for a in have}
                res = {"regime": regime, "greek": greek, "seed": seed, "n_arms": len(have)}
                for sl, mname in (("full", "global"), ("hedge", "box")):
                    m = {a: ev[(regime, greek, seed, sl, a)] for a in have}
                    xs = [m[a] for a in have]
                    ys = [out_v[a] for a in have]
                    rho = spearmanr(xs, ys).statistic
                    ok, tot, bad = concordance(m, out_v, have)
                    res[f"rho_{mname}"] = rho
                    res[f"conc_{mname}"] = ok
                    res[f"pairs_{mname}"] = tot
                    if sl == "hedge":
                        for p in bad:
                            disagree[p] += 1
                rows.append(res)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/metric_validation_per_seed{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Aggregate by (regime, greek)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["regime"], r["greek"])].append(r)
    print(f"{'regime':<18}{'greek':<8}{'n':>3}{'rho global':>12}{'rho box':>10}"
          f"{'conc global':>13}{'conc box':>11}")
    lines = []
    for (rg, gk), rs in sorted(agg.items()):
        n = len(rs)
        mg = sum(r["rho_global"] for r in rs) / n
        mb = sum(r["rho_box"] for r in rs) / n
        cg = sum(r["conc_global"] for r in rs), sum(r["pairs_global"] for r in rs)
        cb = sum(r["conc_box"] for r in rs), sum(r["pairs_box"] for r in rs)
        print(f"{rg:<18}{gk:<8}{n:>3}{mg:>12.3f}{mb:>10.3f}"
              f"{cg[0]}/{cg[1]:>10}{cb[0]}/{cb[1]:>10}")
        lines.append({"regime": rg, "greek": gk, "n_seeds": n,
                      "rho_global": mg, "rho_box": mb,
                      "conc_global": cg[0], "pairs_global": cg[1],
                      "conc_box": cb[0], "pairs_box": cb[1],
                      "conc_rate_global": cg[0] / cg[1] if cg[1] else float("nan"),
                      "conc_rate_box": cb[0] / cb[1] if cb[1] else float("nan")})
    with open(f"{OUT}/metric_validation_agg{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines[0])); w.writeheader(); w.writerows(lines)

    tg = (sum(l["conc_global"] for l in lines), sum(l["pairs_global"] for l in lines))
    tb = (sum(l["conc_box"] for l in lines), sum(l["pairs_box"] for l in lines))
    print(f"\nPOOLED concordance with zero-cost CVaR ranking:")
    print(f"  global-grid RMSE : {tg[0]}/{tg[1]} = {tg[0]/tg[1]:.3f}")
    print(f"  hedge-box  RMSE : {tb[0]}/{tb[1]} = {tb[0]/tb[1]:.3f}")
    print(f"\nMost frequent hedge-box DISAGREEMENTS (arm pairs the box metric mis-orders):")
    for pair, k in sorted(disagree.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {pair[0]:<16} vs {pair[1]:<16} {k:>3} of {len(rows)} cells")
    print(f"\nwrote {OUT}/metric_validation_{{per_seed,agg}}{tag}.csv")


if __name__ == "__main__":
    main()
