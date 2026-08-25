"""R4 -- occupancy-weighted derivative error: the estimand the hedge box approximates.

The paper's proposed metric restricts derivative RMSE to a BOX -- a hand-drawn
(S/K, tau) rectangle around the hedge. The objection writes itself: a hand-selected
subset can be drawn to favour whatever you like. The principled estimand is

    E_{mu^pi} [ (Dhat C - D C)^2 ]

the derivative error integrated against the OCCUPANCY MEASURE mu^pi -- the
distribution of states the controller actually visits under its own policy. The box
is that estimand's first-order proxy: right support, uniform weights.

This computes the estimand itself. Paths are regenerated from the frozen seed-keyed
streams (default_rng([seed, _STREAM_DIFFUSION]), arm-independent, so CRN holds), the
visited states are exactly those the engine forms positions at, and the reference is
the same theta_train CF oracle the engine hedges against. Two mismatches the box
metric cannot avoid are removed at once: the box is scored at OOD parameter anchors
while hedging runs at the confirmatory cell, and the box weights every grid node
equally while the controller spends its time near the money.

Then the R2 test is rerun on this metric: does it rank the arms against realised
zero-cost CVaR better than either the global grid or the box?
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np
from scipy.stats import spearmanr

from Hedging_backtest import (SimParams, perturb_params, resolve_config,
                              simulate_heston_qe)
from pinn_provider import build_providers

ARMS = ["standard_pinn", "rung1", "rung2", "rung3", "feedforward", "sans_pde", "bs_gamma"]
HEDGE = "results/hedging_sweep/full_sweep/headline_delta_only_per_seed.csv"
OUT = "results/reanalysis"


def visited_states(cfg: dict, seed: int, n_paths: int):
    """Exactly the (S, v, tau) the engine forms a position at, for the
    confirmatory cell (combined perturbation at full magnitude)."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    r, q = float(bm["grid"]["r"]), float(bm["grid"]["q"])
    inst, hor = hs["instrument"], hs["horizon"]
    S0, K, tau0 = float(inst["S0"]), float(inst["K"]), float(inst["tau0"])
    tp = float(hor["T_prime"])
    n_steps = int(hs["rebalancing"]["n_steps"])
    base = SimParams.from_regime(bm["regimes"][hs["misspecification"]["train_params"]], r, q)
    hp = perturb_params(base, "combined", 1.0, eng["misspecification"]["directions"])
    times, S, v = simulate_heston_qe(hp, S0, tp, n_steps, n_paths, seed,
                                     float(eng["simulation"]["psi_c"]))
    # Positions are formed at i = 0..n_steps-1 (the liquidation at n_steps expresses
    # no view). One (S_t, v_t) slice per step at a SCALAR tau, exactly how the engine
    # calls the provider -- so the states scored here are the states hedged on.
    steps = [(S[:, i], v[:, i], float(tau0 - times[i])) for i in range(n_steps)]
    return steps, K, r, q


def load_outcome() -> dict:
    out = {}
    for r in csv.DictReader(open(HEDGE)):
        if r["tc"] == "0.0" and r["in_model"] == "False":
            out[(int(r["seed"]), r["method"])] = float(r["cvar"])
    return out


def concordance(metric: dict, outcome: dict, arms: list):
    ok = tot = 0
    for a, b in itertools.combinations(arms, 2):
        dm, do = metric[a] - metric[b], outcome[a] - outcome[b]
        if dm == 0 or do == 0:
            continue
        tot += 1
        ok += (dm > 0) == (do > 0)
    return ok, tot


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="R4 occupancy-weighted derivative error")
    ap.add_argument("--ckpt-root", default="results/grid")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--n-paths", type=int, default=1000,
                    help="subsample of the 10k production paths; mu^pi is preserved")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg = resolve_config()
    oc = load_outcome()

    rows = []
    for seed in seeds:
        steps, K, r, q = visited_states(cfg, seed, a.n_paths)
        provs = build_providers(cfg["benchmark"], a.ckpt_root, ARMS, seed, r, q,
                                include_oracle=True)
        GK = ("delta", "gamma", "vega")
        sq = {arm: {gk: [0.0, 0] for gk in GK} for arm in ARMS if arm in provs}
        for Ss, vs, tau in steps:
            ref = provs["oracle"].evaluate(Ss, vs, tau, K)
            for arm in sq:
                g = provs[arm].evaluate(Ss, vs, tau, K)
                for gk in GK:
                    if gk not in g or gk not in ref:
                        continue
                    e = np.asarray(g[gk], float) - np.asarray(ref[gk], float)
                    ok = np.isfinite(e)
                    sq[arm][gk][0] += float(np.sum(e[ok] ** 2))
                    sq[arm][gk][1] += int(ok.sum())
        for arm, d in sq.items():
            rec = {"seed": seed, "arm": arm,
                   "n_states": d["delta"][1]}
            for gk in GK:
                tot, n = d[gk]
                rec[f"occ_rmse_{gk}"] = float(np.sqrt(tot / n)) if n else float("nan")
            rows.append(rec)
            print(f"  seed {seed} {arm:<16} occ delta RMSE = {rec['occ_rmse_delta']:.6f}"
                  f"   ({rec['n_states']} states)")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/occupancy_metric_per_seed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Rank test against realised zero-cost CVaR, same protocol as R2.
    print(f"\n{'greek':<8}{'mean rho':>10}{'concordance':>14}")
    agg = []
    for gk in ("delta", "gamma", "vega"):
        rhos, ok_t, tot_t = [], 0, 0
        for seed in seeds:
            m = {r_["arm"]: r_[f"occ_rmse_{gk}"] for r_ in rows
                 if r_["seed"] == seed and f"occ_rmse_{gk}" in r_}
            have = [x for x in m if (seed, x) in oc]
            if len(have) < 4:
                continue
            o = {x: oc[(seed, x)] for x in have}
            rhos.append(spearmanr([m[x] for x in have], [o[x] for x in have]).statistic)
            k, t = concordance(m, o, have)
            ok_t += k; tot_t += t
        mr = float(np.mean(rhos)) if rhos else float("nan")
        print(f"{gk:<8}{mr:>10.3f}{ok_t}/{tot_t:>10}")
        agg.append({"greek": gk, "mean_rho": mr, "concordant": ok_t, "pairs": tot_t,
                    "conc_rate": ok_t / tot_t if tot_t else float("nan"),
                    "n_seeds": len(rhos), "n_paths": a.n_paths})
    with open(f"{OUT}/occupancy_metric_agg.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    print(f"\nwrote {OUT}/occupancy_metric_{{per_seed,agg}}.csv")


if __name__ == "__main__":
    main()
