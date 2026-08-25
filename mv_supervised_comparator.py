"""Can a SUPERVISED arm BUILD the minimum-variance hedge the oracle does not use?

THE QUESTION. `mv_delta_comparator.py` established that the frictionless Heston
delta is not the best hedge ratio on this metric: Delta_MV = dC/dS + (rho*xi/S)*dC/dv
beats the exact-delta oracle. That comparator built Delta_MV from the ANALYTIC
oracle, so it says nothing about what a trained network can do.

Delta_MV needs BOTH first derivatives. rung3 supervises dC/dS and dC/dv, so it can
assemble Delta_MV from its own outputs; the price-only and residual-only arms can
only assemble it from an UNSUPERVISED dC/dv (their vega RMSE is an order of
magnitude worse). If MV-from-rung3 lands near mv_oracle while MV-from-baseline does
not, the claim strengthens from "supervision reaches the oracle" to "supervision
builds a hedge the oracle itself does not use, and the unsupervised arms cannot".

LADDER ATTRIBUTION. rung2 supervises Delta and Gamma but NOT vega, so MV-from-rung2
isolates what the vega rung buys: the vega rung had no measurable payoff on the
registered delta-only metric, and this is the cell where it would show up.

SCOPE. Exploratory diagnostic on the SAME frozen checkpoints and the SAME CRN path
banks as every other cell. rho and xi in the correction are theta_train's (the same
sourcing mv_delta_comparator uses; no DGP leakage). Not a registered arm, not a
registered verdict, and it cannot move one. The MV / local-risk-minimisation
citation is still NOT in the verified record: describe by formula, cite nothing.
"""
# repo-local module; run from the repo root
import copy, json, os
import numpy as np

import Hedging_backtest as hb
import run_hedging as rh
from oracle import HestonParams
from providers import HestonCFProvider
from pinn_provider import build_providers


class MVDeltaProvider:
    """Wrap a GreekProvider and replace delta with the minimum-variance delta.

    Mirrors mv_delta_comparator.MVDeltaProvider (duplicated rather than imported
    because that module runs its study at import time). price and gamma pass
    through untouched, so the comparison isolates the hedge RATIO on fixed paths.
    """

    def __init__(self, base, rho: float, xi: float) -> None:
        self.base, self.rho, self.xi = base, float(rho), float(xi)

    def evaluate(self, S, v, tau, K) -> dict:
        out = dict(self.base.evaluate(S, v, tau, K))
        S_arr = np.broadcast_to(np.asarray(S, float), np.asarray(out["delta"]).shape)
        out["delta"] = out["delta"] + (self.rho * self.xi / S_arr) * out["vega"]
        return out


ARMS = ["standard_pinn", "feedforward", "sans_pde", "rung1", "rung2", "rung3"]

cfg = hb.resolve_config("heston_benchmark_v6.yaml", "hedging_config.yaml")
meta = cfg["benchmark"]["meta"]
# bounded by the scarcest arm in the comparison: feedforward / sans_pde are
# default-seed (5) arms, so the whole run uses seeds_min for a like-for-like table.
g, n_seeds = int(meta["global_seed"]), int(meta["seeds_min"])
r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
base_regime = cfg["benchmark"]["regimes"]["baseline"]
params = HestonParams(**{k: float(base_regime[k]) for k in ("kappa", "theta", "xi", "rho", "v0")})

prog = copy.deepcopy(cfg)
prog["derived"]["seeds"] = [g + i for i in range(n_seeds)]
prog["engine"]["risk"]["persist_pnl"] = True
mis = prog["benchmark"]["hedging_simulation"]["misspecification"]
mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
mis["cross_model"] = []
prog["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]
prog["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"] = [0.0, 0.01, 0.02]

run_root = "results/mv_supervised/confirmatory"
os.makedirs(run_root, exist_ok=True)
rows_all = []
for seed, tag, _k, _d in hb._iter_sim_cells(prog):
    prov = build_providers(cfg["benchmark"], "results/grid", ARMS, seed, r, q,
                           include_oracle=True)
    # analytic MV reference, then one MV variant per arm built from ITS OWN Greeks
    prov["mv_oracle"] = MVDeltaProvider(HestonCFProvider(params, r, q),
                                        rho=params.rho, xi=params.xi)
    for arm in ARMS:
        prov[f"{arm}_mv"] = MVDeltaProvider(prov[arm], rho=params.rho, xi=params.xi)
    cell = rh._cell_cfg(prog, tag, seed)
    rows = hb.run_headline(cell, prov, run_root)
    rows_all.extend(rows)
    print(f"  seed {seed} {tag}: {len(rows)} rows", flush=True)

by = {}
for r_ in rows_all:
    key = ("in-model" if float(r_.get("magnitude", 0)) == 0.0 else "misspec",
           float(r_["tc"]), r_["method"])
    by.setdefault(key, []).append(float(r_["cvar"]))
summary = {f"{c}|tc{tc}|{m}": float(np.mean(v)) for (c, tc, m), v in by.items()}
json.dump(summary, open("results/mv_supervised/summary.json", "w"), indent=2)

for cell in ("in-model", "misspec"):
    for tc in (0.0, 0.01, 0.02):
        sub = {m: float(np.mean(v)) for (c, t, m), v in by.items() if c == cell and t == tc}
        if not sub:
            continue
        print(f"\n=== {cell.upper()}  tc={tc}  mean CVaR95 over {n_seeds} seeds ===")
        ref_mv = sub.get("mv_oracle", float("nan"))
        ref_or = sub.get("oracle", float("nan"))
        print(f"  {'arm':16s} {'plain':>9s} {'MV-built':>9s}   MV vs oracle   MV vs mv_oracle")
        for arm in ARMS:
            p, m = sub.get(arm, float("nan")), sub.get(f"{arm}_mv", float("nan"))
            print(f"  {arm:16s} {p:9.4f} {m:9.4f}   {(ref_or-m)/ref_or*100:+9.2f}%   "
                  f"{(m-ref_mv)/ref_mv*100:+9.2f}%")
        print(f"  {'oracle':16s} {ref_or:9.4f}")
        print(f"  {'mv_oracle':16s} {'':9s} {ref_mv:9.4f}")
