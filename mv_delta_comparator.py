"""MV-delta comparator — the diagnostic for the bs_gamma anomaly.

THE ANOMALY. In-model at tc=0, bs_gamma (a WRONG-model Black-Scholes gamma label)
hedges BETTER than the exact-Greek oracle: rel +8.94%, CI [-0.164, -0.147], excluding
zero. If that is real it says exact-delta tracking is measurably suboptimal for
discrete CVaR95 hedging under Heston, which is a claim about the metric, not about
the network.

THE CANDIDATE EXPLANATION. The oracle hedges the FRICTIONLESS Heston delta
dC/dS. Under stochastic volatility the variance-minimising hedge ratio is not that
delta: spot and variance are correlated (rho < 0), so a spot move carries
information about v, and the minimum-variance delta picks up a second term,

    Delta_MV = dC/dS + (rho * xi / S) * dC/dv .

With rho < 0 this SHIFTS the hedge ratio down. A BS-gamma-biased fit is damped in a
similar direction, so it may be landing nearer the discrete-optimal hedge by
accident. If MV-oracle ~ bs_gamma, the anomaly is explained and stops being a
mystery; if MV-oracle does NOT reproduce it, the explanation is wrong and bs_gamma
stays "observed, unexplained".

SCOPE. Exploratory diagnostic on the registered in-model cell, run on the SAME CRN
paths as everything else. Not a registered arm, not a claim. The citation for the
MV-delta / local-risk-minimisation family (Foellmer-Schweizer and successors) is NOT
in the verified record; nothing here may be cited until that is obtained and read.
"""
# repo-local module; run from the repo root
import copy, json
import numpy as np
import yaml

import Hedging_backtest as hb
import run_hedging as rh
from oracle import HestonParams
from providers import HestonCFProvider
from pinn_provider import build_providers


class MVDeltaProvider:
    """Wraps a GreekProvider and replaces delta with the minimum-variance delta.

    price and gamma pass through untouched; only the hedge ratio changes, which is
    the whole point -- the comparison isolates the hedge RATIO, holding the priced
    instrument and every path fixed.
    """

    def __init__(self, base, rho: float, xi: float) -> None:
        self.base, self.rho, self.xi = base, float(rho), float(xi)

    def evaluate(self, S, v, tau, K) -> dict:
        out = dict(self.base.evaluate(S, v, tau, K))
        S_arr = np.broadcast_to(np.asarray(S, float), np.asarray(out["delta"]).shape)
        # dC/dv is the provider's variance vega (v0 is the variance input)
        out["delta"] = out["delta"] + (self.rho * self.xi / S_arr) * out["vega"]
        return out


cfg = hb.resolve_config("heston_benchmark_v6.yaml", "hedging_config.yaml")
meta = cfg["benchmark"]["meta"]
# bs_gamma is a DEFAULT-seed arm (5), not a confirmatory-seed arm (10). Looping the
# confirmatory seeds asks for checkpoints that were never trained; the comparator is
# bounded by the scarcest arm in it.
g, n_seeds = int(meta["global_seed"]), int(meta["seeds_min"])
r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
base = cfg["benchmark"]["regimes"]["baseline"]
params = HestonParams(**{k: float(base[k]) for k in ("kappa", "theta", "xi", "rho", "v0")})

prog = copy.deepcopy(cfg)
prog["derived"]["seeds"] = [g + i for i in range(n_seeds)]
prog["engine"]["risk"]["persist_pnl"] = True
mis = prog["benchmark"]["hedging_simulation"]["misspecification"]
mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
mis["cross_model"] = []
prog["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]
prog["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"] = [0.0]

run_root = "results/mv_delta/confirmatory"
import os; os.makedirs(run_root, exist_ok=True)
rows_all = []
for seed, tag, _k, _d in hb._iter_sim_cells(prog):
    prov = build_providers(cfg["benchmark"], "results/grid", ["rung3", "bs_gamma", "standard_pinn"],
                           seed, r, q, include_oracle=True)
    prov["mv_oracle"] = MVDeltaProvider(HestonCFProvider(params, r, q),
                                        rho=params.rho, xi=params.xi)
    cell = rh._cell_cfg(prog, tag, seed)
    rows = hb.run_headline(cell, prov, run_root)
    rows_all.extend(rows)
    print(f"  seed {seed}: {len(rows)} rows", flush=True)

by = {}
for r_ in rows_all:
    key = ("in-model" if float(r_.get("magnitude", 0)) == 0.0 else "misspec", r_["method"])
    by.setdefault(key, []).append(float(r_["cvar"]))
for cell in ("in-model", "misspec"):
    sub = {m: v for (c, m), v in by.items() if c == cell}
    if not sub: continue
    print(f"\n=== {cell.upper()}, tc=0, mean CVaR95 over seeds ===")
    ref = float(np.mean(sub["oracle"]))
    base = float(np.mean(sub["standard_pinn"])) if "standard_pinn" in sub else float("nan")
    for m in sorted(sub):
        mu = float(np.mean(sub[m]))
        gc = (base - mu) / (base - ref) * 100 if np.isfinite(base) and abs(base - ref) > 1e-9 else float("nan")
        print(f"  {m:16s} {mu:8.4f}   vs oracle {(ref-mu)/ref*100:+7.2f}%   gap_closed(vs oracle) {gc:7.1f}%")
json.dump({f"{c}|{m}": float(np.mean(v)) for (c, m), v in by.items()},
          open("results/mv_delta/summary.json", "w"), indent=2)
