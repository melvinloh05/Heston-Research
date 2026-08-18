"""R03 — Hedging_backtest: rebalance grid vs the contract dt, and CRN checks.

(1) n_steps = round(T_prime * frequency_per_year) with T_prime=0.17 and 252/yr
    is NOT an integer number of daily steps; the realized dt differs from the
    contract's declared rebalancing.dt = 0.003968.
(2) CRN: every arm in a cell hedges the identical (times, S, v) arrays, and the
    per-arm RNG consumption is zero (paths are simulated once, outside the arm
    loop) -> verified by re-simulating and comparing bitwise.
(3) The QE branch mix does not desynchronise the stream across DGPs.
(4) tc-invariance of positions (the premise of the t_ex column) — checked by
    settling the same position array at every tier.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
import Hedging_backtest as hb  # noqa: E402

cfg = hb.resolve_config()
bm, eng = cfg["benchmark"], cfg["engine"]
hs = bm["hedging_simulation"]

print("=" * 74)
print("(1) rebalance grid vs contract dt")
print("=" * 74)
T_prime = float(eng["horizon"]["T_prime"])
freq = eng["rebalancing"]["frequency_per_year"]
tau0 = float(hs["instrument"]["tau0"])
n_steps = int(round(T_prime * freq))
dt_real = T_prime / n_steps
dt_contract = float(hs["rebalancing"]["dt"])
print(f"  contract hedging_simulation.rebalancing.dt = {dt_contract!r}")
print(f"  contract horizon.T_prime                   = {T_prime}")
print(f"  engine  rebalancing.frequency_per_year     = {freq}   (1/252 = {1/252:.9f})")
print(f"  T_prime * freq                             = {T_prime * freq}   (not an integer)")
print(f"  n_steps = round(...)                       = {n_steps}")
print(f"  realized dt = T_prime / n_steps            = {dt_real:.9f}")
print(f"  contract dt                                = {dt_contract:.9f}")
print(f"  relative difference                        = {abs(dt_real - dt_contract)/dt_contract:.4%}")
print(f"  n_steps implied by contract dt             = {T_prime/dt_contract:.4f}")
print(f"  => realized rebalances per year            = {1.0/dt_real:.4f}")

print()
print("=" * 74)
print("(2)+(3) CRN / stream alignment")
print("=" * 74)
base = hb.SimParams.from_regime(bm["regimes"][hs["misspecification"]["train_params"]],
                                bm["grid"]["r"], bm["grid"]["q"])
psi_c = eng["simulation"]["psi_c"]
S0 = float(hs["instrument"]["S0"])
t1, S1, v1 = hb.simulate_heston_qe(base, S0, T_prime, n_steps, 500, 42, psi_c)
t2, S2, v2 = hb.simulate_heston_qe(base, S0, T_prime, n_steps, 500, 42, psi_c)
print(f"  same seed re-simulated bit-identical : S {np.array_equal(S1, S2)}  v {np.array_equal(v1, v2)}")
bat = hb.dataclasses.replace(base, lambda_j=0.0, mu_j=-0.10, sigma_j=0.10)
_, S3, v3 = hb.simulate_heston_qe(bat, S0, T_prime, n_steps, 500, 42, psi_c)
print(f"  Bates lambda_j=0 recovers Heston     : S {np.array_equal(S1, S3)}  v {np.array_equal(v1, v3)}")
pert = hb.perturb_params(base, "combined", 1.0, eng["misspecification"]["directions"])
print(f"  combined m=1 -> xi={pert.xi}, rho={pert.rho} (contract targets 0.45 / -0.80)")
_, S4, v4 = hb.simulate_heston_qe(pert, S0, T_prime, n_steps, 500, 42, psi_c)
print(f"  perturbed cell differs from base     : {not np.array_equal(S1, S4)}")
n_zero = int((v1 == 0.0).sum())
print(f"  QE exponential-branch atom: exact v==0 states in base cell = {n_zero}"
      f" / {v1.size}")

print()
print("=" * 74)
print("(4) positions are tc-invariant (premise of the t_ex column)")
print("=" * 74)


class _Lin:
    """Toy provider: price/delta depend only on state, never on tc."""

    def evaluate(self, S, v, tau, K):
        S = np.asarray(S, float)
        d = np.clip((S - K) / 20.0 + 0.5, 0.0, 1.0)
        return {"price": np.maximum(S - K, 0.0) + 1.0, "delta": d,
                "gamma": np.full_like(S, 0.05)}


pos, prem = hb.delta_positions(S1, v1, t1, _Lin(), 100.0, tau0)
liab = np.maximum(S1[:, -1] - 100.0, 0.0)
res = {tc: hb.settle_delta(S1, t1, pos, 0.02, 0.0, tc, prem, liab, True)
       for tc in (0.0, 0.01, 0.02)}
print(f"  positions identical across tiers : "
      f"{all(np.array_equal(res[0.0].positions, res[t].positions) for t in (0.01, 0.02))}")
print(f"  t_ex source _total_traded equal   : "
      f"{np.array_equal(hb._total_traded(res[0.0].positions), hb._total_traded(res[0.02].positions))}")
for tc, r_ in res.items():
    print(f"    tc={tc:<5} mean pnl={r_.pnl_total.mean():+.6f}  "
          f"cvar95={hb.cvar(r_.pnl_total, 0.95):+.6f}  "
          f"decomposition ok={np.allclose(r_.pnl_total, r_.pnl_directional - r_.tc_paid_fv)}")
