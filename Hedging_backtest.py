"""Hedging-backtest engine: DELTA-ONLY headline, delta-gamma SECONDARY contrast.

Consumes any (price, Greeks) provider through the `GreekProvider` protocol —
oracle, PINN and baselines all plug in through the same interface, and no
provider internals are imported. The only oracle import is the LAZY one inside
the terminal-liability mark helpers: the mark is benchmark truth (the TRUE-DGP
price at the horizon), not a competing provider. Everything resolves from the
frozen benchmark contract (heston_benchmark_v6.yaml) plus the engine
supplement (hedging_config.yaml); the resolved config is written alongside
every run.

Contract PnL convention (hedging_simulation.pnl_convention):
- initial premium = theta_train ORACLE price for ALL arms (a provider that
  overprices by eps must NOT gain a uniform eps*e^{rT'} PnL shift); when the
  oracle provider is absent the engine falls back to per-provider premiums,
  warns loudly, and stamps premium_convention_ok=False on every row.
- horizon: hedge to T' = horizon.T_prime < tau0 and liquidate there; the
  terminal liability is marked at the TRUE-DGP price of the remaining
  (K, tau0 - T') call — never intrinsic-at-expiry unless T' == tau0.

Additive PnL decomposition (asserted at every settlement): positions never
depend on transaction costs, hence per path, exactly,
    pnl_total = pnl_directional - tc_paid_fv
where `pnl_directional` is the frictionless PnL of the SAME position path
(directional-accuracy component) and `tc_paid_fv` is the sum of transaction
costs future-valued to the settlement horizon at r (turnover/cost component).

No look-ahead: the position held over [t_i, t_{i+1}) is a function of
(S_i, v_i, tau_i) only; paths are generated strictly forward.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

_log = logging.getLogger(__name__)

_STREAM_DIFFUSION, _STREAM_JUMP, _STREAM_BOOT, _STREAM_PAIRED = 0, 1, 2, 3
_QE_GAMMA1 = _QE_GAMMA2 = 0.5      # Andersen central discretisation weights
_DECOMP_ATOL = 1e-9                # decomposition balance tolerance per unit S0

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_config(contract_path: str = "heston_benchmark_v6.yaml",
                   engine_path: str = "hedging_config.yaml") -> dict:
    """Frozen contract + engine supplement -> {'benchmark','engine','derived'};
    seeds derived as global_seed + 0..seeds_min-1 (never hard-coded). Asserts
    the engine misspecification shifts land exactly on the contract's
    perturbation targets (see _assert_contract_targets)."""
    bm, eng = _load_yaml(contract_path), _load_yaml(engine_path)
    _assert_contract_targets(bm, eng)
    g, n = int(bm["meta"]["global_seed"]), int(bm["meta"]["seeds_min"])
    return {"benchmark": bm, "engine": eng,
            "derived": {"seeds": [g + i for i in range(n)],
                        "contract_path": str(contract_path),
                        "engine_path": str(engine_path)}}


def log_resolved_config(cfg: dict, out_dir: str) -> str:
    """Write the fully resolved config next to the run outputs."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "resolved_config.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path

# ---------------------------------------------------------------------------
# (price, Greeks) interface — the only door into the engine
# ---------------------------------------------------------------------------

@runtime_checkable
class GreekProvider(Protocol):
    """Interface every method (oracle, PINN, baseline) plugs in through.

    evaluate(S, v, tau, K) -> {'price','delta','gamma'} broadcast to S.shape,
    with Delta = dV/dS and Gamma = d2V/dS2 at state (S, v), time-to-maturity
    tau > 0, strike K. r/q handling is internal to the provider. Delta-only
    hedging uses 'price' and 'delta'; delta-gamma additionally uses 'gamma'.
    Outputs must be finite for all v >= 0 (QE paths can hit v = 0 exactly);
    any variance flooring is the provider's responsibility.
    """

    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float,
                 K: float) -> dict: ...

# ---------------------------------------------------------------------------
# hedge-side dynamics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimParams:
    """Heston diffusion + optional lognormal jumps (lambda_j > 0 => Bates)."""
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    r: float
    q: float
    lambda_j: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0

    @classmethod
    def from_regime(cls, regime: dict, r: float, q: float, **jumps) -> "SimParams":
        return cls(kappa=regime["kappa"], theta=regime["theta"], xi=regime["xi"],
                   rho=regime["rho"], v0=regime["v0"], r=r, q=q, **jumps)


def perturb_params(base: SimParams, direction: str, magnitude: float,
                   directions_cfg: dict) -> SimParams:
    """hedge-params = train-params + magnitude * shift_at_m1 along each leg of
    `direction` (composite directions list their legs)."""
    spec = directions_cfg[direction]
    legs = spec["legs"] if "legs" in spec else [direction]
    kw = {}
    for leg in legs:
        ls = directions_cfg[leg]
        kw[ls["param"]] = getattr(base, ls["param"]) + magnitude * ls["shift_at_m1"]
    return dataclasses.replace(base, **kw)


def _assert_contract_targets(bm: dict, eng: dict) -> None:
    """Contract-target lock: perturbing the train-params regime by magnitude 1.0
    along every engine direction must land EXACTLY (atol 1e-12) on the
    contract's hedging_simulation.misspecification.perturbations values
    (xi_up -> xi 0.45, rho_down -> rho -0.80), so the contract numbers can
    never silently drift from the engine shift_at_m1 choices. Composite
    directions are asserted leg by leg against the composite target."""
    hs = bm["hedging_simulation"]["misspecification"]
    dirs = eng["misspecification"]["directions"]
    base = SimParams.from_regime(bm["regimes"][hs["train_params"]],
                                 bm["grid"]["r"], bm["grid"]["q"])
    for name, target in hs["perturbations"].items():
        assert name in dirs, (
            f"contract perturbation {name!r} has no engine direction "
            "(hedging_config.yaml misspecification.directions)")
        legs = dirs[name].get("legs", [name])
        for leg in legs:
            param = dirs[leg]["param"]
            got = getattr(perturb_params(base, leg, 1.0, dirs), param)
            want = float(target[param])
            assert abs(got - want) < 1e-12, (
                f"engine direction {leg!r} at magnitude 1.0 gives {param}="
                f"{got!r} but the contract target ({name!r}) is {want!r} — "
                "shift_at_m1 and the contract have drifted apart")


def simulate_heston_qe(p: SimParams, S0: float, T: float, n_steps: int,
                       n_paths: int, seed: int, psi_c: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Andersen (2008) QE scheme under Q; returns (times, S, v) with S, v of
    shape [n_paths, n_steps+1], generated strictly forward (no look-ahead).
    Jumps draw from an independent RNG stream and the block is skipped entirely
    at lambda_j=0, so lambda_j=0 recovers pure-Heston paths bit-for-bit.
    Plain QE (no martingale correction): the O(dt) drift bias is common to all
    hedgers, leaving method comparisons unaffected."""
    rng = np.random.default_rng([seed, _STREAM_DIFFUSION])
    rng_j = np.random.default_rng([seed, _STREAM_JUMP])
    times = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps
    emk = np.exp(-p.kappa * dt)
    c1 = p.xi ** 2 * emk * (1.0 - emk) / p.kappa
    c2 = p.theta * p.xi ** 2 * (1.0 - emk) ** 2 / (2.0 * p.kappa)
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    k1 = _QE_GAMMA1 * dt * (p.kappa * p.rho / p.xi - 0.5) - p.rho / p.xi
    k2 = _QE_GAMMA2 * dt * (p.kappa * p.rho / p.xi - 0.5) + p.rho / p.xi
    k3 = _QE_GAMMA1 * dt * (1.0 - p.rho ** 2)
    k4 = _QE_GAMMA2 * dt * (1.0 - p.rho ** 2)
    drift = (p.r - p.q) * dt
    comp = p.lambda_j * (np.exp(p.mu_j + 0.5 * p.sigma_j ** 2) - 1.0) * dt
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0], v[:, 0] = S0, p.v0
    ln_s = np.full(n_paths, np.log(S0))
    vc = np.full(n_paths, p.v0)
    for i in range(1, n_steps + 1):
        m = p.theta + (vc - p.theta) * emk
        s2 = vc * c1 + c2
        psi = s2 / m ** 2
        z_v = rng.standard_normal(n_paths)     # always drawn: stream alignment
        u = rng.random(n_paths)                # independent of branch mix
        v_next = np.empty(n_paths)
        quad = psi <= psi_c
        iq = 2.0 / psi[quad]
        b2 = iq - 1.0 + np.sqrt(iq) * np.sqrt(iq - 1.0)
        v_next[quad] = m[quad] / (1.0 + b2) * (np.sqrt(b2) + z_v[quad]) ** 2
        pe = psi[~quad]
        pp = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - pp) / m[~quad]
        ue = u[~quad]
        v_next[~quad] = np.where(ue <= pp, 0.0,
                                 np.log((1.0 - pp) / (1.0 - ue)) / beta)
        z_s = rng.standard_normal(n_paths)
        ln_s = ln_s + drift + k0 + k1 * vc + k2 * v_next \
            + np.sqrt(k3 * vc + k4 * v_next) * z_s
        if p.lambda_j > 0.0:
            n_j = rng_j.poisson(p.lambda_j * dt, n_paths)
            ln_s = ln_s + p.mu_j * n_j \
                + p.sigma_j * np.sqrt(n_j) * rng_j.standard_normal(n_paths) - comp
        vc = v_next
        S[:, i] = np.exp(ln_s)
        v[:, i] = vc
    return times, S, v


def simulate_merton(mp: dict, S0: float, T: float, n_steps: int, n_paths: int,
                    seed: int, r: float, q: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GBM + lognormal jumps (Merton) under Q. Returns the constant variance
    path v = sigma^2 so providers keep the uniform (S, v, tau) state interface."""
    sigma, lam = mp["sigma"], mp["lambda_j"]
    mu_j, sigma_j = mp["mu_j"], mp["sigma_j"]
    rng = np.random.default_rng([seed, _STREAM_DIFFUSION])
    rng_j = np.random.default_rng([seed, _STREAM_JUMP])
    times = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps
    comp = lam * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0) * dt
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    ln_s = np.full(n_paths, np.log(S0))
    for i in range(1, n_steps + 1):
        ln_s = ln_s + (r - q - 0.5 * sigma ** 2) * dt \
            + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        if lam > 0.0:
            n_j = rng_j.poisson(lam * dt, n_paths)
            ln_s = ln_s + mu_j * n_j \
                + sigma_j * np.sqrt(n_j) * rng_j.standard_normal(n_paths) - comp
        S[:, i] = np.exp(ln_s)
    v = np.full((n_paths, n_steps + 1), sigma ** 2)
    return times, S, v

# ---------------------------------------------------------------------------
# positions (cost-independent by construction) and smoothing control
# ---------------------------------------------------------------------------

def delta_positions(S: np.ndarray, v: np.ndarray, times: np.ndarray,
                    provider: GreekProvider, K: float, T_opt: float,
                    premium_override: np.ndarray | float | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Raw delta path pos[:, i] = provider delta at (S_i, v_i, tau_i); premium
    = provider price at inception, unless `premium_override` (contract pnl
    convention: theta_train ORACLE price for ALL arms) replaces it — positions
    are unaffected by the override. Depends only on time-i state (no
    look-ahead) and never on transaction costs."""
    n_paths, n_steps = S.shape[0], S.shape[1] - 1
    pos = np.empty((n_paths, n_steps))
    premium = np.empty(n_paths)
    for i in range(n_steps):
        out = provider.evaluate(S[:, i], v[:, i], float(T_opt - times[i]), K)
        pos[:, i] = np.asarray(out["delta"], float)
        if i == 0:
            premium[:] = np.asarray(out["price"], float)
    if premium_override is not None:
        premium = np.array(np.broadcast_to(
            np.asarray(premium_override, float), (n_paths,)))
    return pos, premium


def dg_positions(S: np.ndarray, v: np.ndarray, times: np.ndarray,
                 provider: GreekProvider, K: float, T_opt: float,
                 K_h: float, T_h: float, min_tau_h: float, gamma_floor: float,
                 premium_override: np.ndarray | float | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SECONDARY delta-gamma legs: option leg n = Gamma_tgt / Gamma_hedge
    (gamma-neutral), underlying leg u = Delta_tgt - n * Delta_hedge. The hedge
    option (strike K_h, maturity T_h; T_h < T_opt supported) is marked with the
    same provider. Guards (the hedge option is NOT used when unusable, rather
    than divided by ~0): no new option position when it is within min_tau_h of
    its own expiry or would expire before the next rebalance, and a per-path
    delta-only fallback (n = 0, u = Delta_tgt) where its Gamma <= gamma_floor
    (also excludes non-positive Gamma). The provider is never called with
    tau <= 0; once the hedge option has expired its price column is intrinsic,
    reached only with zero holdings. `premium_override` replaces the
    per-provider inception premium (contract pnl convention: theta_train
    ORACLE price for ALL arms) without touching positions. Returns
    (u, n_opt, ph, premium) with ph the hedge-option price path
    [n_paths, n_steps+1]."""
    n_paths, n_steps = S.shape[0], S.shape[1] - 1
    u = np.empty((n_paths, n_steps))
    n_opt = np.empty((n_paths, n_steps))
    ph = np.empty((n_paths, n_steps + 1))
    premium = np.empty(n_paths)
    for i in range(n_steps):
        tgt = provider.evaluate(S[:, i], v[:, i], float(T_opt - times[i]), K)
        if i == 0:
            premium[:] = np.asarray(tgt["price"], float)
        d_t = np.asarray(tgt["delta"], float)
        tau_h = float(T_h - times[i])
        if tau_h <= 0.0:
            ph[:, i] = np.maximum(S[:, i] - K_h, 0.0)
            n_opt[:, i] = 0.0
            u[:, i] = d_t
            continue
        hdg = provider.evaluate(S[:, i], v[:, i], tau_h, K_h)
        ph[:, i] = np.asarray(hdg["price"], float)
        if tau_h > min_tau_h and float(T_h - times[i + 1]) > 0.0:
            g_h = np.asarray(hdg["gamma"], float)
            ok = g_h > gamma_floor
            n_opt[:, i] = np.where(
                ok, np.asarray(tgt["gamma"], float) / np.where(ok, g_h, 1.0), 0.0)
            u[:, i] = d_t - n_opt[:, i] * np.asarray(hdg["delta"], float)
        else:
            n_opt[:, i] = 0.0
            u[:, i] = d_t
    tau_h = float(T_h - times[-1])          # grid may end at T' < T_opt
    if tau_h > 0.0:
        out = provider.evaluate(S[:, -1], v[:, -1], tau_h, K_h)
        ph[:, -1] = np.asarray(out["price"], float)
    else:
        ph[:, -1] = np.maximum(S[:, -1] - K_h, 0.0)
    if premium_override is not None:
        premium = np.array(np.broadcast_to(
            np.asarray(premium_override, float), (n_paths,)))
    return u, n_opt, ph, premium


def smooth_positions(pos: np.ndarray, ema_alpha: float | None = None,
                     no_trade_band: float = 0.0) -> np.ndarray:
    """PURE no-trade band on a raw position path — the HEDGE-TIME 'is it just
    smoothing?' control (the decisive TRAIN-TIME control is Task 3), kept in
    the citable Whalley-Wilmott-style band policy class the contract names:
    hold the current position until |raw_delta_t - held| > no_trade_band,
    then jump TO raw_delta_t. `ema_alpha` is accepted only for signature
    compatibility and IGNORED (the former EMA pre-filter was a different,
    uncitable policy). Output at t uses raw positions up to t only (no
    look-ahead)."""
    del ema_alpha
    out = np.empty_like(pos)
    held = pos[:, 0].copy()
    out[:, 0] = held
    for i in range(1, pos.shape[1]):
        held = np.where(np.abs(pos[:, i] - held) > no_trade_band,
                        pos[:, i], held)
        out[:, i] = held
    return out

# ---------------------------------------------------------------------------
# settlement with exact additive decomposition
# ---------------------------------------------------------------------------

@dataclass
class HedgeResult:
    """Per-path results; pnl_total = pnl_directional - tc_paid_fv (asserted)."""
    pnl_total: np.ndarray
    pnl_directional: np.ndarray
    tc_paid_fv: np.ndarray
    turnover: np.ndarray
    positions: np.ndarray


def _settle_core(times: np.ndarray, r: float, tc: float, premium: np.ndarray,
                 liability: np.ndarray, legs: list, charge_final_unwind: bool
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Self-financing settlement of a short option hedged with `legs`, each
    (positions [n_paths, n_steps], prices [n_paths, n_steps+1], dividend_yield).
    `liability` is the per-path mark of the short option at the LAST grid time
    (true-DGP price at T' under the contract convention; intrinsic only when
    the grid runs to expiry), computed by the caller. Dividend income accrues
    as pos * dy * price * dt (order-dt accurate). Returns (pnl_total,
    transaction costs future-valued to the settlement horizon at r)."""
    n_steps = legs[0][0].shape[1]
    dt = np.diff(times)
    cash = premium.astype(float).copy()
    tcf = np.zeros_like(cash)
    for pos, px, _dy in legs:
        cost = tc * px[:, 0] * np.abs(pos[:, 0])
        cash -= pos[:, 0] * px[:, 0] + cost
        tcf += cost
    for i in range(1, n_steps):
        g = np.exp(r * dt[i - 1])
        income = sum(dy * pos[:, i - 1] * px[:, i - 1] * dt[i - 1]
                     for pos, px, dy in legs)
        cash = cash * g + income
        tcf = tcf * g
        for pos, px, _dy in legs:
            d = pos[:, i] - pos[:, i - 1]
            cost = tc * px[:, i] * np.abs(d)
            cash -= d * px[:, i] + cost
            tcf += cost
    g = np.exp(r * dt[-1])
    income = sum(dy * pos[:, -1] * px[:, n_steps - 1] * dt[-1]
                 for pos, px, dy in legs)
    cash = cash * g + income
    tcf = tcf * g
    for pos, px, _dy in legs:
        cost = tc * px[:, n_steps] * np.abs(pos[:, -1]) if charge_final_unwind else 0.0
        cash += pos[:, -1] * px[:, n_steps] - cost
        tcf += cost
    return cash - liability, tcf


def _turnover(pos: np.ndarray) -> np.ndarray:
    """Per-step MEAN |position change| per path, EXCLUDING the inception trade
    and final unwind — kept as the `turnover` column for continuity with
    earlier runs; the contract T_ex statistic uses _total_traded instead."""
    if pos.shape[1] < 2:
        return np.zeros(pos.shape[0])
    return np.abs(np.diff(pos, axis=1)).mean(axis=1)


def _total_traded(pos: np.ndarray) -> np.ndarray:
    """Per-path TOTAL traded quantity INCLUDING endpoints:
    |pos_0| + sum|diff(pos)| + |pos_last| — exactly the quantity tc is charged
    on, at UNIT price weighting (the tc charge weights each trade by the
    concurrent price; this statistic deliberately does not). Contract
    falsifier: T_ex = total_traded(method) - total_traded(oracle) per path."""
    return (np.abs(pos[:, 0]) + np.abs(np.diff(pos, axis=1)).sum(axis=1)
            + np.abs(pos[:, -1]))


def _assert_decomposition(pnl: np.ndarray, pnl0: np.ndarray, tcf: np.ndarray,
                          scale: float) -> None:
    assert np.allclose(pnl, pnl0 - tcf, rtol=0.0,
                       atol=_DECOMP_ATOL * max(scale, 1.0)), \
        "PnL decomposition does not balance: total != directional - tc_fv"


def settle_delta(S: np.ndarray, times: np.ndarray, positions: np.ndarray,
                 r: float, q: float, tc: float, premium: np.ndarray,
                 liability: np.ndarray,
                 charge_final_unwind: bool = True) -> HedgeResult:
    """HEADLINE delta-only settlement of a short call hedged with the
    underlying only, liquidated at the last grid time (T' when hedging to a
    horizon). `liability` = per-path terminal mark of the short option there,
    computed by the caller (contract: true-DGP price at T'). Turnover = mean
    |delta change| per path. The additive decomposition is verified by an
    independent frictionless settlement."""
    liability = np.asarray(liability, float)
    legs = [(positions, S, q)]
    pnl, tcf = _settle_core(times, r, tc, premium, liability, legs, charge_final_unwind)
    pnl0, _ = _settle_core(times, r, 0.0, premium, liability, legs, charge_final_unwind)
    _assert_decomposition(pnl, pnl0, tcf, float(S[0, 0]))
    return HedgeResult(pnl, pnl0, tcf, _turnover(positions), positions)


def settle_delta_gamma(S: np.ndarray, ph: np.ndarray, times: np.ndarray,
                       u: np.ndarray, n_opt: np.ndarray, r: float,
                       q: float, tc: float, premium: np.ndarray,
                       liability: np.ndarray,
                       charge_final_unwind: bool = True) -> HedgeResult:
    """SECONDARY two-instrument settlement (underlying + hedge option); the
    same tc rate applies to traded value on both legs. `liability` as in
    settle_delta. Turnover is measured on the OPTION leg (mean |n change| per
    path), per the contract."""
    liability = np.asarray(liability, float)
    legs = [(u, S, q), (n_opt, ph, 0.0)]
    pnl, tcf = _settle_core(times, r, tc, premium, liability, legs, charge_final_unwind)
    pnl0, _ = _settle_core(times, r, 0.0, premium, liability, legs, charge_final_unwind)
    _assert_decomposition(pnl, pnl0, tcf, float(S[0, 0]))
    return HedgeResult(pnl, pnl0, tcf, _turnover(n_opt), n_opt)

# ---------------------------------------------------------------------------
# terminal liability mark (contract: true-DGP price at T', never intrinsic
# unless T' == tau0). Lazy oracle imports: the mark is benchmark truth, not a
# GreekProvider — providers still enter only through the interface above.
# ---------------------------------------------------------------------------

def heston_bates_terminal_mark(S_T: np.ndarray, v_T: np.ndarray, tau: float,
                               K: float, p: SimParams,
                               chunk: int = 2048) -> np.ndarray:
    """TRUE-DGP mark of the call with `tau` remaining at liquidation state
    (S_T', v_T'): Bates CF price (pure Heston when lambda_j = 0, bit-for-bit —
    the jump factor is skipped entirely) evaluated with per-path v0 = v_T'.
    oracle.bates_price_cf vectorizes per-path v0 exactly (v0 is affine in the
    log-CF); chunking only bounds the [paths x quadrature-nodes] workspace."""
    import oracle
    hp = oracle.HestonParams(kappa=p.kappa, theta=p.theta, xi=p.xi,
                             rho=p.rho, v0=p.v0)
    S_T = np.asarray(S_T, float)
    v_T = np.maximum(np.asarray(v_T, float), 0.0)   # QE can hit v = 0 exactly
    out = np.empty(S_T.size)
    for lo in range(0, S_T.size, chunk):
        sl = slice(lo, lo + chunk)
        out[sl] = oracle.bates_price_cf(S_T[sl], K, float(tau), hp, p.r, p.q,
                                        p.lambda_j, p.mu_j, p.sigma_j,
                                        v0=v_T[sl])
    return out


def merton_terminal_mark(S_T: np.ndarray, tau: float, K: float, mp: dict,
                         r: float, q: float) -> np.ndarray:
    """TRUE-DGP mark under the Merton DGP: closed-form Poisson-mixture price of
    the call with `tau` remaining at spot S_T' (variance is constant, so no
    per-path state beyond spot)."""
    import oracle
    return np.asarray(oracle.merton_price(
        np.asarray(S_T, float), K, float(tau), mp["sigma"], mp["lambda_j"],
        mp["mu_j"], mp["sigma_j"], r, q))

# ---------------------------------------------------------------------------
# risk statistics
# ---------------------------------------------------------------------------

def cvar(pnl: np.ndarray, level: float) -> float:
    """Expected shortfall: mean of the worst ceil((1-level)*n) losses (-pnl)."""
    losses = -np.asarray(pnl, float)
    n_tail = max(1, int(np.ceil((1.0 - level) * losses.size - 1e-12)))
    part = np.partition(losses, losses.size - n_tail)
    return float(part[losses.size - n_tail:].mean())


def bootstrap_cvar_se(pnl: np.ndarray, level: float, n_boot: int,
                      seed: int) -> float:
    """Bootstrap-over-paths SE of CVaR (path/bootstrap variance component;
    seed-to-seed variance is reported separately by aggregate_over_seeds)."""
    rng = np.random.default_rng([seed, _STREAM_BOOT])
    pnl = np.asarray(pnl, float)
    ests = np.empty(n_boot)
    for b in range(n_boot):
        ests[b] = cvar(pnl[rng.integers(0, pnl.size, pnl.size)], level)
    return float(ests.std(ddof=1))


def paired_bootstrap_cvar_diff(pnl_a: np.ndarray, pnl_b: np.ndarray,
                               level: float, n_boot: int, seed: int) -> dict:
    """CONFIRMATORY statistic: paired per-path bootstrap of the CVaR
    difference between two arms hedging the SAME CRN paths. Each replicate
    resamples ONE set of path indices and applies it to BOTH arrays — the
    common discretization/DGP noise shared by the arms cancels in the
    difference, which is what makes the pre-registered CI decidable. Returns
    {'diff': cvar(a) - cvar(b) (< 0 when a beats b under loss = -PnL),
     'ci_lo'/'ci_hi': percentile 95% CI on the difference,
     'rel_improvement': (cvar(b) - cvar(a)) / cvar(b) — the pre-registered
     relative-improvement reading (NaN when cvar(b) == 0)}."""
    pnl_a, pnl_b = np.asarray(pnl_a, float), np.asarray(pnl_b, float)
    assert pnl_a.shape == pnl_b.shape, \
        "paired bootstrap requires both arms on the same CRN paths"
    rng = np.random.default_rng([seed, _STREAM_PAIRED])
    n = pnl_a.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = cvar(pnl_a[idx], level) - cvar(pnl_b[idx], level)
    cvar_a, cvar_b = cvar(pnl_a, level), cvar(pnl_b, level)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(cvar_a - cvar_b),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "rel_improvement": (float((cvar_b - cvar_a) / cvar_b)
                                if cvar_b != 0.0 else float("nan"))}

# ---------------------------------------------------------------------------
# sweep drivers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Ctx:
    K: float
    T: float
    K_h: float
    T_h: float
    min_tau_h: float
    gamma_floor: float
    r: float
    q: float
    tiers: tuple
    risk: dict
    smoothing: dict | None
    charge_final_unwind: bool
    oracle_name: str = "oracle"
    baseline_name: str = "standard_pinn"
    pnl_dir: str | None = None          # risk.persist_pnl: per-cell PnL npz


# CSV column glossary (falsifier-relevant columns):
#   turnover          per-step MEAN |position change| per path, EXCLUDING the
#                     inception trade and final unwind (continuity column)
#   t_ex              CONTRACT falsifier statistic: mean over paths of
#                     total_traded(method) - total_traded(oracle), where
#                     total traded = |pos_0| + sum|diff| + |pos_last|
#                     (INCLUDING endpoints, unit price weighting)
#   t_ex_definition   "sum_incl_endpoints" (blank on delta-gamma rows)
#   pnl_vs_{baseline,oracle}_cvar_diff / _ci_lo / _ci_hi
#                     paired per-path bootstrap CVaR difference vs the
#                     reference arm on the same CRN paths (confirmatory
#                     statistic); blank on the reference's own rows and when
#                     the reference provider is absent
#   gap_closed        per-seed (cvar_base - cvar_method)/(cvar_base -
#                     cvar_oracle); denominator floored via gap_floor_frac
_T_EX_DEFINITION = "sum_incl_endpoints"
_PAIRED_REFS = (("baseline_name", "pnl_vs_baseline"),
                ("oracle_name", "pnl_vs_oracle"))


def _metrics_row(res: HedgeResult, method: str, tc: float, seed: int,
                 tag: dict, risk: dict) -> dict:
    level = risk["cvar_level"]
    return {**tag, "method": method, "tc": tc, "seed": seed,
            "mean_pnl": float(res.pnl_total.mean()),
            "cvar": cvar(res.pnl_total, level),
            "cvar_level": level,
            "cvar_boot_se": bootstrap_cvar_se(res.pnl_total, level,
                                              risk["bootstrap_B"], seed),
            "turnover": float(res.turnover.mean()),
            "tc_component": float(res.tc_paid_fv.mean()),
            "directional_component": float(res.pnl_directional.mean()),
            "n_paths": int(res.pnl_total.size)}


def _cell_slug(tag: dict, seed: int) -> str:
    parts = [f"seed={seed}"] + [f"{k}={v}" for k, v in tag.items() if v != ""]
    return "__".join(parts).replace("/", "-").replace(" ", "")


def _eval_methods(kind: str, S: np.ndarray, v: np.ndarray, times: np.ndarray,
                  providers: dict, ctx: _Ctx, tag: dict, seed: int,
                  liability: np.ndarray) -> list[dict]:
    """Evaluate every provider on one simulated cell. Positions/premiums are
    computed ONCE per method (they never depend on tc), then settled at every
    tc tier with ALL methods' per-path PnL held in memory per (cell, tc): the
    paired per-path bootstrap of the CVaR difference vs the baseline and
    oracle arms (the confirmatory statistic — the common CRN-path noise
    cancels only when both arms are resampled with the same indices) needs
    every arm's PnL simultaneously."""
    oracle_cache, prem_override = None, None
    if ctx.oracle_name in providers:
        # contract pnl convention: initial premium = theta_train ORACLE price
        # for ALL arms (incl. smoothed variants), computed ONCE per cell
        if kind == "delta_only":
            # doubles as the T_ex reference: the oracle hedge's per-path
            # total-traded on the SAME paths (positions never depend on tc)
            oracle_cache = delta_positions(S, v, times,
                                           providers[ctx.oracle_name], ctx.K, ctx.T)
            prem_override = oracle_cache[1]
        else:
            out0 = providers[ctx.oracle_name].evaluate(S[:, 0], v[:, 0],
                                                       ctx.T, ctx.K)
            prem_override = np.array(np.broadcast_to(
                np.asarray(out0["price"], float), (S.shape[0],)))
    else:
        _log.warning("PREMIUM CONVENTION VIOLATED: no provider named %r — "
                     "falling back to per-provider inception premiums; every "
                     "row carries premium_convention_ok=False", ctx.oracle_name)
        if kind == "delta_only":
            _log.warning("T_ex: no provider named %r in providers; t_ex = NaN",
                         ctx.oracle_name)
    premium_ok = prem_override is not None

    # ---- phase 1: positions once per method/variant (never per tc) --------
    if kind == "delta_only":
        oracle_traded = (_total_traded(oracle_cache[0])
                         if oracle_cache is not None else None)
        specs: dict[str, tuple] = {}
        for name, prov in providers.items():
            pos, premium = (oracle_cache if name == ctx.oracle_name
                            else delta_positions(S, v, times, prov, ctx.K, ctx.T,
                                                 premium_override=prem_override))
            specs[name] = (pos, premium)
            if ctx.smoothing and name in ctx.smoothing.get("applies_to", []):
                specs[f"{name}_smoothed"] = (
                    smooth_positions(pos, ctx.smoothing.get("ema_alpha"),
                                     ctx.smoothing["no_trade_band"]), premium)
        extras = {
            vname: {"t_ex": (float((_total_traded(p_) - oracle_traded).mean())
                             if oracle_traded is not None else float("nan")),
                    "t_ex_definition": _T_EX_DEFINITION}
            for vname, (p_, _prem) in specs.items()}

        def _settle(vname: str, tc: float) -> HedgeResult:
            p_, prem = specs[vname]
            return settle_delta(S, times, p_, ctx.r, ctx.q, tc, prem,
                                liability, ctx.charge_final_unwind)
    else:
        specs = {name: dg_positions(S, v, times, prov, ctx.K, ctx.T, ctx.K_h,
                                    ctx.T_h, ctx.min_tau_h, ctx.gamma_floor,
                                    premium_override=prem_override)
                 for name, prov in providers.items()}
        # T_ex is delta-only only
        extras = {name: {"t_ex": "", "t_ex_definition": ""} for name in specs}

        def _settle(vname: str, tc: float) -> HedgeResult:
            u, n_opt, ph, prem = specs[vname]
            return settle_delta_gamma(S, ph, times, u, n_opt, ctx.r, ctx.q,
                                      tc, prem, liability,
                                      ctx.charge_final_unwind)

    # ---- phase 2: settle per tc, emit rows with paired CVaR-diff stats ----
    rows: list[dict] = []
    pnl_bank: dict[str, np.ndarray] = {}
    for tc in ctx.tiers:
        results = {vname: _settle(vname, tc) for vname in specs}
        for vname, res in results.items():
            row = _metrics_row(res, vname, tc, seed, tag, ctx.risk)
            row.update(extras[vname])
            row["premium_convention_ok"] = premium_ok
            for ref_attr, prefix in _PAIRED_REFS:
                ref = getattr(ctx, ref_attr)
                if ref in results and vname != ref:
                    st = paired_bootstrap_cvar_diff(
                        res.pnl_total, results[ref].pnl_total,
                        ctx.risk["cvar_level"], ctx.risk["bootstrap_B"], seed)
                    row[f"{prefix}_cvar_diff"] = st["diff"]
                    row[f"{prefix}_ci_lo"] = st["ci_lo"]
                    row[f"{prefix}_ci_hi"] = st["ci_hi"]
                else:   # blank on the reference's own rows / absent reference
                    row[f"{prefix}_cvar_diff"] = ""
                    row[f"{prefix}_ci_lo"] = ""
                    row[f"{prefix}_ci_hi"] = ""
            rows.append(row)
            if ctx.pnl_dir:
                pnl_bank[f"{vname}__tc{tc}"] = res.pnl_total
    if ctx.pnl_dir:
        os.makedirs(ctx.pnl_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(ctx.pnl_dir, _cell_slug(tag, seed) + ".npz"),
            **pnl_bank)
    return rows

# ---------------------------------------------------------------------------
# frozen path banks: freeze every simulated cell once, hedge every arm on the
# SAME loaded paths (CRN invariant). ONE bank serves headline and secondary —
# a cell's paths depend on (DGP, seed, engine grid), never the hedge kind.
# ---------------------------------------------------------------------------

def _dgp_params_dict(dgp_kind: str, dgp, r: float, q: float) -> dict:
    """Canonical DGP-identity dict for a path cell (JSON round-trip stable):
    Heston/Bates -> every SimParams field (already carries r, q and the jump
    params); Merton -> the sigma/lambda_j/mu_j/sigma_j params plus shared r, q.
    Both bank_write and _BankLoader build it identically, so the on-disk and the
    requested DGPs compare with `==` after a json.dumps/loads round trip."""
    if dgp_kind == "merton":
        d = {k: float(dgp[k]) for k in ("sigma", "lambda_j", "mu_j", "sigma_j")}
        d["r"], d["q"] = float(r), float(q)
        return d
    return {f.name: float(getattr(dgp, f.name)) for f in dataclasses.fields(dgp)}


def _resolved_config_hash(cfg: dict) -> str:
    """Provenance sha256 over the resolved benchmark + engine + derived seeds,
    stamped into bank_manifest.json. NOT gated at load time: a trimmed sub-sweep
    (run_confirmatory, a single-cell test) legitimately hashes differently while
    still loading a superset bank, so the per-cell sha + DGP checks decide."""
    payload = {"benchmark": cfg["benchmark"], "engine": cfg["engine"],
               "seeds": list(cfg["derived"]["seeds"])}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _iter_sim_cells(cfg: dict):
    """Yield (seed, tag, dgp_kind, dgp) for every simulated path cell of the
    misspecification sweep, in _run_sweep's EXACT run order (seed outer, then
    perturbation direction x magnitude, then the Bates severity sweep, then
    Merton). `dgp` is a SimParams (dgp_kind 'heston_qe') or the Merton params
    dict (dgp_kind 'merton'). bank_write, the engine loader, and the runners all
    share this generator so the on-disk cells and the requested cells cannot
    drift."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    mis = hs["misspecification"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    base = SimParams.from_regime(bm["regimes"][mis["train_params"]], r, q)
    dirs_cfg = eng["misspecification"]["directions"]
    blank = {"direction": "", "magnitude": "", "lambda_j": "", "sigma_j": "",
             "in_model": ""}
    for seed in cfg["derived"]["seeds"]:
        for direction in mis["perturbations"]:
            for mag in eng["misspecification"]["magnitudes"]:
                hp = perturb_params(base, direction, mag, dirs_cfg)
                tag = {**blank, "sweep": "perturbation", "direction": direction,
                       "magnitude": mag, "in_model": mag == 0}
                yield seed, tag, "heston_qe", hp
        if "bates_severity_sweep" in mis["cross_model"]:
            sw = bm["models"]["bates"]["jump_severity_sweep"]
            for lam in sw["lambda_j"]:
                for sj in (sw["sigma_j"] if lam > 0.0 else sw["sigma_j"][:1]):
                    bp = dataclasses.replace(base, lambda_j=lam, mu_j=sw["mu_j"],
                                             sigma_j=sj)
                    tag = {**blank, "sweep": "bates", "lambda_j": lam,
                           "sigma_j": sj}
                    yield seed, tag, "heston_qe", bp
        if "merton" in mis["cross_model"]:
            tag = {**blank, "sweep": "merton"}
            yield seed, tag, "merton", bm["models"]["merton"]["params"]


class _BankLoader:
    """Reader for a frozen path bank written by bank_write. Every load HARD-CHECKS
    the npz file's sha256 against the manifest (tamper detection) and that the
    stored DGP kind/params and array shape match the cell being requested, then
    returns (times, S, v). Any mismatch raises — the engine NEVER silently
    resimulates when simulation.paths_dir is set."""

    def __init__(self, paths_dir, cfg: dict) -> None:
        from make_labels import _sha256_file           # lazy (providers.py-style)
        self._sha256_file = _sha256_file
        self.dir = Path(paths_dir)
        man_path = self.dir / "bank_manifest.json"
        if not man_path.exists():
            raise FileNotFoundError(
                f"path bank manifest not found: {man_path} "
                "(simulation.paths_dir must point at a bank_write output)")
        self.manifest = json.loads(man_path.read_text())
        self.files = self.manifest["files"]
        self.r = float(cfg["benchmark"]["grid"]["r"])
        self.q = float(cfg["benchmark"]["grid"]["q"])

    def load(self, slug: str, dgp_kind: str, dgp, n_paths: int, n_steps: int):
        if slug not in self.files:
            head = sorted(self.files)[:8]
            raise KeyError(
                f"path bank {self.dir} has no cell {slug!r}; available "
                f"({len(self.files)}): {head}{' ...' if len(self.files) > 8 else ''}")
        entry = self.files[slug]
        npz_path = self.dir / entry["file"]
        got_sha = self._sha256_file(str(npz_path))
        if got_sha != entry["sha256"]:
            raise ValueError(
                f"path bank cell {slug!r} sha256 mismatch (file {got_sha} != "
                f"manifest {entry['sha256']}): bank tampered — refusing to load")
        data = np.load(npz_path, allow_pickle=False)
        want = _dgp_params_dict(dgp_kind, dgp, self.r, self.q)
        got = json.loads(str(data["dgp_params_json"]))
        if str(data["dgp_kind"]) != dgp_kind or got != want:
            raise ValueError(
                f"path bank cell {slug!r} DGP mismatch: bank has "
                f"{str(data['dgp_kind'])}/{got} but the sweep requested "
                f"{dgp_kind}/{want} — refusing to hedge on the wrong paths")
        S = data["S"]
        if S.shape != (n_paths, n_steps + 1):
            raise ValueError(
                f"path bank cell {slug!r} shape {S.shape} != expected "
                f"{(n_paths, n_steps + 1)}: n_paths/n_steps drifted from the bank")
        return data["times"], S, data["v"]


def bank_write(cfg: dict, out_dir: str) -> dict:
    """Freeze every simulated path cell of the misspecification sweep to a
    per-cell npz under a NON-frozen out_dir, plus bank_manifest.json carrying
    each file's sha256 and the resolved-config hash. Cell content depends only
    on (DGP, seed, engine grid) — never on the hedge kind — so one bank serves
    the headline and secondary runs alike. A human later PROMOTES the directory
    into data/paths/<tag> and points simulation.paths_dir at it (freezing is a
    human step; out_dir must not contain 'frozen', reusing make_labels'
    frozen-path refusal)."""
    from make_labels import _assert_not_frozen, _git_rev, _sha256_file
    out = _assert_not_frozen(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    spec = {**eng["contract"], **{k: hs["instrument"][m] for k, m in
                                  (("S0", "S0"), ("K", "K"), ("T", "tau0"))
                                  if m in hs["instrument"]}}
    S0, T = spec["S0"], spec["T"]
    T_prime = float(eng["horizon"]["T_prime"]) if "horizon" in eng else float(T)
    n_steps = int(round(T_prime * eng["rebalancing"]["frequency_per_year"]))
    n_paths = eng["simulation"]["n_paths"]
    psi_c = eng["simulation"]["psi_c"]
    base = SimParams.from_regime(bm["regimes"][hs["misspecification"]["train_params"]],
                                 r, q)
    files: dict[str, dict] = {}
    for seed, tag, dgp_kind, dgp in _iter_sim_cells(cfg):
        if dgp_kind == "merton":
            times, S, v = simulate_merton(dgp, S0, T_prime, n_steps, n_paths,
                                          seed, r, q)
            scheme = "merton_euler"
        else:
            times, S, v = simulate_heston_qe(dgp, S0, T_prime, n_steps, n_paths,
                                             seed, psi_c)
            scheme = "qe_andersen"
        if tag["sweep"] == "bates" and tag["lambda_j"] == 0.0:
            _, S_h, v_h = simulate_heston_qe(base, S0, T_prime, n_steps, n_paths,
                                             seed, psi_c)
            assert np.array_equal(S, S_h) and np.array_equal(v, v_h), \
                "Bates lambda_j=0 must recover Heston bit-for-bit"
        slug = _cell_slug(tag, seed)
        params = _dgp_params_dict(dgp_kind, dgp, r, q)
        fname = slug + ".npz"
        np.savez_compressed(
            out / fname, times=times, S=S, v=v, dgp_kind=dgp_kind,
            dgp_params_json=json.dumps(params, sort_keys=True), seed=int(seed),
            scheme=scheme, psi_c=float(psi_c))
        files[slug] = {"file": fname, "sha256": _sha256_file(str(out / fname)),
                       "dgp_kind": dgp_kind, "seed": int(seed),
                       "dgp_params": params}
    manifest = {
        "kind": "hedging_path_bank",
        "config_hash": _resolved_config_hash(cfg),
        "contract_path": cfg["derived"].get("contract_path"),
        "engine_path": cfg["derived"].get("engine_path"),
        "scheme": "qe_andersen", "S0": float(S0), "T_prime": float(T_prime),
        "n_paths": int(n_paths), "n_steps": int(n_steps), "psi_c": float(psi_c),
        "n_cells": len(files), "git_rev": _git_rev(), "files": files,
    }
    (out / "bank_manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"out_dir": str(out), "n_cells": len(files), "manifest": manifest}


def _run_sweep(cfg: dict, providers: dict, kind: str, out_name: str,
               out_dir: str | None) -> list[dict]:
    """Misspecified-dynamics driver: perturbation-magnitude TREND (train-params
    -> swept hedge-params) plus cross-model (Bates severity sweep incl. the
    lambda_j=0 Heston-recovery assertion, and Merton), all TC tiers, all seeds.
    Paths run to the hedge horizon T' = engine horizon.T_prime (< tau0); the
    short option is liquidated there and marked at its TRUE-DGP price with
    tau0 - T' remaining (contract pnl_convention.terminal_mark)."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    mis = hs["misspecification"]
    tiers = tuple(hs["transaction_costs"]["tiers"])
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    base = SimParams.from_regime(bm["regimes"][mis["train_params"]], r, q)
    # instrument: the CONTRACT wins over the engine block (v6 defines S0/K/tau0;
    # the engine `contract` block remains a fallback for engine-only overrides)
    spec = {**eng["contract"], **{k: hs["instrument"][m] for k, m in
                                  (("S0", "S0"), ("K", "K"), ("T", "tau0"))
                                  if m in hs["instrument"]}}
    S0, K, T = spec["S0"], spec["K"], spec["T"]
    has_horizon = "horizon" in eng
    T_prime = float(eng["horizon"]["T_prime"]) if has_horizon else float(T)
    if T_prime > T + 1e-12:
        raise ValueError(f"horizon.T_prime={T_prime} exceeds instrument "
                         f"tau0={T}: cannot hedge past expiry")
    tau_settle = float(T - T_prime)     # option maturity remaining at T'
    expiry_settlement = tau_settle <= 1e-12
    if not expiry_settlement:
        # providers are only ever called with tau >= tau0 - T' (+ one rebalance
        # step); assert that stays inside the oracle-trusted / grid tau range
        grid_tau_min = float(bm["grid"]["tau"]["min"])
        assert tau_settle >= grid_tau_min > 0.0, (
            f"tau0 - T_prime = {tau_settle:.4f} is below the grid/oracle-trusted "
            f"tau floor {grid_tau_min}: providers would be evaluated outside "
            "the trusted maturity range")
    n_steps = int(round(T_prime * eng["rebalancing"]["frequency_per_year"]))
    n_paths = eng["simulation"]["n_paths"]
    psi_c = eng["simulation"]["psi_c"]

    def _liability(times: np.ndarray, S: np.ndarray, v: np.ndarray,
                   heston_dgp: SimParams | None = None,
                   merton_dgp: dict | None = None) -> np.ndarray:
        """Per-path terminal mark at times[-1]; raises rather than silently
        settling at expiry when the engine declares a T' < tau0 horizon."""
        if has_horizon and abs(float(times[-1]) - T_prime) > 1e-9:
            raise RuntimeError(
                f"horizon.T_prime={T_prime} declared but the path grid settles "
                f"at t={float(times[-1])}: refusing silent expiry settlement")
        if not expiry_settlement:
            if merton_dgp is not None:
                return merton_terminal_mark(S[:, -1], tau_settle, K,
                                            merton_dgp, r, q)
            return heston_bates_terminal_mark(S[:, -1], v[:, -1], tau_settle,
                                              K, heston_dgp)
        if has_horizon and not np.isclose(T_prime, T):
            raise RuntimeError("engine horizon.T_prime < tau0 but settlement "
                               "reached expiry — refusing intrinsic fallback")
        return np.maximum(S[:, -1] - K, 0.0)   # bridge case: T' == tau0 exactly
    ctx = _Ctx(K=K, T=T, K_h=S0, T_h=eng["hedge_option"]["maturity_factor"] * T,
               min_tau_h=eng["hedge_option"]["min_tau"],
               gamma_floor=eng["hedge_option"]["gamma_floor"],
               r=r, q=q, tiers=tiers, risk=eng["risk"],
               smoothing=eng["smoothing_baseline"] if kind == "delta_only" else None,
               charge_final_unwind=eng["rebalancing"]["charge_final_unwind"],
               oracle_name=eng.get("oracle_provider_name", "oracle"),
               baseline_name=eng.get("baseline_provider_name", "standard_pinn"),
               pnl_dir=(os.path.join(out_dir, f"pnl_{out_name}")
                        if out_dir and eng["risk"].get("persist_pnl", False)
                        else None))
    paths_dir = eng["simulation"].get("paths_dir")
    loader = _BankLoader(paths_dir, cfg) if paths_dir else None

    def _paths(tag: dict, seed: int, dgp_kind: str, dgp):
        """Simulate the cell on the fly, or load it from a frozen path bank when
        simulation.paths_dir is set (sha + DGP + shape checked; never a silent
        resimulate). Loaded arrays are bit-identical to the simulated ones, so a
        bank-backed run reproduces an on-the-fly run row for row."""
        if loader is not None:
            return loader.load(_cell_slug(tag, seed), dgp_kind, dgp, n_paths, n_steps)
        if dgp_kind == "merton":
            return simulate_merton(dgp, S0, T_prime, n_steps, n_paths, seed, r, q)
        return simulate_heston_qe(dgp, S0, T_prime, n_steps, n_paths, seed, psi_c)

    rows: list[dict] = []
    for seed, tag, dgp_kind, dgp in _iter_sim_cells(cfg):
        times, S, v = _paths(tag, seed, dgp_kind, dgp)
        if loader is None and tag["sweep"] == "bates" and tag["lambda_j"] == 0.0:
            _, S_h, v_h = simulate_heston_qe(base, S0, T_prime, n_steps,
                                             n_paths, seed, psi_c)
            assert np.array_equal(S, S_h) and np.array_equal(v, v_h), \
                "Bates lambda_j=0 must recover Heston bit-for-bit"
        liab = (_liability(times, S, v, merton_dgp=dgp) if dgp_kind == "merton"
                else _liability(times, S, v, heston_dgp=dgp))
        rows += _eval_methods(kind, S, v, times, providers, ctx, tag, seed, liab)
    gap_floor_frac = float(eng["risk"].get("gap_floor_frac", 0.01))
    rows = add_gap_closed(rows, eng.get("baseline_provider_name", "standard_pinn"),
                          eng.get("oracle_provider_name", "oracle"),
                          gap_floor_frac)
    if out_dir:
        log_resolved_config(cfg, out_dir)
        # hidden per-seed columns (leading "_": gap numerator/denominator
        # carriers for the ratio-of-means aggregate) stay out of the CSVs
        write_rows_csv([{k: val for k, val in row.items()
                         if not k.startswith("_")} for row in rows],
                       os.path.join(out_dir, f"{out_name}_per_seed.csv"))
        write_rows_csv(aggregate_over_seeds(rows, gap_floor_frac),
                       os.path.join(out_dir, f"{out_name}_agg.csv"))
    return rows


def run_headline(cfg: dict, providers: dict, out_dir: str | None = None) -> list[dict]:
    """HEADLINE: delta-only hedging (underlying only; Gamma is supervised
    information, not a traded instrument). Rebalance frequency is FIXED by
    config so Greek accuracy is the only varied lever. Misspecified sweep runs
    FIRST (contract F5 run order)."""
    return _run_sweep(cfg, providers, "delta_only", "headline_delta_only", out_dir)


def run_secondary_delta_gamma(cfg: dict, providers: dict,
                              out_dir: str | None = None) -> list[dict]:
    """SECONDARY CONTRAST (not the headline): two-instrument delta-gamma hedge
    with an ATM-at-inception hedge option; turnover on the option leg."""
    return _run_sweep(cfg, providers, "delta_gamma", "secondary_delta_gamma",
                      out_dir)

# ---------------------------------------------------------------------------
# aggregation / output
# ---------------------------------------------------------------------------

_METRIC_FIELDS = ("mean_pnl", "cvar", "turnover", "tc_component",
                  "directional_component")
# numeric when defined, "" when not
_OPTIONAL_METRIC_FIELDS = ("t_ex", "gap_closed",
                           "pnl_vs_baseline_cvar_diff", "pnl_vs_baseline_ci_lo",
                           "pnl_vs_baseline_ci_hi",
                           "pnl_vs_oracle_cvar_diff", "pnl_vs_oracle_ci_lo",
                           "pnl_vs_oracle_ci_hi")


def _gap_denominator_floor(cvar_oracle: float, gap_floor_frac: float) -> float:
    """gap_closed denominator floor: gap_floor_frac * |cvar_oracle| — on the
    CVaR scale, not an absolute epsilon — with a 1.0 absolute fallback when
    the oracle CVaR is itself ~ 0 (no scale to anchor the fraction to)."""
    floor = gap_floor_frac * abs(float(cvar_oracle))
    return floor if floor > 1e-12 else 1.0


def add_gap_closed(rows: list[dict], baseline_name: str, oracle_name: str,
                   gap_floor_frac: float) -> list[dict]:
    """v6 headline scale-free metric: within each (tag fields, tc, seed) cell,
    gap_closed = (cvar_baseline - cvar_method) / (cvar_baseline - cvar_oracle).
    Blank ("") for the whole cell when the baseline or oracle row is missing,
    or when |denominator| < gap_floor_frac * |cvar_oracle| (fallback 1.0
    absolute if oracle cvar ~ 0) — undefined, not infinity: a seed where
    baseline ~ oracle must not emit an exploding ratio. Whenever baseline and
    oracle are both present, also attaches HIDDEN per-seed columns (stripped
    from CSVs): _gap_num = cvar_base - cvar_method, _gap_den = cvar_base -
    cvar_oracle, _cvar_oracle — carriers for the ratio-of-mean-gaps aggregate
    in aggregate_over_seeds. Mutates and returns `rows`."""
    skip = set(_METRIC_FIELDS) | set(_OPTIONAL_METRIC_FIELDS) | {
        "method", "cvar_level", "cvar_boot_se", "n_paths"}
    key_fields = [k for k in rows[0]
                  if k not in skip and not k.startswith("_")]  # keeps tc, seed
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in key_fields), []).append(row)
    for grp in groups.values():
        by_method = {g["method"]: g for g in grp}
        base, orac = by_method.get(baseline_name), by_method.get(oracle_name)
        if base is None or orac is None:
            for g in grp:
                g["gap_closed"] = ""
            continue
        denom = base["cvar"] - orac["cvar"]
        for g in grp:
            g["_gap_num"] = float(base["cvar"] - g["cvar"])
            g["_gap_den"] = float(denom)
            g["_cvar_oracle"] = float(orac["cvar"])
        floor = _gap_denominator_floor(orac["cvar"], gap_floor_frac)
        for g in grp:
            g["gap_closed"] = (float((base["cvar"] - g["cvar"]) / denom)
                               if abs(denom) >= floor else "")
    return rows


def aggregate_over_seeds(rows: list[dict],
                         gap_floor_frac: float = 0.01) -> list[dict]:
    """Collapse seeds: per-metric mean and seed-to-seed std (ddof=1), keeping
    the per-seed path-bootstrap SE as a SEPARATE component (cvar_boot_se_mean).
    Optional metrics (t_ex, paired-bootstrap columns) aggregate numerically
    when numeric and stay blank ("") when blank in the per-seed rows.
    gap_closed aggregates TWO ways: gap_closed_mean is the RATIO OF MEAN GAPS,
    mean_over_seeds(cvar_base - cvar_method) / mean_over_seeds(cvar_base -
    cvar_oracle), from the hidden _gap_num/_gap_den columns with the same
    gap_floor_frac denominator floor (a single seed with baseline ~ oracle
    cannot blow it up); gap_closed_mean_of_ratios keeps the legacy
    mean-of-per-seed-ratios for comparison."""
    metrics = _METRIC_FIELDS
    optional = tuple(m for m in _OPTIONAL_METRIC_FIELDS if m in rows[0])
    drop = set(metrics) | set(optional) | {"seed", "cvar_boot_se", "n_paths"}
    key_fields = [k for k in rows[0]
                  if k not in drop and not k.startswith("_")]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in key_fields), []).append(row)

    def _all_num(vals: list) -> bool:
        return all(isinstance(v_, (int, float)) for v_ in vals)

    out = []
    for key, grp in groups.items():
        rec = dict(zip(key_fields, key))
        rec["n_seeds"] = len(grp)
        for mk in metrics:
            vals = np.array([g[mk] for g in grp], float)
            rec[f"{mk}_mean"] = float(vals.mean())
            rec[f"{mk}_seed_std"] = float(vals.std(ddof=1)) if len(grp) > 1 else 0.0
        for mk in optional:
            if mk == "gap_closed":
                continue                    # two-way aggregate, handled below
            vals = [g[mk] for g in grp]
            if _all_num(vals):
                arr = np.array(vals, float)
                rec[f"{mk}_mean"] = float(arr.mean())
                rec[f"{mk}_seed_std"] = float(arr.std(ddof=1)) if len(grp) > 1 else 0.0
            else:
                rec[f"{mk}_mean"] = ""
                rec[f"{mk}_seed_std"] = ""
        if "gap_closed" in rows[0]:
            nums = [g.get("_gap_num", "") for g in grp]
            dens = [g.get("_gap_den", "") for g in grp]
            oracs = [g.get("_cvar_oracle", "") for g in grp]
            if _all_num(nums) and _all_num(dens) and _all_num(oracs):
                den_mean = float(np.mean(dens))
                floor = _gap_denominator_floor(float(np.mean(oracs)),
                                               gap_floor_frac)
                rec["gap_closed_mean"] = (float(np.mean(nums)) / den_mean
                                          if abs(den_mean) >= floor else "")
            else:
                rec["gap_closed_mean"] = ""
            ratios = [g["gap_closed"] for g in grp]
            if _all_num(ratios):
                arr = np.array(ratios, float)
                rec["gap_closed_mean_of_ratios"] = float(arr.mean())
                rec["gap_closed_seed_std"] = \
                    float(arr.std(ddof=1)) if len(grp) > 1 else 0.0
            else:
                rec["gap_closed_mean_of_ratios"] = ""
                rec["gap_closed_seed_std"] = ""
        rec["cvar_boot_se_mean"] = float(np.mean([g["cvar_boot_se"] for g in grp]))
        out.append(rec)
    return out


def write_rows_csv(rows: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path

# ---------------------------------------------------------------------------
# delta-path overlay diagnostic
# ---------------------------------------------------------------------------

_OVERLAY_COLORS = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7",
                   "#e34948", "#e87ba4", "#eb6834")   # fixed categorical order


def plot_delta_path_overlay(times: np.ndarray, positions_by_method: dict,
                            out_path: str, path_idx: int = 0) -> str:
    """Overlay hedge-position trajectories of several methods on one simulated
    path (delta-jitter visualisation, e.g. sobolev vs standard_pinn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for (name, pos), color in zip(positions_by_method.items(), _OVERLAY_COLORS):
        ax.step(times[:pos.shape[1]], pos[path_idx], where="post", label=name,
                lw=1.4, color=color)
    ax.set_xlabel("t (years)")
    ax.set_ylabel("hedge position (delta)")
    ax.grid(True, lw=0.4, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_delta_overlay(cfg: dict, providers: dict, out_path: str,
                      direction: str | None = None, magnitude: float = 0.0,
                      seed: int | None = None) -> str:
    """Simulate one small path set (misspecified if `direction` given) and plot
    each provider's delta path on the configured overlay path."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    mis = hs["misspecification"]
    base = SimParams.from_regime(bm["regimes"][mis["train_params"]], r, q)
    if direction is not None:
        base = perturb_params(base, direction, magnitude,
                              eng["misspecification"]["directions"])
    spec = {**eng["contract"], **{k: hs["instrument"][m] for k, m in
                                  (("S0", "S0"), ("K", "K"), ("T", "tau0"))
                                  if m in hs["instrument"]}}
    T_prime = float(eng["horizon"]["T_prime"]) if "horizon" in eng \
        else float(spec["T"])
    n_steps = int(round(T_prime * eng["rebalancing"]["frequency_per_year"]))
    times, S, v = simulate_heston_qe(
        base, spec["S0"], T_prime, n_steps, eng["overlay"]["n_paths"],
        cfg["derived"]["seeds"][0] if seed is None else seed,
        eng["simulation"]["psi_c"])
    overlays = {name: delta_positions(S, v, times, prov, spec["K"], spec["T"])[0]
                for name, prov in providers.items()}
    return plot_delta_path_overlay(times, overlays, out_path,
                                   eng["overlay"]["path_idx"])
