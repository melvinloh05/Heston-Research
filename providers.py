"""(price, Greeks) providers for the hedging engine + pathwise oracle trust check.

ORACLE-HEDGER SEMANTICS (pinned; do not reinterpret):
`HestonCFProvider` constructed from the THETA_TRAIN parameters is the
benchmark's "oracle" hedger: EXACT Greeks of the theta_train HESTON MODEL —
the model every arm was trained under — evaluated at the realized path state
(S_t, v_t, tau). It is NOT the true data-generating process. Under
misspecified dynamics its deltas are wrong in exactly the way a
zero-approximation-error theta_train model's would be: it is the
APPROXIMATION-ERROR-FREE CEILING of the model class, so any gap between a
PINN arm and this provider is pure approximation/training error, never model
error. The excess-turnover statistic T_ex and the gap_closed metric are
DEFINED against this provider (hedging_config.yaml oracle_provider_name), and
the contract PnL convention marks EVERY arm's inception premium at this
provider's theta_train price.

Standalone by design: this module imports oracle only. Hedging_backtest is
imported LAZILY inside sample_qe_states (that file is under concurrent
modification; providers must stay importable — and HestonCFProvider usable —
without it). Interface conformance with Hedging_backtest.GreekProvider is
structural (runtime_checkable Protocol) and asserted in test_providers.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import oracle
from oracle import HestonParams

_STREAM_SPOTCHECK = 3   # rng stream id; distinct from the engine's 0/1/2


class HestonCFProvider:
    """GreekProvider backed by oracle.heston_greeks_cf_v0 (analytic trap-free
    CF, leg A): exact theta_train Heston price/delta/gamma at pathwise state
    (S, v, tau), plus vega/vanna for diagnostics.

    Pathwise variance handling: each point is priced at initial variance
    v0 := max(v, v_floor). QE paths hit v = 0 EXACTLY (the exponential branch
    of Andersen's scheme has an atom at zero), and the GreekProvider contract
    requires finite output for all v >= 0, so flooring is this provider's
    responsibility; v_floor = 1e-6 is far below any economically distinct
    variance at the trusted tau range. Conceptually this builds a per-call
    HestonParams with v0 := floored pathwise v for every point — implemented
    exactly (no v-binning) via the affine-v0 identity: the CF blocks are
    evaluated once per (tau, u-grid) and only exp(c) is per-point, so the cost
    is O(paths * quadrature-nodes) broadcasting, not O(paths) CF setups.
    `chunk` bounds the [points x nodes] complex workspace.
    """

    def __init__(self, params: HestonParams, r: float, q: float,
                 v_floor: float = 1e-6, chunk: int = 2048,
                 n_panel: int = 12, n_node: int = 64) -> None:
        self.params = params
        self.r = float(r)
        self.q = float(q)
        self.v_floor = float(v_floor)
        self.chunk = int(chunk)
        self.n_panel = int(n_panel)
        self.n_node = int(n_node)

    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float,
                 K: float) -> dict:
        """Exact theta_train Greeks at (S, max(v, v_floor), tau, K); output
        arrays broadcast to S.shape.

        REPRODUCIBILITY, precisely (measured 2026-08-20, docs/CODE_AUDIT_2026-08-20.md):
        there is no randomness here — the quadrature is a pure function of its inputs —
        but repeated identical calls are reproducible only to ABOUT ONE ULP, not
        bitwise: on real path states max |delta - delta| across repeat calls is
        ~1.8e-15, and the wobble persists with the BLAS thread count pinned to 1. It is
        floating-point reduction order inside the vectorised CF evaluation, not state
        carried between calls. Nothing downstream is affected at 1e-15 (thirteen orders
        below any reported effect), but a test that asserts `array_equal` on this
        output WILL flake; assert `assert_allclose(..., rtol=0, atol=1e-12)` instead.
        The earlier docstring claimed bit-equal outputs and that claim was wrong.
        """
        S, v = np.broadcast_arrays(np.asarray(S, float), np.asarray(v, float))
        shape = S.shape
        Sf = S.ravel()
        vf = np.maximum(v.ravel(), self.v_floor)
        names = ("price", "delta", "gamma", "vega", "vanna")
        out = {nm: np.empty(Sf.size) for nm in names}
        for lo in range(0, Sf.size, self.chunk):
            sl = slice(lo, lo + self.chunk)
            g = oracle.heston_greeks_cf_v0(Sf[sl], K, float(tau), self.params,
                                           self.r, self.q, v0=vf[sl],
                                           n_panel=self.n_panel,
                                           n_node=self.n_node)
            for nm in names:
                out[nm][sl] = getattr(g, nm)
        return {nm: out[nm].reshape(shape) for nm in names}


# ---------------------------------------------------------------------------
# Pathwise trust check: the oracle legs are cross-validated on the config
# (S, K, tau) GRID at the regime v0, but the hedging engine evaluates the
# provider at PATH states (S_t, v_t, tau_t) never visited by that grid. The
# spot check below closes that gap by re-validating the provider against the
# independent COS leg at states drawn from actual QE paths.
# ---------------------------------------------------------------------------

@dataclass
class SpotCheckResult:
    """frac_ok / n count the AWAY band (v >= low_v_band, the gated one);
    frac_ok_low_v / n_low_v report the v < low_v_band band separately (NaN /
    0 when no such state was sampled). `worst` lists the n_worst largest
    relative errors across both bands, worst first."""
    frac_ok: float
    frac_ok_low_v: float
    n: int
    n_low_v: int
    worst: list
    note: str


def _cos_fd_reference(S: float, v0: float, tau: float, K: float,
                      p: HestonParams, r: float, q: float,
                      rel_h: float = 2e-2) -> tuple[float, float, float]:
    """Independent (price, delta, gamma) at ONE path state: the COS pricer
    (oracle leg B — shares no code with the CF leg) evaluated with v0 := the
    state's variance, spot-bumped with 4th-order 5-point stencils at steps h
    and h/2 and Richardson-extrapolated ((16*D_{h/2} - D_h)/15, error O(h^6));
    price is the unbumped COS value."""
    h = rel_h * S
    off = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    spots = np.concatenate([S + off * h, S + off * (0.5 * h)])
    f = oracle._cos_price(spots, np.full(spots.size, K), tau, p.kappa, p.theta,
                          p.xi, p.rho, v0, r, q)
    d1, g1 = oracle._fd4(f[0], f[1], f[2], f[3], f[4], h)
    d2, g2 = oracle._fd4(f[5], f[6], f[7], f[8], f[9], 0.5 * h)
    return float(f[2]), float((16.0 * d2 - d1) / 15.0), float((16.0 * g2 - g1) / 15.0)


def sample_qe_states(hp: HestonParams, r: float, q: float, S0: float,
                     tau0: float, T_prime: float, n_steps: int, n_paths: int,
                     seed: int, psi_c: float = 1.5, n_states: int = 2000
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, v, tau) states drawn uniformly from an actual QE path bank —
    exactly the state distribution the hedging engine feeds providers:
    rebalance steps i = 0..n_steps-1 with tau_i = tau0 - t_i (the terminal
    step belongs to the liability mark, not the provider). Deterministic in
    `seed`: the path simulation uses the engine's own streams and the state
    sampler its own [seed, _STREAM_SPOTCHECK] stream. Hedging_backtest is
    imported lazily (see module docstring)."""
    from Hedging_backtest import SimParams, simulate_heston_qe
    sp = SimParams(kappa=hp.kappa, theta=hp.theta, xi=hp.xi, rho=hp.rho,
                   v0=hp.v0, r=r, q=q)
    times, S, v = simulate_heston_qe(sp, S0, T_prime, n_steps, n_paths, seed,
                                     psi_c)
    rng = np.random.default_rng([seed, _STREAM_SPOTCHECK])
    ip = rng.integers(0, n_paths, n_states)
    it = rng.integers(0, n_steps, n_states)
    return S[ip, it], v[ip, it], tau0 - times[it]


def pathwise_oracle_spotcheck(provider: HestonCFProvider, states, r: float,
                              q: float, tol_rel: float = 1e-3,
                              K: float = 100.0, low_v_band: float = 1e-3,
                              min_agree: float = 0.99,
                              wing_floor_frac: float = 1e-2,
                              n_worst: int = 10) -> SpotCheckResult:
    """Pathwise trust check of `provider` against the independent COS leg.

    `states` = (S, v, tau) equal-length arrays (e.g. from sample_qe_states);
    `K` is the hedged strike. At every state the provider's price/delta/gamma
    are compared to _cos_fd_reference evaluated at the SAME floored variance
    max(v, provider.v_floor) — the check certifies the per-path CF evaluation,
    not the flooring policy. A state agrees when all three quantities match
    within tol_rel on the oracle comparison scale max(|ref|, wing_floor_frac *
    max|ref|) (relative on the body, absolute in the wings — same convention
    as oracle.cross_validate).

    Gate: raises ValueError if the fraction agreeing AWAY from the v <
    low_v_band band falls below `min_agree`; the low-v band (where the CF
    quadrature is most stressed and v_floor bites) is REPORTED in the result,
    never gated. Returns fractions per band plus the worst offenders."""
    S, v, tau = (np.asarray(a, float).ravel() for a in states)
    if not (S.size == v.size == tau.size):
        raise ValueError("states arrays (S, v, tau) must have equal length")
    names = ("price", "delta", "gamma")
    prov = {nm: np.empty(S.size) for nm in names}
    for t in np.unique(tau):
        m = tau == t
        got = provider.evaluate(S[m], v[m], float(t), K)
        for nm in names:
            prov[nm][m] = np.asarray(got[nm], float)
    vf = np.maximum(v, provider.v_floor)
    ref = {nm: np.empty(S.size) for nm in names}
    for i in range(S.size):
        (ref["price"][i], ref["delta"][i], ref["gamma"][i]) = _cos_fd_reference(
            float(S[i]), float(vf[i]), float(tau[i]), K, provider.params, r, q)
    rel = {}
    rel_max = np.zeros(S.size)
    for nm in names:
        scale = np.maximum(np.abs(ref[nm]),
                           wing_floor_frac * np.max(np.abs(ref[nm])))
        rel[nm] = np.abs(prov[nm] - ref[nm]) / scale
        rel_max = np.maximum(rel_max, rel[nm])
    ok = rel_max <= tol_rel
    away = v >= low_v_band
    n_away, n_low = int(away.sum()), int((~away).sum())
    frac = float(ok[away].mean()) if n_away else float("nan")
    frac_low = float(ok[~away].mean()) if n_low else float("nan")
    order = np.argsort(-rel_max)[:n_worst]
    worst = [{"index": int(i), "S": float(S[i]), "v": float(v[i]),
              "tau": float(tau[i]),
              "greek": max(names, key=lambda nm: rel[nm][i]),
              "rel_err": float(rel_max[i]),
              "in_low_v_band": bool(v[i] < low_v_band)} for i in order]
    note = (f"away band (v >= {low_v_band:g}): {int(ok[away].sum())}/{n_away} "
            f"within tol_rel={tol_rel:g}; low-v band: {n_low} states"
            + (f", frac_ok={frac_low:.4f} (reported, not gated)"
               if n_low else " (none sampled)"))
    if n_away and frac < min_agree:
        raise ValueError(
            f"pathwise oracle spot check FAILED: {frac:.4f} < {min_agree} of "
            f"away-band states agree with the independent COS leg within "
            f"tol_rel={tol_rel:g}. Worst offenders: {worst}")
    return SpotCheckResult(frac, frac_low, n_away, n_low, worst, note)
