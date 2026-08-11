"""Heston Greek oracle: ONE oracle cross-validated by independent legs.

Leg A `heston_greeks_cf`: analytic derivatives of the trap-free CF
    (Albrecher et al. 2007, phi_2 form), differentiated before integration.
Leg B `heston_greeks_fd`: independent COS pricer + high-order finite differences.
Leg C `heston_greeks_mc`: Monte-Carlo pathwise / likelihood-ratio.
Leg D `heston_greeks_adi`: Craig-Sneyd ADI PDE, near_feller regime only.

Symbol map vs Albrecher et al. 2007: paper lambda == config xi (vol-of-vol),
paper eta == config theta (long-run variance). NumPy/SciPy only; no PyTorch.

vanna (d2V/dS dv0) is cross-validated by CF/FD (+ ADI where the fourth leg is
required); the MC leg carries no vanna estimator and returns NaN for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

GREEK_NAMES = ("price", "delta", "gamma", "vega", "vanna", "theta", "d_xi", "d_rho")


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters; vega everywhere in this module = dV/dv0
    (sigma-vega = 2*sqrt(v0)*dV/dv0 by the sqrt(v) chain rule)."""
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    @property
    def feller_ratio(self) -> float:
        return 2.0 * self.kappa * self.theta / self.xi ** 2


@dataclass
class GreekSet:
    """One leg's output on a common-shaped grid; NaN where the leg has no estimator."""
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    vanna: np.ndarray
    theta: np.ndarray
    d_xi: np.ndarray
    d_rho: np.ndarray

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in GREEK_NAMES}


def bs_call(S, K, tau, sigma, r: float, q: float) -> dict:
    """Black-Scholes call: reference for the xi->0 trust gate and MC control variate.
    vega_sigma = dV/dsigma; vanna_sigma = d2V/dSdsigma; theta = -dV/dtau."""
    S, K, tau, sigma = map(np.asarray, (S, K, tau, sigma))
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q) * tau) / sq + 0.5 * sq
    d2 = d1 - sq
    dfq, dfr = np.exp(-q * tau), np.exp(-r * tau)
    n1, N1, N2 = norm.pdf(d1), norm.cdf(d1), norm.cdf(d2)
    return {
        "price": S * dfq * N1 - K * dfr * N2,
        "delta": dfq * N1,
        "gamma": dfq * n1 / (S * sq),
        "vega_sigma": S * dfq * n1 * np.sqrt(tau),
        "vanna_sigma": -dfq * n1 * d2 / sigma,
        "theta": -(S * dfq * n1 * sigma / (2.0 * np.sqrt(tau))
                   - q * S * dfq * N1 + r * K * dfr * N2),
    }


def bs_limit_reference(S, K, tau, p: HestonParams, r: float, q: float) -> dict:
    """xi->0 Heston limit: v(t) = theta + (v0-theta)e^{-kappa t} is deterministic, so
    V = BS(sigma_eff) with sigma_eff^2 = IV/tau, IV = integral of v over [0, tau].
    Chain rules: dV/dv0 = vega_sigma * (1-e^{-kappa tau})/(2 sigma_eff kappa tau);
    theta picks up -vega_sigma * d(sigma_eff)/d(tau).

    Vanna (d2V/dSdv0): sigma_eff is S-independent, so
        d2V/dSdv0 = d/dv0 [BS delta at sigma_eff]
                  = (dDelta_BS/dsigma) * (dsigma_eff/dv0) = vanna_sigma * dsig_dv0.
    dDelta_BS/dsigma = e^{-q tau} n(d1) dd1/dsigma with, m = ln(S/K) + (r-q) tau,
        dd1/dsigma = -m/(sigma^2 sqrt(tau)) + sqrt(tau)/2 = -d2/sigma
    (substitute d2 = d1 - sigma sqrt(tau) = m/(sigma sqrt(tau)) - sigma sqrt(tau)/2),
    hence vanna_sigma = -e^{-q tau} n(d1) d2 / sigma (computed in bs_call), and
        dsig_dv0 = dsigma_eff/dv0 = (1-e^{-kappa tau})/(2 sigma_eff kappa tau),
    the SAME factor already used for vega_v0 (from 2 sigma_eff dsigma_eff/dv0 =
    d(IV/tau)/dv0 = (1-e^{-kappa tau})/(kappa tau))."""
    #if volofvol -> 0, heston collapses to black scholes
    k, th, v0 = p.kappa, p.theta, p.v0
    tau = np.asarray(tau, dtype=float)
    IV = th * tau + (v0 - th) * (1.0 - np.exp(-k * tau)) / k
    sig = np.sqrt(IV / tau)
    ref = bs_call(S, K, tau, sig, r, q)
    v_tau = th + (v0 - th) * np.exp(-k * tau)
    ref["theta"] = ref["theta"] - ref["vega_sigma"] * (v_tau * tau - IV) / (2.0 * tau ** 2 * sig)
    dsig_dv0 = (1.0 - np.exp(-k * tau)) / (2.0 * sig * k * tau)
    ref["vega_v0"] = ref["vega_sigma"] * dsig_dv0
    ref["vanna_v0"] = ref["vanna_sigma"] * dsig_dv0
    return ref

# ----------------------------------------------------------------------------
# Leg A: analytic differentiation of the trap-free CF (Albrecher 2007, phi_2)
# ----------------------------------------------------------------------------

def _gl_grid(u_max: float, n_panel: int, n_node: int) -> tuple[np.ndarray, np.ndarray]:
    """Panelled Gauss-Legendre nodes/weights on (0, u_max]."""
    x, w = leggauss(n_node)
    edges = np.linspace(0.0, u_max, n_panel + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    u = ((hi - lo) * (x[None, :] + 1.0) * 0.5 + lo).ravel()
    wts = ((hi - lo) * 0.5 * w[None, :]).ravel()
    return u, wts


def _cf_blocks(w: np.ndarray, tau: float, p: HestonParams) -> dict:
    """Trap-free log-CF core: ln phi(w) = i w (ln S + (r-q) tau) + c(w), with analytic
    derivatives of c wrt v0, tau, xi, rho (evaluate at w=u for P2, w=u-i for P1)."""
    k, th, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    beta = k - 1j * rho * xi * w
    s = w * w + 1j * w
    d = np.sqrt(beta * beta + xi * xi * s)
    g = (beta - d) / (beta + d)
    E = np.exp(-d * tau)
    M = 1.0 - g * E
    # Branch handling: with the phi_2 (minus-d) convention, (1 - g E)/(1 - g) never
    # crosses the negative real axis for any admissible parameters (Albrecher et al.
    # 2007, Thm 3), so numpy's PRINCIPAL sqrt/log ARE the correct branches -- no
    # rotation counting. Complex-step differentiation would NOT rescue the phi_1
    # form: the CF is already complex-valued (there is no free imaginary direction
    # left to perturb in), and complex-step requires analyticity, which is exactly
    # what fails across the log's branch cut where phi_1 misbehaves.
    lnG = np.log(M / (1.0 - g))
    bd = beta - d
    Psi = bd * tau - 2.0 * lnG
    Phi = bd * (1.0 - E) / (xi * xi * M)
    c = (k * th / xi ** 2) * Psi + v0 * Phi

    Psi_t = bd - 2.0 * g * d * E / M
    Phi_t = bd * d * E * (1.0 - g) / (xi * xi * M * M)
    c_tau = (k * th / xi ** 2) * Psi_t + v0 * Phi_t

    N = bd * (1.0 - E)

    def param_deriv(beta_p, d2_half_p):
        """Psi_p, Phi_p for a parameter with d(beta)/dp = beta_p and
        d(d^2)/dp = 2*(beta*beta_p + d2_half_p); explicit xi factors handled by caller."""
        d_p = (beta * beta_p + d2_half_p) / d
        g_p = 2.0 * (beta_p * d - beta * d_p) / (beta + d) ** 2
        E_p = -tau * d_p * E
        M_p = -(g_p * E + g * E_p)
        Psi_p = (beta_p - d_p) * tau - 2.0 * (M_p / M + g_p / (1.0 - g))
        N_p = (beta_p - d_p) * (1.0 - E) - bd * E_p
        Phi_p = (N_p * M - N * M_p) / (xi * xi * M * M)
        return Psi_p, Phi_p

    Psi_r, Phi_r = param_deriv(-1j * xi * w, 0.0)
    c_rho = (k * th / xi ** 2) * Psi_r + v0 * Phi_r
    Psi_x, Phi_x = param_deriv(-1j * rho * w, xi * s)
    c_xi = k * th * (Psi_x / xi ** 2 - 2.0 * Psi / xi ** 3) + v0 * (Phi_x - 2.0 * Phi / xi)
    # every c_* above is AFFINE in v0; the c_v0_* keys are their exact v0
    # coefficients, so a per-point variance v0 + dv shifts them by dv * c_v0_*
    # (used by _cf_greeks_slice's dv0 path; additive, nothing else changes)
    return {"c": c, "c_v0": Phi, "c_tau": c_tau, "c_xi": c_xi, "c_rho": c_rho,
            "c_v0_tau": Phi_t, "c_v0_xi": Phi_x - 2.0 * Phi / xi,
            "c_v0_rho": Phi_r}


def _cf_greeks_slice(S, K, tau: float, p: HestonParams, r: float, q: float,
                     u: np.ndarray, gw: np.ndarray,
                     dv0: np.ndarray | None = None) -> dict:
    """All Greeks at one maturity via P1/P2 with integrands differentiated analytically.
    C = S e^{-q tau} P1 - K e^{-r tau} P2; spot enters only via e^{i u y}, so
    d/dlnS multiplies integrands by iu (Delta, Gamma exact, no bumping).

    Optional `dv0` (shape [n], one entry per point) evaluates point i at
    initial variance p.v0 + dv0[i]: v0 enters the log-CF affinely
    (c = c_base + dv0 * dc/dv0 exactly, both already returned by _cf_blocks —
    the identity bates_price_cf exploits for the terminal mark), so the CF
    blocks are still built ONCE per (tau, u-grid) and per-point v0 costs one
    broadcast exp. Every Greek integrand (delta/gamma/vega/vanna/theta/...)
    reuses the same per-point W1/W2. With dv0=None the scalar-v0 code path is
    unchanged (bit-identical to the pre-refactor function)."""
    b2 = _cf_blocks(u + 0j, tau, p)
    b1 = _cf_blocks(u - 1j, tau, p)
    iu = 1j * u
    if dv0 is None:
        W1, W2 = np.exp(b1["c"]) / iu, np.exp(b2["c"]) / iu
        ct1, ct2 = b1["c_tau"], b2["c_tau"]
        cx1, cx2 = b1["c_xi"], b2["c_xi"]
        cr1, cr2 = b1["c_rho"], b2["c_rho"]
    else:
        dv = np.asarray(dv0, float)[:, None]
        W1 = np.exp(b1["c"][None, :] + dv * b1["c_v0"][None, :]) / iu
        W2 = np.exp(b2["c"][None, :] + dv * b2["c_v0"][None, :]) / iu
        # c_tau/c_xi/c_rho are affine in v0 as well — shift by their exact
        # v0 coefficients so theta/d_xi/d_rho are per-point exact, not frozen
        # at p.v0
        ct1 = b1["c_tau"][None, :] + dv * b1["c_v0_tau"][None, :]
        ct2 = b2["c_tau"][None, :] + dv * b2["c_v0_tau"][None, :]
        cx1 = b1["c_xi"][None, :] + dv * b1["c_v0_xi"][None, :]
        cx2 = b2["c_xi"][None, :] + dv * b2["c_v0_xi"][None, :]
        cr1 = b1["c_rho"][None, :] + dv * b1["c_v0_rho"][None, :]
        cr2 = b2["c_rho"][None, :] + dv * b2["c_v0_rho"][None, :]
    y = np.log(S / K) + (r - q) * tau
    ker = np.exp(1j * y[:, None] * u[None, :]) * gw[None, :]

    def J(wgt):
        if wgt.ndim == 2:                     # per-point weights (dv0 path)
            return (ker * wgt).sum(axis=1).real / np.pi
        return (ker @ wgt).real / np.pi

    P1, P2 = 0.5 + J(W1), 0.5 + J(W2)
    A1, A2 = J(iu * W1), J(iu * W2)
    B1, B2 = J(iu * iu * W1), J(iu * iu * W2)
    V1, V2 = b1["c_v0"] * W1, b2["c_v0"] * W2       # vega integrands (analytic d/dv0)
    dfq, dfr = np.exp(-q * tau), np.exp(-r * tau)
    T1 = J((iu * (r - q) + ct1) * W1)
    T2 = J((iu * (r - q) + ct2) * W2)
    return {
        "price": S * dfq * P1 - K * dfr * P2,
        "delta": dfq * P1 + dfq * A1 - (K / S) * dfr * A2,
        "gamma": (dfq * (A1 + B1) + (K / S) * dfr * (A2 - B2)) / S,
        "vega": S * dfq * J(V1) - K * dfr * J(V2),
        # vanna = d(vega)/dS: the delta product/chain rule applied to the vega
        # integrand (spot enters only via e^{iuy}, so d/dlnS multiplies by iu,
        # then /S); no new differentiation machinery.
        "vanna": dfq * (J(V1) + J(iu * V1)) - (K / S) * dfr * J(iu * V2),
        "theta": -(-q * S * dfq * P1 + S * dfq * T1 + r * K * dfr * P2 - K * dfr * T2),
        "d_xi": S * dfq * J(cx1 * W1) - K * dfr * J(cx2 * W2),
        "d_rho": S * dfq * J(cr1 * W1) - K * dfr * J(cr2 * W2),
    }


def heston_greeks_cf(S, K, tau, p: HestonParams, r: float, q: float,
                     n_panel: int = 12, n_node: int = 64) -> GreekSet:
    """Leg A: price and Greeks from the trap-free phi_2 CF, all derivatives taken
    analytically on the integrand before Gauss-Legendre integration."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    S, K, tau = S.ravel(), K.ravel(), tau.ravel()
    out = {name: np.empty(S.size) for name in GREEK_NAMES}
    for t in np.unique(tau):
        m = tau == t
        u_max = max(200.0, 15.0 / np.sqrt(max(min(p.v0, p.theta), 1e-4) * t))
        u, gw = _gl_grid(min(u_max, 3000.0), n_panel, n_node)
        res = _cf_greeks_slice(S[m], K[m], float(t), p, r, q, u, gw)
        for name in GREEK_NAMES:
            out[name][m] = res[name]
    return GreekSet(**{name: out[name].reshape(shape) for name in GREEK_NAMES})


def heston_greeks_cf_v0(S, K, tau, p: HestonParams, r: float, q: float,
                        v0=None, n_panel: int = 12, n_node: int = 64) -> GreekSet:
    """Leg-A CF Greeks with PER-POINT initial variance (ADDITIVE companion to
    heston_greeks_cf, which is unchanged). Optional `v0` (broadcast against S)
    evaluates each point at its own initial variance via the affine-v0
    identity in _cf_greeks_slice's dv0 path: the CF blocks and every Greek
    integrand are built once per (tau, u-grid); only exp(c) is per-point.
    With constant v0 (== p.v0 or not) this reproduces heston_greeks_cf at the
    corresponding parameters up to floating-point summation order (unit-tested
    at rtol 1e-10 in test_providers.py). The u-grid extent follows the same
    rule as bates_price_cf: with per-point v0 the smallest variance sets the
    slowest CF decay, so it drives u_max. Consumed by providers.HestonCFProvider
    to evaluate exact theta_train Greeks at pathwise (S, v_t, tau) hedge states."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    v0a = (np.full(S.size, p.v0) if v0 is None
           else np.broadcast_to(np.asarray(v0, float), shape).ravel().copy())
    S, K, tau = S.ravel(), K.ravel(), tau.ravel()
    out = {name: np.empty(S.size) for name in GREEK_NAMES}
    for t in np.unique(tau):
        m = tau == t
        vref = max(min(float(v0a[m].min()), p.theta), 1e-4)
        u_max = max(200.0, 15.0 / np.sqrt(vref * float(t)))
        u, gw = _gl_grid(min(u_max, 3000.0), n_panel, n_node)
        res = _cf_greeks_slice(S[m], K[m], float(t), p, r, q, u, gw,
                               dv0=v0a[m] - p.v0)
        for name in GREEK_NAMES:
            out[name][m] = res[name]
    return GreekSet(**{name: out[name].reshape(shape) for name in GREEK_NAMES})


# ----------------------------------------------------------------------------
# True-DGP terminal-mark pricers (PRICE ONLY, no Greeks): Bates CF and Merton
# series. Used by the hedging engine to mark the liability at the horizon T';
# they are NOT oracle legs and are NOT cross-validated — each is validated by
# its own unit tests (lambda_j=0 Heston recovery, Merton limit, MC check).
# ----------------------------------------------------------------------------

def _bates_jump_cexp(w: np.ndarray, tau: float, lambda_j: float, mu_j: float,
                     sigma_j: float) -> np.ndarray:
    """Log-CF exponent of compensated lognormal jumps (the Bates factor on top
    of the Heston CF): tau*lambda_j*(E[e^{i w J}] - 1) - i w tau lambda_j *
    (E[e^J] - 1) with J ~ N(mu_j, sigma_j^2). Vanishes identically at w = -i
    (compensation = martingale), so the P1/P2 normalisation phi(-i) is
    unchanged and the jump exponent simply adds to c(w) in both legs."""
    ejump = np.exp(1j * w * mu_j - 0.5 * w * w * sigma_j ** 2)
    comp = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    return tau * lambda_j * (ejump - 1.0) - 1j * w * tau * lambda_j * comp


def bates_price_cf(S, K, tau, p: HestonParams, r: float, q: float,
                   lambda_j: float = 0.0, mu_j: float = 0.0,
                   sigma_j: float = 0.0, v0=None,
                   n_panel: int = 12, n_node: int = 64) -> np.ndarray:
    """Bates call PRICE: trap-free phi_2 CF blocks times the lognormal-jump CF
    factor. The jump block is skipped entirely at lambda_j = 0, so lambda_j = 0
    reproduces the pure-Heston price integral (heston_greeks_cf's price) on the
    same quadrature. Optional `v0` (broadcast against S) prices each point at
    its own initial variance: v0 enters the log-CF affinely (c = c_base +
    (v0 - p.v0) * dc/dv0 exactly, both already returned by _cf_blocks), so
    per-point v0 vectorizes without re-evaluating the CF blocks."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    v0a = (np.full(S.size, p.v0) if v0 is None
           else np.broadcast_to(np.asarray(v0, float), shape).ravel().copy())
    S, K, tau = S.ravel(), K.ravel(), tau.ravel()
    out = np.empty(S.size)
    for t in np.unique(tau):
        m = tau == t
        t_ = float(t)
        # same u_max rule as heston_greeks_cf; with per-point v0 the smallest
        # variance sets the slowest CF decay, so it drives the grid extent
        vref = max(min(float(v0a[m].min()), p.theta), 1e-4)
        u_max = max(200.0, 15.0 / np.sqrt(vref * t_))
        u, gw = _gl_grid(min(u_max, 3000.0), n_panel, n_node)
        iu = 1j * u
        b1, b2 = _cf_blocks(u - 1j, t_, p), _cf_blocks(u + 0j, t_, p)
        j1 = j2 = 0.0
        if lambda_j != 0.0:
            j1 = _bates_jump_cexp(u - 1j, t_, lambda_j, mu_j, sigma_j)
            j2 = _bates_jump_cexp(u + 0j, t_, lambda_j, mu_j, sigma_j)
        dv = (v0a[m] - p.v0)[:, None]
        W1 = np.exp(b1["c"][None, :] + dv * b1["c_v0"][None, :] + j1) / iu
        W2 = np.exp(b2["c"][None, :] + dv * b2["c_v0"][None, :] + j2) / iu
        y = np.log(S[m] / K[m]) + (r - q) * t_
        ker = np.exp(1j * y[:, None] * u[None, :]) * gw[None, :]
        P1 = 0.5 + (ker * W1).sum(axis=1).real / np.pi
        P2 = 0.5 + (ker * W2).sum(axis=1).real / np.pi
        out[m] = S[m] * np.exp(-q * t_) * P1 - K[m] * np.exp(-r * t_) * P2
    return out.reshape(shape)


def merton_price(S, K, tau, sigma: float, lambda_j: float, mu_j: float,
                 sigma_j: float, r: float, q: float,
                 n_terms: int = 60) -> np.ndarray:
    """Closed-form Merton (1976) jump-diffusion call: Poisson mixture of
    Black-Scholes prices, sum_n w_n * BS(sigma_n, r_n) with
        lam' = lambda_j * e^{mu_j + sigma_j^2/2},
        w_n = e^{-lam' tau} (lam' tau)^n / n!,
        sigma_n^2 = sigma^2 + n sigma_j^2 / tau,
        r_n = r - lambda_j*(e^{mu_j + sigma_j^2/2} - 1) + n*(mu_j + sigma_j^2/2)/tau.
    Truncation error after n_terms is the Poisson tail mass — negligible for
    the lam'*tau values in scope (< 1)."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    if lambda_j == 0.0:
        return bs_call(S, K, tau, sigma, r, q)["price"]
    k_m = np.exp(mu_j + 0.5 * sigma_j ** 2)
    comp = lambda_j * (k_m - 1.0)
    lam_p = lambda_j * k_m
    w = np.exp(-lam_p * tau)
    total = np.zeros(S.shape)
    for n in range(n_terms):
        if n > 0:
            w = w * (lam_p * tau) / n
        sig_n = np.sqrt(sigma ** 2 + n * sigma_j ** 2 / tau)
        r_n = r - comp + n * (mu_j + 0.5 * sigma_j ** 2) / tau
        total = total + w * bs_call(S, K, tau, sig_n, r_n, q)["price"]
    return total


# ----------------------------------------------------------------------------
# Leg B: independent COS pricer + 4th-order FD with Richardson convergence check
# ----------------------------------------------------------------------------

def _cos_price(S, K, tau: float, kappa, theta, xi, rho, v0, r, q,
               n_terms: int = 512, L: float = 14.0) -> np.ndarray:
    """Fang-Oosterlee COS call price at one maturity. CF re-implemented here from
    the phi_2 formula independently of leg A (no shared helpers): leg-A coding
    errors cannot propagate into this leg."""
    S = np.atleast_1d(np.asarray(S, float))
    K = np.atleast_1d(np.asarray(K, float))
    x = np.log(S / K) + (r - q) * tau
    ivar = theta * tau + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / kappa
    half = L * np.sqrt(ivar + xi * np.sqrt(ivar * tau))
    a = x - 0.5 * ivar - half
    b = x - 0.5 * ivar + half
    om = np.arange(1, n_terms) * np.pi / (2.0 * half)          # shared: b - a = 2*half
    zeta = kappa - 1j * rho * xi * om
    D = np.sqrt(zeta * zeta + (om * om + 1j * om) * xi * xi)
    G = (zeta - D) / (zeta + D)
    ed = np.exp(-D * tau)
    phi = np.exp((kappa * theta / xi ** 2)
                 * ((zeta - D) * tau - 2.0 * np.log((1.0 - G * ed) / (1.0 - G)))
                 + (v0 / xi ** 2) * (zeta - D) * (1.0 - ed) / (1.0 - G * ed))
    T = (phi * np.exp(1j * om * (0.5 * ivar + half))).real      # x - a is constant
    # payoff coefficients on [c, b], c = max(a, 0) (call kink at y = 0)
    c = np.maximum(a, 0.0)
    dead = c >= b
    wb, wc = om[None, :] * (b - a)[:, None], om[None, :] * (c - a)[:, None]
    eb, ec = np.exp(b)[:, None], np.exp(c)[:, None]
    chi = (np.cos(wb) * eb - np.cos(wc) * ec
           + om * (np.sin(wb) * eb - np.sin(wc) * ec)) / (1.0 + om * om)
    psi = (np.sin(wb) - np.sin(wc)) / om
    ex = np.exp(-r * tau) / (2.0 * half)
    series = 0.5 * ((np.exp(b) - np.exp(c)) - (b - c))          # k = 0 term (phi(0)=1)
    price = K * ex * 2.0 * (series + ((chi - psi) @ T))
    return np.where(dead, 0.0, price)


def _fd4(fm2, fm1, f0, fp1, fp2, h):
    """4th-order central first/second derivatives from a 5-point stencil."""
    d1 = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * h)
    d2 = (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * h * h)
    return d1, d2


def heston_greeks_fd(S, K, tau, p: HestonParams, r: float, q: float,
                     rel_h: float = 2e-2, n_terms: int = 512, L: float = 14.0,
                     conv_rtol: float = 5e-4) -> GreekSet:
    """Leg B: COS prices bumped with 4th-order central stencils; step-size sweep
    h, h/2, h/4 with Richardson extrapolation; points failing the convergence
    check are returned NaN (visible to the disagreement mask, never patched).
    vanna = cross stencil D1_S o D1_v0 (both 4th order), same sweep."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    S, K, tau = S.ravel(), K.ravel(), tau.ravel()
    out = {name: np.empty(S.size) for name in GREEK_NAMES}
    for t in np.unique(tau):
        m = tau == t
        Sm, Km, t_ = S[m], K[m], float(t)

        def price(Sv=None, v0=None, tv=None, xiv=None, rhov=None):
            return _cos_price(Sm if Sv is None else Sv, Km, t_ if tv is None else tv,
                              p.kappa, p.theta, p.xi if xiv is None else xiv,
                              p.rho if rhov is None else rhov,
                              p.v0 if v0 is None else v0, r, q, n_terms, L)

        def estimates(scale: float) -> dict:
            hS = rel_h * scale * Sm
            hv = rel_h * scale * max(p.v0, 0.02)
            ht = rel_h * scale * t_
            hx = rel_h * scale * p.xi
            hr = rel_h * scale * min(max(abs(p.rho), 0.1), (1.0 - abs(p.rho)) / 2.5)
            f0 = price()
            delta, gamma = _fd4(price(Sv=Sm - 2 * hS), price(Sv=Sm - hS), f0,
                                price(Sv=Sm + hS), price(Sv=Sm + 2 * hS), hS)
            vega, _ = _fd4(price(v0=p.v0 - 2 * hv), price(v0=p.v0 - hv), f0,
                           price(v0=p.v0 + hv), price(v0=p.v0 + 2 * hv), hv)
            dtau, _ = _fd4(price(tv=t_ - 2 * ht), price(tv=t_ - ht), f0,
                           price(tv=t_ + ht), price(tv=t_ + 2 * ht), ht)
            dxi, _ = _fd4(price(xiv=p.xi - 2 * hx), price(xiv=p.xi - hx), f0,
                          price(xiv=p.xi + hx), price(xiv=p.xi + 2 * hx), hx)
            drho, _ = _fd4(price(rhov=p.rho - 2 * hr), price(rhov=p.rho - hr), f0,
                           price(rhov=p.rho + hr), price(rhov=p.rho + 2 * hr), hr)

            # cross stencil runs at half the base step: at short maturity the
            # full hS spans the entire ATM vanna sign-swing and the (h, h/2)
            # Richardson pair cannot certify, while (h/2, h/4) certifies with an
            # order-of-magnitude margin. Step size is a leg-internal
            # discretization choice; sweep pattern and conv_rtol are unchanged.
            hSv, hvv = 0.5 * hS, 0.5 * hv

            def d1v(Sv):
                """4th-order D1_v0 of the COS price at bumped spot Sv (the f0
                argument is unused by the first-derivative stencil)."""
                return _fd4(price(Sv=Sv, v0=p.v0 - 2 * hvv), price(Sv=Sv, v0=p.v0 - hvv),
                            0.0, price(Sv=Sv, v0=p.v0 + hvv),
                            price(Sv=Sv, v0=p.v0 + 2 * hvv), hvv)[0]

            vanna = _fd4(d1v(Sm - 2 * hSv), d1v(Sm - hSv), 0.0,
                         d1v(Sm + hSv), d1v(Sm + 2 * hSv), hSv)[0]
            return {"price": f0, "delta": delta, "gamma": gamma, "vega": vega,
                    "vanna": vanna, "theta": -dtau, "d_xi": dxi, "d_rho": drho}

        e1, e2, e4 = estimates(1.0), estimates(0.5), estimates(0.25)
        for name in GREEK_NAMES:
            if name == "price":
                out[name][m] = e4[name]
                continue
            r1 = (16.0 * e2[name] - e1[name]) / 15.0
            r2 = (16.0 * e4[name] - e2[name]) / 15.0
            scale_ref = np.maximum(np.abs(r2), np.median(np.abs(r2)) + 1e-300)
            ok = np.abs(r2 - r1) <= conv_rtol * scale_ref
            out[name][m] = np.where(ok, r2, np.nan)
    return GreekSet(**{name: out[name].reshape(shape) for name in GREEK_NAMES})


# ----------------------------------------------------------------------------
# Leg C: Monte-Carlo pathwise / likelihood-ratio, antithetic + control variates
# ----------------------------------------------------------------------------

def heston_greeks_mc(S, K, tau, p: HestonParams, r: float, q: float,
                     n_paths: int = 200_000, steps_per_year: int = 250,
                     seed: int = 42, chunk_pairs: int = 5_000,
                     vfloor: float = 1e-4) -> tuple[GreekSet, GreekSet]:
    """Leg C: full-truncation Euler simulation of (S, v); no CF anywhere.
    Estimators: pathwise Delta, mixed pathwise-likelihood-ratio Gamma (LR weight
    from the first Euler step), tangent-process Vega (dV/dv0). Antithetic pairs;
    GBM/Black-Scholes control variate on price and Delta. theta/d_xi/d_rho/vanna
    are NaN (no estimator in this leg; pathwise d/dv0 of the Delta estimator
    hits the indicator's kink, so a Dirac term survives and the naive mixed
    estimator is biased). Returns (values, standard_errors); output is
    bit-identical for a fixed seed. `vfloor` floors sqrt(v) only inside tangent
    denominators (bias O(vfloor), prevents 1/sqrt blow-up near v=0)."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    Sf, Kf, tf = S.ravel(), K.ravel(), tau.ravel()
    taus = np.unique(tf)
    # time grid hitting every snapshot exactly (no snapshot-alignment bias)
    times = [0.0]
    snap_step = {}
    for tj in taus:
        nseg = max(1, int(np.ceil((tj - times[-1]) * steps_per_year)))
        seg = np.linspace(times[-1], tj, nseg + 1)[1:]
        times.extend(seg.tolist())
        snap_step[float(tj)] = len(times) - 1
    dts = np.diff(np.asarray(times))
    sig_cv = np.sqrt(0.5 * (p.v0 + p.theta))          # constant-vol GBM control
    rho_c = np.sqrt(1.0 - p.rho ** 2)
    n_pairs = n_paths // 2
    stats: dict[float, dict[str, np.ndarray]] = {}
    for tj in taus:
        npts = int((tf == tj).sum())
        stats[float(tj)] = {k: np.zeros(npts) for k in
                            ("pY", "pY2", "pX", "pX2", "pXY",
                             "dY", "dY2", "dX", "dX2", "dXY",
                             "gY", "gY2", "vY", "vY2")}
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_pairs:
        npair = min(chunk_pairs, n_pairs - done)
        done += npair
        n2 = 2 * npair
        v = np.full(n2, p.v0)
        lnM = np.zeros(n2)
        lnMg = np.zeros(n2)
        Yv = np.ones(n2)
        Lt = np.zeros(n2)
        z1_first = z2_first = None
        sig1 = np.sqrt(p.v0 * dts[0])
        snaps = {}
        for i, dt in enumerate(dts):
            Z = rng.standard_normal((npair, 2))
            z1 = np.concatenate([Z[:, 0], -Z[:, 0]])
            z2 = np.concatenate([Z[:, 1], -Z[:, 1]])
            w2 = p.rho * z1 + rho_c * z2
            if i == 0:
                z1_first, z2_first = z1.copy(), z2.copy()
            vp = np.maximum(v, 0.0)
            sq = np.sqrt(vp)
            chi = (v > 0.0).astype(float)
            sqf = np.maximum(sq, vfloor)
            rt = np.sqrt(dt)
            Lt = Lt + (-0.5 * dt + rt * z1 / (2.0 * sqf)) * Yv * chi
            Yv = Yv * (1.0 - p.kappa * dt * chi + p.xi * rt * w2 * chi / (2.0 * sqf))
            lnM = lnM + (r - q - 0.5 * vp) * dt + sq * rt * z1
            lnMg = lnMg + (r - q - 0.5 * sig_cv ** 2) * dt + sig_cv * rt * z1
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * sq * rt * w2
            step = i + 1
            for tj, sj in snap_step.items():
                if sj == step:
                    snaps[tj] = (np.exp(lnM), np.exp(lnMg), Lt.copy())
        # LR score for a mean shift of lnS_1 with v_1 held: the first-step spot and
        # variance shocks are rho-correlated, so the score is the residual
        # (z1 - (rho/sqrt(1-rho^2)) z2)/sigma_1, not z1/sigma_1.
        lrw = (z1_first - (p.rho / rho_c) * z2_first) / sig1 - 1.0
        for tj in taus:
            tj = float(tj)
            m = tf == tj
            S0, K0 = Sf[m], Kf[m]
            M, Mg, L = snaps[tj]
            disc = np.exp(-r * tj)
            acc = stats[tj]
            blk = max(1, int(2e6 / n2))
            for lo in range(0, S0.size, blk):
                sl = slice(lo, lo + blk)
                S0b, K0b = S0[sl][:, None], K0[sl][:, None]
                ST, STg = S0b * M[None, :], Mg[None, :] * S0b
                ind = ST > K0b

                def pair(a):
                    return 0.5 * (a[:, :npair] + a[:, npair:])

                Yp = pair(disc * np.where(ind, ST - K0b, 0.0))
                Xp = pair(disc * np.maximum(STg - K0b, 0.0))
                Yd = pair(disc * ind * M[None, :])
                Xd = pair(disc * (STg > K0b) * Mg[None, :])
                Yg = pair(disc * ind * M[None, :] * lrw[None, :]) / S0[sl][:, None]
                Yv_ = pair(disc * np.where(ind, ST, 0.0) * L[None, :])
                for nm, arr in (("pY", Yp), ("pX", Xp), ("dY", Yd), ("dX", Xd),
                                ("gY", Yg), ("vY", Yv_)):
                    acc[nm][sl] += arr.sum(axis=1)
                    acc[nm + "2"][sl] += (arr * arr).sum(axis=1)
                acc["pXY"][sl] += (Yp * Xp).sum(axis=1)
                acc["dXY"][sl] += (Yd * Xd).sum(axis=1)

    def cv_est(n, sY, sY2, sX, sX2, sXY, EX):
        mY, mX = sY / n, sX / n
        varX = np.maximum(sX2 / n - mX ** 2, 0.0)
        varY = np.maximum(sY2 / n - mY ** 2, 0.0)
        cov = sXY / n - mY * mX
        b = np.where(varX > 0, cov / np.maximum(varX, 1e-300), 0.0)
        est = mY - b * (mX - EX)
        var_e = np.maximum(varY - np.where(varX > 0, cov ** 2 / np.maximum(varX, 1e-300), 0.0), 0.0)
        return est, np.sqrt(var_e / n)

    val = {nm: np.full(Sf.size, np.nan) for nm in GREEK_NAMES}
    se = {nm: np.full(Sf.size, np.nan) for nm in GREEK_NAMES}
    # no MC vanna estimator; CF/FD/ADI cross-validate it
    for tj in taus:
        tj = float(tj)
        m = tf == tj
        a = stats[tj]
        bs = bs_call(Sf[m], Kf[m], tj, sig_cv, r, q)
        val["price"][m], se["price"][m] = cv_est(n_pairs, a["pY"], a["pY2"],
                                                 a["pX"], a["pX2"], a["pXY"], bs["price"])
        val["delta"][m], se["delta"][m] = cv_est(n_pairs, a["dY"], a["dY2"],
                                                 a["dX"], a["dX2"], a["dXY"], bs["delta"])
        for nm, k in (("gamma", "g"), ("vega", "v")):
            mY = a[k + "Y"] / n_pairs
            varY = np.maximum(a[k + "Y2"] / n_pairs - mY ** 2, 0.0)
            val[nm][m], se[nm][m] = mY, np.sqrt(varY / n_pairs)
    return (GreekSet(**{nm: val[nm].reshape(shape) for nm in GREEK_NAMES}),
            GreekSet(**{nm: se[nm].reshape(shape) for nm in GREEK_NAMES}))


# ----------------------------------------------------------------------------
# Leg D: Craig-Sneyd ADI on the Heston PDE (near_feller regime only)
# ----------------------------------------------------------------------------

def _thomas_batch(lo, di, up, rhs):
    """Batched Thomas solve of tridiagonal systems along the last axis."""
    n = di.shape[-1]
    cp = np.empty_like(di)
    x = np.empty_like(rhs)
    cp[..., 0] = up[..., 0] / di[..., 0]
    x[..., 0] = rhs[..., 0] / di[..., 0]
    for i in range(1, n):
        den = di[..., i] - lo[..., i] * cp[..., i - 1]
        cp[..., i] = up[..., i] / den
        x[..., i] = (rhs[..., i] - lo[..., i] * x[..., i - 1]) / den
    for i in range(n - 2, -1, -1):
        x[..., i] -= cp[..., i] * x[..., i + 1]
    return x


def _nonuniform_weights(g: np.ndarray):
    """Interior 3-point first/second derivative weights on a non-uniform grid."""
    h = np.diff(g)
    hm, hp = h[:-1], h[1:]
    w1 = (-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp)))
    w2 = (2.0 / (hm * (hm + hp)), -2.0 / (hm * hp), 2.0 / (hp * (hm + hp)))
    return h, w1, w2


def heston_greeks_adi(S, K, tau, p: HestonParams, r: float, q: float,
                      nx: int = 901, nv: int = 481, xmax: float = 4.0,
                      vmax: float = 1.6, steps_per_year: int = 1000,
                      cx: float = 0.05) -> GreekSet:
    """Leg D (fourth leg): Craig-Sneyd ADI solve of the Heston PDE in
    (x, v), x = S/K (call homogeneity: one solve serves all S, K; tau read off
    as snapshots). Sinh-stretched x grid clustered at the strike and v grid
    dense near v = 0; Rannacher damped half-step start; cell-averaged payoff at
    the kink. Greeks by grid differencing (vanna = D1_v of the delta slice);
    theta from the PDE operator; d_xi/d_rho: NaN.

    vmax=1.6 is LOAD-BEARING, not a comfort margin. At the previous default
    (vmax=0.8) the variance domain is truncated inside the mass of the xi=0.60
    regime, and the artificial boundary contaminates the v-direction derivatives:
    on feller_violating_volvol (Feller 0.444) the vega disagreement with the CF
    leg was SYSTEMATIC (adi > cf on 87.9% of grid points) and grew monotonically
    with tau (0.0 at tau=0.04 to 0.0087 at tau=1.0, the boundary error
    propagating inward), masking 59.4% of that regime's vega grid at the
    contract's agreement_tol_rel=1e-3.

    It is domain truncation, not resolution: refining time (1000 -> 4000
    steps/yr) left the gap identical to 6 decimal places, and refining nv
    (241 -> 481) left vega unchanged while gamma improved. Doubling the DOMAIN at
    an unchanged nv=241 -- i.e. a coarser grid near v=0 -- collapsed the vega
    mask from 0.5942 to 0.0033. Gamma, a pure x-derivative orthogonal to the bad
    boundary, was clean throughout. 1.6 -> 3.2 buys ~nothing (median 4.0e-05 ->
    2.8e-05), so 1.6 is converged; nv=481 matches the setting the frozen label
    artifacts were generated with. Regimes with a tight variance density
    (near_feller, xi=0.34) are unaffected either way -- their mask rates do not
    move under this sweep, which is what localizes the effect to broad-xi
    regimes. Do not lower vmax back to 0.8."""
    S, K, tau = np.broadcast_arrays(np.asarray(S, float), np.asarray(K, float),
                                    np.asarray(tau, float))
    shape = S.shape
    Sf, Kf, tf = S.ravel(), K.ravel(), tau.ravel()
    taus = np.unique(tf)
    eta = np.linspace(np.arcsinh(-1.0 / cx), np.arcsinh((xmax - 1.0) / cx), nx)
    xg = 1.0 + cx * np.sinh(eta)
    xg[0] = 0.0
    c_str = max(p.v0, 0.01)
    vg = c_str * np.sinh(np.linspace(0.0, np.arcsinh(vmax / c_str), nv))
    hx, (u1m, u1c, u1p), (u2m, u2c, u2p) = _nonuniform_weights(xg)
    hv, (w1m, w1c, w1p), (w2m, w2c, w2p) = _nonuniform_weights(vg)

    Xm, Vm = xg[None, :], vg[:, None]          # U has shape (nv, nx)
    # A1: x-direction (1/2 v x^2 D2x + (r-q) x D1x - r/2), zero on boundary rows
    a1l = np.zeros((nv, nx))
    a1c = np.zeros((nv, nx))
    a1u = np.zeros((nv, nx))
    a1l[:, 1:-1] = 0.5 * Vm * Xm[:, 1:-1] ** 2 * u2m + (r - q) * Xm[:, 1:-1] * u1m
    a1c[:, 1:-1] = 0.5 * Vm * Xm[:, 1:-1] ** 2 * u2c + (r - q) * Xm[:, 1:-1] * u1c - 0.5 * r
    a1u[:, 1:-1] = 0.5 * Vm * Xm[:, 1:-1] ** 2 * u2p + (r - q) * Xm[:, 1:-1] * u1p
    for a in (a1l, a1c, a1u):
        a[:, 0] = a[:, -1] = 0.0
        a[-1, :] = 0.0
    # A2: v-direction (1/2 xi^2 v D2v + kappa(theta - v) D1v - r/2)
    a2l = np.zeros((nv, nx))
    a2c = np.zeros((nv, nx))
    a2u = np.zeros((nv, nx))
    drift = p.kappa * (p.theta - vg)
    diff2 = 0.5 * p.xi ** 2 * vg
    a2l[1:-1, :] = (diff2[1:-1] * w2m + drift[1:-1] * w1m)[:, None]
    a2c[1:-1, :] = (diff2[1:-1] * w2c + drift[1:-1] * w1c)[:, None] - 0.5 * r
    a2u[1:-1, :] = (diff2[1:-1] * w2p + drift[1:-1] * w1p)[:, None]
    a2c[0, :] = -p.kappa * p.theta / hv[0] - 0.5 * r      # v=0: forward difference
    a2u[0, :] = p.kappa * p.theta / hv[0]
    for a in (a2l, a2c, a2u):
        a[-1, :] = 0.0                                     # v=vmax Dirichlet
        a[:, 0] = a[:, -1] = 0.0

    def apply_a1(U):
        out = np.zeros_like(U)
        out[:, 1:-1] = (a1l[:, 1:-1] * U[:, :-2] + a1c[:, 1:-1] * U[:, 1:-1]
                        + a1u[:, 1:-1] * U[:, 2:])
        out[-1, :] = 0.0
        return out

    def apply_a2(U):
        out = np.zeros_like(U)
        out[1:-1, 1:-1] = (a2l[1:-1, 1:-1] * U[:-2, 1:-1] + a2c[1:-1, 1:-1] * U[1:-1, 1:-1]
                           + a2u[1:-1, 1:-1] * U[2:, 1:-1])
        out[0, 1:-1] = a2c[0, 1:-1] * U[0, 1:-1] + a2u[0, 1:-1] * U[1, 1:-1]
        return out

    mixc = p.rho * p.xi * Vm[1:-1] * Xm[:, 1:-1]

    def apply_a0(U):
        out = np.zeros_like(U)
        Ux = u1m * U[:, :-2] + u1c * U[:, 1:-1] + u1p * U[:, 2:]   # (nv, nx-2)
        Uxv = (w1m[:, None] * Ux[:-2] + w1c[:, None] * Ux[1:-1] + w1p[:, None] * Ux[2:])
        out[1:-1, 1:-1] = mixc * Uxv
        return out

    def solve_x(theta_s, dt, rhs):
        lo, di, up = -theta_s * dt * a1l, 1.0 - theta_s * dt * a1c, -theta_s * dt * a1u
        return _thomas_batch(lo, di, up, rhs)

    def solve_v(theta_s, dt, rhs):
        lo = (-theta_s * dt * a2l).T.copy()
        di = (1.0 - theta_s * dt * a2c).T.copy()
        up = (-theta_s * dt * a2u).T.copy()
        return _thomas_batch(lo, di, up, rhs.T.copy()).T

    def boundaries(U, t):
        U[:, 0] = 0.0
        U[-1, :] = xg * np.exp(-q * t)
        U[:, -1] = xmax * np.exp(-q * t) - np.exp(-r * t)

    # cell-averaged payoff at the kink node
    U = np.maximum(xg - 1.0, 0.0)
    ik = int(np.argmin(np.abs(xg - 1.0)))
    lo_e, hi_e = xg[ik] - hx[ik - 1] / 2, xg[ik] + hx[ik] / 2
    if lo_e < 1.0 < hi_e:
        U[ik] = 0.5 * (hi_e - 1.0) ** 2 / (0.5 * (hx[ik - 1] + hx[ik]))
    U = np.tile(U, (nv, 1))
    boundaries(U, 0.0)

    times = [0.0]
    snap_at = {}
    for tj in taus:
        nseg = max(2, int(np.ceil((tj - times[-1]) * steps_per_year)))
        if not snap_at:                       # damped start pollutes the first
            nseg = max(nseg, 100)             # segment: resolve it finely
        times.extend(np.linspace(times[-1], tj, nseg + 1)[1:].tolist())
        snap_at[len(times) - 1] = float(tj)
    snaps = {}
    n_rannacher = 4
    for step, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
        dt = t1 - t0
        if step < n_rannacher:                              # damped: 2 implicit half-steps
            for _ in range(2):
                A1U, A2U = apply_a1(U), apply_a2(U)
                Y0 = U + 0.5 * dt * (A1U + A2U + apply_a0(U))
                Y1 = solve_x(1.0, 0.5 * dt, Y0 - 0.5 * dt * A1U)
                U = solve_v(1.0, 0.5 * dt, Y1 - 0.5 * dt * A2U)
            boundaries(U, t1)
            if step + 1 in snap_at:
                snaps[snap_at[step + 1]] = U.copy()
            continue
        A1U, A2U = apply_a1(U), apply_a2(U)                 # Craig-Sneyd, theta = 1/2
        A0U = apply_a0(U)
        Y0 = U + dt * (A1U + A2U + A0U)
        Y1 = solve_x(0.5, dt, Y0 - 0.5 * dt * A1U)
        Y2 = solve_v(0.5, dt, Y1 - 0.5 * dt * A2U)
        Y0h = Y0 + 0.5 * dt * (apply_a0(Y2) - A0U)
        Y1 = solve_x(0.5, dt, Y0h - 0.5 * dt * A1U)
        U = solve_v(0.5, dt, Y1 - 0.5 * dt * A2U)
        boundaries(U, t1)
        if step + 1 in snap_at:
            snaps[snap_at[step + 1]] = U.copy()

    def v_interp(surf):
        """Linear interpolation of a (nv, nx) surface to v = v0 -> (nx,) row."""
        j = int(np.searchsorted(vg, p.v0)) - 1
        j = min(max(j, 0), nv - 2)
        w = (p.v0 - vg[j]) / (vg[j + 1] - vg[j])
        return (1.0 - w) * surf[j] + w * surf[j + 1]

    val = {nm: np.full(Sf.size, np.nan) for nm in GREEK_NAMES}
    for tj, Us in snaps.items():
        m = tf == tj
        xe = Sf[m] / Kf[m]
        Ux = np.zeros_like(Us)
        Ux[:, 1:-1] = u1m * Us[:, :-2] + u1c * Us[:, 1:-1] + u1p * Us[:, 2:]
        Uxx = np.zeros_like(Us)
        Uxx[:, 1:-1] = u2m * Us[:, :-2] + u2c * Us[:, 1:-1] + u2p * Us[:, 2:]
        Uv = np.zeros_like(Us)
        Uv[1:-1] = w1m[:, None] * Us[:-2] + w1c[:, None] * Us[1:-1] + w1p[:, None] * Us[2:]
        # vanna: v-first-derivative stencil applied to the delta slice Ux — the
        # same D1x∘D1v composition apply_a0 uses for the mixed PDE term, and it
        # composes exactly the two extraction stencils already trusted here for
        # delta (D1x of U) and vega (D1v of U). Delta is dimensionless in
        # (x = S/K, v), so vanna = d(delta)/dv needs no K factor.
        Uxv = np.zeros_like(Us)
        Uxv[1:-1] = w1m[:, None] * Ux[:-2] + w1c[:, None] * Ux[1:-1] + w1p[:, None] * Ux[2:]
        Ut = apply_a1(Us) + apply_a2(Us) + apply_a0(Us)
        val["price"][m] = Kf[m] * np.interp(xe, xg, v_interp(Us))
        val["delta"][m] = np.interp(xe, xg, v_interp(Ux))
        val["gamma"][m] = np.interp(xe, xg, v_interp(Uxx)) / Kf[m]
        val["vega"][m] = Kf[m] * np.interp(xe, xg, v_interp(Uv))
        val["vanna"][m] = np.interp(xe, xg, v_interp(Uxv))
        val["theta"][m] = -Kf[m] * np.interp(xe, xg, v_interp(Ut))
    return GreekSet(**{nm: val[nm].reshape(shape) for nm in GREEK_NAMES})


# ----------------------------------------------------------------------------
# Cross-validation of the legs and the public oracle entry point
# ----------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Consensus oracle output. mask[greek] True = point untrusted (legs disagree
    beyond tolerance, or a providing leg failed to converge there). `legs` keeps
    every leg's raw values for audit; nothing is dropped silently."""
    consensus: GreekSet
    uncertainty: GreekSet
    mask: dict
    feller_ratio: float
    trust_note: str
    legs: dict


def cross_validate(legs: dict, tol_rel: float, mc_se: GreekSet | None = None,
                   wing_floor_frac: float = 1e-2) -> tuple[GreekSet, GreekSet, dict]:
    """Per grid point: consensus = median across legs; uncertainty = max pairwise
    |difference| folded with the MC standard error; mask = any pairwise difference
    above tol_rel * scale (+ 3*SE for pairs involving the MC leg). The comparison
    scale is max(|consensus|, wing_floor_frac * global max |consensus|): relative
    on the band bounded away from zero, absolute in the wings (config metric
    convention). The same floor covers sign-changing Greeks — vanna crosses zero
    near ATM, where |consensus| collapses and the comparison goes absolute,
    exactly as gamma's wings do. A point-wise NaN from a providing leg masks the
    point: absence of evidence is treated as disagreement (a leg with NO
    estimator anywhere — e.g. MC vanna — is excluded from that Greek's panel
    instead). Rule declared pre-training."""
    names = list(legs)
    cons, unc, mask = {}, {}, {}
    for gname in GREEK_NAMES:
        provide = [n for n in names
                   if not np.all(np.isnan(getattr(legs[n], gname)))]
        vals = np.stack([getattr(legs[n], gname) for n in provide])
        cons[gname] = np.nanmedian(vals, axis=0)
        scale = np.maximum(np.abs(cons[gname]),
                           wing_floor_frac * np.nanmax(np.abs(cons[gname])))
        se = np.zeros_like(cons[gname])
        if mc_se is not None and "mc" in provide:
            se = np.nan_to_num(getattr(mc_se, gname))
        maxdiff = np.zeros_like(cons[gname])
        bad = np.zeros(cons[gname].shape, dtype=bool)
        for i in range(len(provide)):
            for j in range(i + 1, len(provide)):
                d = np.abs(vals[i] - vals[j])
                tol = tol_rel * scale
                if "mc" in (provide[i], provide[j]):
                    tol = tol + 3.0 * se
                bad |= d > tol
                maxdiff = np.fmax(maxdiff, d)
            bad |= np.isnan(vals[i])
        unc[gname] = np.maximum(maxdiff, se)
        mask[gname] = bad
    mask["any"] = np.any(np.stack([mask[g] for g in GREEK_NAMES]), axis=0)
    return GreekSet(**cons), GreekSet(**unc), mask


def load_config(cfg_path: str) -> dict:
    """Load the frozen benchmark YAML."""
    import yaml
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def oracle_greeks(cfg_path: str = "heston_benchmark_v6.yaml",
                  regime: str = "baseline", seed: int = 42,
                  mc_paths: int = 200_000,
                  near_feller_mc_multiplier: int = 4) -> OracleResult:
    """Run every applicable leg on the config (S, K, tau) grid for one regime and
    cross-validate. near_feller ('oracle: requires_fourth_leg') adds the ADI leg,
    multiplies the MC path budget, and is stamped least-trustworthy regardless of
    agreement (Feller ratio ~1; config F7: report WITH uncertainty, never drop)."""
    cfg = load_config(cfg_path)
    gc = cfg["grid"]
    reg = cfg["regimes"][regime]
    p = HestonParams(kappa=reg["kappa"], theta=reg["theta"], xi=reg["xi"],
                     rho=reg["rho"], v0=reg["v0"])
    # fourth-leg detection: oracle.fourth_leg.required_for is AUTHORITATIVE (the
    # inline regime flag is self-documentation only); the [0.40, 0.60] band clause
    # catches feller-stressed sampled hypercube points that are not named regimes.
    fourth = (regime in cfg["oracle"]["fourth_leg"]["required_for"]
              or 0.40 <= p.feller_ratio <= 0.60)
    assert (fourth or reg.get("oracle") == "requires_fourth_leg"
            or p.feller_ratio >= 0.5), (
        f"regime '{regime}' has feller_ratio {p.feller_ratio:.2f} < 0.5 but is not "
        "in oracle.fourth_leg.required_for, not flagged 'oracle: "
        "requires_fourth_leg', and outside the [0.40, 0.60] band — a 3-leg-only "
        "oracle is untrusted this far below the Feller boundary")
    r, q = float(gc["r"]), float(gc["q"])
    S, K, T = np.meshgrid(
        np.linspace(gc["S"]["min"], gc["S"]["max"], gc["S"]["n"]),
        np.linspace(gc["K"]["min"], gc["K"]["max"], gc["K"]["n"]),
        np.linspace(gc["tau"]["min"], gc["tau"]["max"], gc["tau"]["n"]),
        indexing="ij")
    tol = float(cfg["oracle"]["three_way_validation"]["agreement_tol_rel"])
    n_mc = mc_paths * (near_feller_mc_multiplier if fourth else 1)
    legs = {"cf": heston_greeks_cf(S, K, T, p, r, q),
            "fd": heston_greeks_fd(S, K, T, p, r, q)}
    legs["mc"], mc_se = heston_greeks_mc(S, K, T, p, r, q, n_paths=n_mc, seed=seed)
    if fourth:
        legs["adi"] = heston_greeks_adi(S, K, T, p, r, q)
    consensus, uncertainty, mask = cross_validate(legs, tol, mc_se)
    fr = p.feller_ratio
    note = f"regime={regime}, feller_ratio={fr:.2f}, legs={sorted(legs)}"
    if fourth or fr < 1.5:
        note += (" | NEAR-FELLER: least-trustworthy regime; oracle noise largest"
                 " here. Report WITH uncertainty bars; do not drop (config F7).")
    return OracleResult(consensus, uncertainty, mask, fr, note, legs)


# ----------------------------------------------------------------------------
# Self-tests. The oracle is UNTRUSTED until every one of these passes.
# ----------------------------------------------------------------------------

def _selftest(full: bool = False) -> None:
    r, q = 0.025, 0.0
    # 1) Literature: Albrecher et al. 2007, Table 1, phi_2 column (ATM % of spot)
    p_lit = HestonParams(kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711, v0=0.0175)
    refs = {1: 7.27, 2: 11.73, 3: 15.48, 4: 18.77, 5: 21.75, 10: 33.84, 15: 43.17}
    for T, ref in refs.items():
        got = float(heston_greeks_cf(100.0, 100.0, float(T), p_lit, r, q).price)
        assert abs(got - ref) < 0.02, f"Table1 T={T}: {got} vs {ref}"
    print("PASS 1: literature prices (Albrecher Table 1, phi_2 column, T=1..15)")

    # 2) TRUST GATE: xi -> 0 must reproduce Black-Scholes (deterministic variance)
    r2, q2 = 0.02, 0.0
    pg = HestonParams(kappa=2.0, theta=0.04, xi=1e-4, rho=-0.50, v0=0.03)
    S, T = np.meshgrid([80.0, 100.0, 120.0], [0.04, 0.5, 1.0], indexing="ij")
    K = np.full_like(S, 100.0)
    bs = bs_limit_reference(S, K, T, pg, r2, q2)
    for leg, lim in (("cf", 5e-4), ("fd", 2e-3)):
        g = (heston_greeks_cf if leg == "cf" else heston_greeks_fd)(S, K, T, pg, r2, q2)
        for nm, bk in (("price", "price"), ("delta", "delta"), ("gamma", "gamma"),
                       ("vega", "vega_v0"), ("theta", "theta")):
            scale = np.maximum(np.abs(bs[bk]), 1e-3 * np.max(np.abs(bs[bk])))
            rel = np.abs(getattr(g, nm) - bs[bk]) / scale
            worst = float(np.nanmax(rel))
            assert worst < lim, f"BS gate {leg}/{nm}: rel {worst:.2e} >= {lim}"
    # vanna joins the gate at a rho = 0 companion point: the finite-xi gap to the
    # BS limit is FIRST order in xi (the rho*xi*v*S d2/dSdv mixed term) while
    # vanna's ATM magnitude is small, so at rho = -0.5 no xi satisfies both
    # O(rho xi) < tol and roundoff < tol (measured: the gap scales linearly in xi
    # down to 5e-5, below which cancellation in the CF/COS dominates). At rho = 0
    # the leading correction is O(xi^2) and the limit is clean; the sign/units/
    # chain-rule errors this gate exists to catch are rho-independent, and
    # rho != 0 vanna is cross-validated by PASS 3 and PASS 5. Tolerances unchanged.
    pg0 = HestonParams(kappa=2.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.03)
    ref0 = bs_limit_reference(S, K, T, pg0, r2, q2)["vanna_v0"]
    scale0 = np.maximum(np.abs(ref0), 1e-3 * np.max(np.abs(ref0)))
    for leg, lim in (("cf", 5e-4), ("fd", 2e-3)):
        g = (heston_greeks_cf if leg == "cf" else heston_greeks_fd)(S, K, T, pg0, r2, q2)
        worst = float(np.nanmax(np.abs(g.vanna - ref0) / scale0))
        assert worst < lim, f"BS gate {leg}/vanna: rel {worst:.2e} >= {lim}"
    print("PASS 2: TRUST GATE xi->0 == Black-Scholes (legs cf, fd; all Greeks,"
          " vanna at the rho=0 companion point)")

    # 3) Legs agree on the baseline regime within tolerance
    pb = HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.50, v0=0.04)
    S = np.tile([80.0, 90.0, 100.0, 110.0, 125.0], 3)
    K = np.full_like(S, 100.0)
    T = np.repeat([0.04, 0.5, 1.0], 5)
    legs = {"cf": heston_greeks_cf(S, K, T, pb, r2, q2),
            "fd": heston_greeks_fd(S, K, T, pb, r2, q2)}
    legs["mc"], mc_se = heston_greeks_mc(S, K, T, pb, r2, q2, n_paths=100_000, seed=42)
    _, _, mask = cross_validate(legs, tol_rel=1e-3, mc_se=mc_se)
    n_bad = int(mask["any"].sum())
    assert n_bad == 0, f"baseline: {n_bad}/{S.size} points masked"
    # explicit CF-vs-FD vanna agreement (MC carries no vanna estimator, so the
    # vanna panel above is exactly these two legs; restated here for legibility)
    va_cf, va_fd = legs["cf"].vanna, legs["fd"].vanna
    scale = np.maximum(np.abs(va_cf), 1e-2 * np.max(np.abs(va_cf)))
    worst = float(np.nanmax(np.abs(va_cf - va_fd) / scale))
    assert worst < 1e-3, f"CF-vs-FD vanna baseline: rel {worst:.2e}"
    print("PASS 3: three legs agree within tol on baseline (0 masked of "
          f"{S.size} points; CF-vs-FD vanna worst rel {worst:.1e})")

    # 4) MC leg deterministic from seed
    a, _ = heston_greeks_mc(S[:5], K[:5], T[:5], pb, r2, q2, n_paths=20_000, seed=42)
    b, _ = heston_greeks_mc(S[:5], K[:5], T[:5], pb, r2, q2, n_paths=20_000, seed=42)
    assert all(np.array_equal(getattr(a, nm), getattr(b, nm), equal_nan=True)
               for nm in GREEK_NAMES), "MC not deterministic"
    print("PASS 4: MC leg bit-identical from seed")

    # 5) Pure FD sanity: 4-point cross-FD of the leg-A PRICE grid (bump +-h in S
    # and +-h in v0) must reproduce leg-A analytic vanna — catches any sign or
    # chain-rule slip in the analytic integrand independent of the COS leg.
    rng = np.random.default_rng(0)
    S5 = rng.uniform(85.0, 120.0, 5)
    K5 = np.full(5, 100.0)
    T5 = rng.uniform(0.15, 1.0, 5)
    hS, hv = 0.05, 5e-5

    def cf_price(dS, dv):
        pp = HestonParams(pb.kappa, pb.theta, pb.xi, pb.rho, pb.v0 + dv)
        return heston_greeks_cf(S5 + dS, K5, T5, pp, r2, q2).price

    cross = (cf_price(hS, hv) - cf_price(hS, -hv) - cf_price(-hS, hv)
             + cf_price(-hS, -hv)) / (4.0 * hS * hv)
    va = heston_greeks_cf(S5, K5, T5, pb, r2, q2).vanna
    scale = np.maximum(np.abs(va), 1e-2 * np.max(np.abs(va)))
    worst = float(np.max(np.abs(cross - va) / scale))
    assert worst < 1e-4, f"cross-FD vanna sanity: rel {worst:.2e}"
    print(f"PASS 5: 4-point cross-FD of CF price grid == analytic vanna "
          f"(worst rel {worst:.1e} < 1e-4, 5 random points)")

    if full:  # near_feller fourth leg vs CF leg, near-money band
        pn = HestonParams(kappa=1.5, theta=0.04, xi=0.34, rho=-0.70, v0=0.02)
        S = np.tile([95.0, 100.0, 105.0, 110.0], 3)
        K = np.full_like(S, 100.0)
        T = np.repeat([0.04, 0.5, 1.0], 4)
        ga = heston_greeks_cf(S, K, T, pn, r2, q2)
        gd = heston_greeks_adi(S, K, T, pn, r2, q2)
        for nm in ("price", "delta", "gamma", "vega", "vanna", "theta"):
            av, dv = getattr(ga, nm), getattr(gd, nm)
            scale = np.maximum(np.abs(av), 1e-2 * np.max(np.abs(av)))
            worst = float(np.nanmax(np.abs(av - dv) / scale))
            assert worst < 5e-3, f"ADI near_feller {nm}: {worst:.2e}"
        print("PASS 6: ADI fourth leg matches CF leg near-money (near_feller)")
        # vanna on the second Feller-stressed anchor (feller_violating_volvol),
        # same tolerance pattern as above
        pf = HestonParams(kappa=2.0, theta=0.04, xi=0.60, rho=-0.50, v0=0.04)
        gaf = heston_greeks_cf(S, K, T, pf, r2, q2)
        gdf = heston_greeks_adi(S, K, T, pf, r2, q2)
        av, dv = gaf.vanna, gdf.vanna
        scale = np.maximum(np.abs(av), 1e-2 * np.max(np.abs(av)))
        worst = float(np.nanmax(np.abs(av - dv) / scale))
        assert worst < 5e-3, f"ADI feller_violating_volvol vanna: {worst:.2e}"
        print("PASS 7: ADI vanna matches CF on both Feller-stressed anchors")
    print("ALL SELF-TESTS PASSED — trust gate cleared; oracle output usable.")


if __name__ == "__main__":
    import sys
    _selftest(full="--full" in sys.argv)
