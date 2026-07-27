"""Gamma-label sources for the v6 dose-response arms.

NumPy/SciPy only (oracle-side, framework-independent). Delta and Vega labels stay
TRUE in every dose-response arm; only the GAMMA label varies (A2a/A6). The
gradient_penalty_only arm has labels: none per the contract YAML and never calls here.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from oracle import bs_call

GAMMA_LABEL_SOURCES = ("oracle", "bs_gamma", "shuffled", "none")


def implied_vol(price: float, S: float, K: float, tau: float, r: float, q: float,
                lo: float = 1e-6, hi: float = 5.0) -> float:
    """Black-Scholes implied vol of one call price by brentq."""
    return brentq(lambda sig: float(bs_call(S, K, tau, sig, r, q)["price"]) - price, lo, hi)


def bs_gamma_at_matched_iv(price: np.ndarray, S: np.ndarray, K: np.ndarray,
                           tau: np.ndarray, r: float, q: float) -> np.ndarray:
    """Cheap-label point: BS Gamma at the IV matching each oracle price."""
    ivs = np.array([implied_vol(float(p), float(s), float(k), float(t), r, q)
                    for p, s, k, t in zip(price, S, K, tau)])
    return np.asarray(bs_call(S, K, tau, ivs, r, q)["gamma"])


def make_gamma_labels(source: str, gamma_oracle: np.ndarray, sigma: float = 0.0,
                      seed: int = 42, *, price: np.ndarray | None = None,
                      S: np.ndarray | None = None, K: np.ndarray | None = None,
                      tau: np.ndarray | None = None, r: float = 0.02,
                      q: float = 0.0) -> np.ndarray | None:
    """Gamma labels for one arm; sigma>0 (oracle only) = gamma*(1+N(0,sigma)), v6."""
    assert source in GAMMA_LABEL_SOURCES
    assert sigma == 0.0 or source == "oracle"
    rng = np.random.default_rng(seed)
    if source == "none":
        return None
    if source == "oracle":
        g = np.asarray(gamma_oracle, dtype=float).copy()
        if sigma > 0.0:
            g *= 1.0 + rng.normal(0.0, sigma, g.shape)
        return g
    if source == "shuffled":   # structurally-wrong endpoint
        return rng.permutation(np.asarray(gamma_oracle, dtype=float))
    return bs_gamma_at_matched_iv(np.asarray(price, dtype=float), np.asarray(S, dtype=float),
                                  np.asarray(K, dtype=float), np.asarray(tau, dtype=float), r, q)
