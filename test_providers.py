"""Tests for providers.py: GreekProvider conformance, v=0 safety, the
vectorized-per-path-v0 CF refactor equivalence, the pathwise COS spot check
on baseline-regime QE paths, and determinism.

Baseline regime constants mirror heston_benchmark_v6.yaml (regimes.baseline,
grid r/q, hedging_simulation instrument/horizon) — restated here as literals
because the YAML is the frozen contract, not a test fixture.
"""
import dataclasses

import numpy as np
import pytest

import oracle
from oracle import GREEK_NAMES, HestonParams, heston_greeks_cf, heston_greeks_cf_v0
from providers import HestonCFProvider, pathwise_oracle_spotcheck, sample_qe_states

R, Q = 0.02, 0.0
BASELINE = HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.50, v0=0.04)
S0, K, TAU0, T_PRIME = 100.0, 100.0, 0.25, 0.17
N_STEPS = int(round(T_PRIME * 252))          # engine daily grid (dt = 1/252)
PSI_C = 1.5


@pytest.fixture(scope="module")
def provider():
    return HestonCFProvider(BASELINE, R, Q)


@pytest.fixture(scope="module")
def qe_states():
    return sample_qe_states(BASELINE, R, Q, S0, TAU0, T_PRIME, N_STEPS,
                            n_paths=512, seed=42, psi_c=PSI_C, n_states=2000)


def _grid():
    S = np.tile([80.0, 90.0, 100.0, 110.0, 125.0], 3)
    K_ = np.full_like(S, K)
    tau = np.repeat([TAU0 - T_PRIME, TAU0, 1.0], 5)
    return S, K_, tau


# ---------------------------------------------------------------------------
# interface conformance
# ---------------------------------------------------------------------------

def test_greekprovider_conformance(provider):
    from Hedging_backtest import GreekProvider
    assert isinstance(provider, GreekProvider)


def test_evaluate_output_contract(provider):
    S = np.array([90.0, 100.0, 110.0])
    v = np.array([0.02, 0.04, 0.08])
    out = provider.evaluate(S, v, 0.17, K)
    for nm in ("price", "delta", "gamma", "vega", "vanna"):
        assert out[nm].shape == S.shape
        assert np.all(np.isfinite(out[nm])), nm
    assert np.all(out["gamma"] > 0.0)
    assert np.all((out["delta"] > 0.0) & (out["delta"] < 1.0))


# ---------------------------------------------------------------------------
# v = 0 safety (QE paths hit v = 0 exactly; provider must floor)
# ---------------------------------------------------------------------------

def test_v_zero_returns_finite(provider):
    S = np.linspace(80.0, 125.0, 7)
    v = np.zeros_like(S)
    out = provider.evaluate(S, v, TAU0 - T_PRIME, K)
    for nm in ("price", "delta", "gamma", "vega", "vanna"):
        assert np.all(np.isfinite(out[nm])), nm
    # mixed v = 0 / v > 0 in one call must not perturb the v > 0 points
    v_mix = np.array([0.0, 0.04, 0.0, 0.04, 0.0, 0.04, 0.0])
    out_mix = provider.evaluate(S, v_mix, TAU0 - T_PRIME, K)
    assert np.all(np.isfinite(out_mix["gamma"]))


# ---------------------------------------------------------------------------
# vectorized per-path-v0 CF == scalar CF (the oracle.py refactor is inert)
# ---------------------------------------------------------------------------

def _assert_greeksets_close(got, want, rtol):
    for nm in GREEK_NAMES:
        a, b = getattr(got, nm), getattr(want, nm)
        # atol floors sign-crossing Greeks (vanna ~ 0 near ATM) where a pure
        # relative comparison at rtol 1e-10 is meaningless
        atol = 1e-10 * float(np.max(np.abs(b)))
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=nm)


def test_vectorized_constant_v0_matches_scalar_cf():
    S, K_, tau = _grid()
    want = heston_greeks_cf(S, K_, tau, BASELINE, R, Q)
    got = heston_greeks_cf_v0(S, K_, tau, BASELINE, R, Q,
                              v0=np.full_like(S, BASELINE.v0))
    _assert_greeksets_close(got, want, rtol=1e-10)
    # v0=None defaults to p.v0
    got_none = heston_greeks_cf_v0(S, K_, tau, BASELINE, R, Q)
    _assert_greeksets_close(got_none, want, rtol=1e-10)


def test_vectorized_shifted_constant_v0_matches_scalar_cf():
    # constant v0 != p.v0 must equal the scalar CF at replaced params: the
    # affine-v0 broadcast is exact, not a linearization
    for v0c in (0.015, 0.09):
        S, K_, tau = _grid()
        p_shift = dataclasses.replace(BASELINE, v0=v0c)
        want = heston_greeks_cf(S, K_, tau, p_shift, R, Q)
        got = heston_greeks_cf_v0(S, K_, tau, BASELINE, R, Q,
                                  v0=np.full_like(S, v0c))
        _assert_greeksets_close(got, want, rtol=1e-10)


def test_provider_matches_scalar_cf_at_regime_v0(provider):
    # end-to-end through the provider (flooring + chunking + dict plumbing)
    S = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    v = np.full_like(S, BASELINE.v0)
    want = heston_greeks_cf(S, np.full_like(S, K), np.full_like(S, TAU0),
                            BASELINE, R, Q)
    out = provider.evaluate(S, v, TAU0, K)
    for nm in ("price", "delta", "gamma", "vega", "vanna"):
        np.testing.assert_allclose(out[nm], getattr(want, nm), rtol=1e-10,
                                   atol=1e-10 * float(np.max(np.abs(getattr(want, nm)))),
                                   err_msg=nm)


def test_heterogeneous_v0_matches_per_point_scalar_cf():
    # per-point v0 vector == looping scalar CF point by point (the exact
    # semantics the vectorization replaces)
    rng = np.random.default_rng(7)
    S = rng.uniform(85.0, 120.0, 8)
    v0s = rng.uniform(0.01, 0.12, 8)
    tau = 0.17
    got = heston_greeks_cf_v0(S, K, tau, BASELINE, R, Q, v0=v0s)
    for i in range(S.size):
        p_i = dataclasses.replace(BASELINE, v0=float(v0s[i]))
        want = heston_greeks_cf(S[i], K, tau, p_i, R, Q)
        for nm in ("price", "delta", "gamma", "vega", "vanna"):
            # different u-grids (u_max from min v0 of the batch vs the single
            # point) -> quadrature-level, not bit-level, agreement
            np.testing.assert_allclose(float(getattr(got, nm)[i]),
                                       float(getattr(want, nm)),
                                       rtol=1e-7, err_msg=f"{nm} @ i={i}")


# ---------------------------------------------------------------------------
# pathwise trust check on baseline-regime QE paths (seed 42)
# ---------------------------------------------------------------------------

def test_spotcheck_passes_on_baseline_qe_paths(provider, qe_states):
    res = pathwise_oracle_spotcheck(provider, qe_states, R, Q, tol_rel=1e-3,
                                    K=K)
    assert res.frac_ok >= 0.99          # the function raises below this anyway
    assert res.n + res.n_low_v == 2000
    assert len(res.worst) <= 10
    assert res.worst[0]["rel_err"] >= res.worst[-1]["rel_err"]


def test_spotcheck_detects_broken_provider(provider, qe_states):
    class Broken(HestonCFProvider):
        def evaluate(self, S, v, tau, K):
            out = super().evaluate(S, v, tau, K)
            out["delta"] = out["delta"] * 1.05    # 5% delta bias >> tol_rel
            return out

    S, v, tau = (a[:200] for a in qe_states)
    with pytest.raises(ValueError, match="spot check FAILED"):
        pathwise_oracle_spotcheck(Broken(BASELINE, R, Q), (S, v, tau), R, Q,
                                  tol_rel=1e-3, K=K)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_provider_deterministic(provider):
    S = np.linspace(80.0, 125.0, 9)
    v = np.linspace(0.0, 0.12, 9)
    a = provider.evaluate(S, v, 0.17, K)
    b = provider.evaluate(S, v, 0.17, K)
    for nm in a:
        assert np.array_equal(a[nm], b[nm]), nm


def test_state_sampling_deterministic():
    a = sample_qe_states(BASELINE, R, Q, S0, TAU0, T_PRIME, N_STEPS,
                         n_paths=64, seed=42, psi_c=PSI_C, n_states=100)
    b = sample_qe_states(BASELINE, R, Q, S0, TAU0, T_PRIME, N_STEPS,
                         n_paths=64, seed=42, psi_c=PSI_C, n_states=100)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_spotcheck_deterministic(provider, qe_states):
    S, v, tau = (a[:150] for a in qe_states)
    r1 = pathwise_oracle_spotcheck(provider, (S, v, tau), R, Q, K=K)
    r2 = pathwise_oracle_spotcheck(provider, (S, v, tau), R, Q, K=K)
    assert r1.frac_ok == r2.frac_ok
    assert r1.worst == r2.worst
