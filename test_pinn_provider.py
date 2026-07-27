"""Tests for pinn_provider.py: GreekProvider conformance, exact parity with the
underlying SobolevPINN, a finite-difference consistency check THROUGH the provider,
v=0 safety / row isolation, determinism, and the build_providers registry.

A tiny TRAINED checkpoint is built in-module exactly as test_train.py does (the P8
label artifact via generate_labels with shrunk oracle legs, then a short train.main
run) so the real best.pt format — cfg dict + state_dict + frozen scales — is exercised
end to end. Baseline (theta_train) constants mirror heston_benchmark_v6.yaml.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import train
from make_labels import generate_labels
from pinn_provider import PINNProvider, build_providers

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
N_POINTS, N_SKT, SEED = 8, 4, 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}

R, Q = 0.02, 0.0
BASELINE_PARAMS = {"kappa": 2.0, "theta": 0.04, "xi": 0.30, "rho": -0.50, "v0": 0.04}
TRAIN_REGIME = {k: BASELINE_PARAMS[k] for k in ("kappa", "theta", "xi", "rho")}
K, TAU = 100.0, 0.17
GREEKS = ("price", "delta", "gamma", "vega", "vanna")


# ---------------------------------------------------------------------------
# fixtures: label artifact -> trained best.pt (train.py smoke settings) -> provider
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npz(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("pinn_prov_labels")
    res = generate_labels(CONTRACT, CFG, N_POINTS, SEED, str(out),
                          n_skt=N_SKT, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


@pytest.fixture(scope="module")
def ckpt_root(npz, tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpts")
    out = root / "rung3_delta_gamma_vega" / f"s{SEED}"
    # 500 steps: enough curvature that the h=1e-3 second-difference gamma check is
    # well above the float64 FD roundoff floor (a 40-step surface is ~flat, gamma~1e-6).
    train.main(["--arm", "rung3_delta_gamma_vega", "--seed", str(SEED), "--data", npz,
                "--pinn-cfg", CFG, "--contract", CONTRACT,
                "--lambdas", str(root / "absent.yaml"), "--out", str(out), "--steps", "500"])
    return root


@pytest.fixture(scope="module")
def provider(ckpt_root):
    path = ckpt_root / "rung3_delta_gamma_vega" / f"s{SEED}" / "best.pt"
    return PINNProvider(str(path), TRAIN_REGIME, R, Q)


# ---------------------------------------------------------------------------
# interface conformance
# ---------------------------------------------------------------------------

def test_greekprovider_conformance(provider):
    from Hedging_backtest import GreekProvider
    assert isinstance(provider, GreekProvider)


def test_evaluate_output_contract(provider):
    S = np.array([90.0, 100.0, 110.0])
    v = np.array([0.02, 0.04, 0.08])
    out = provider.evaluate(S, v, TAU, K)
    for nm in GREEKS:
        assert out[nm].shape == S.shape
        assert out[nm].dtype == np.float64
        assert np.all(np.isfinite(out[nm])), nm


# ---------------------------------------------------------------------------
# exact parity: the provider is a thin wrapper over model.greeks_eval + forward
# ---------------------------------------------------------------------------

def test_parity_with_direct_model_call(provider):
    rng = np.random.default_rng(0)
    S = rng.uniform(85.0, 120.0, 64)
    v = rng.uniform(0.01, 0.12, 64)
    out = provider.evaluate(S, v, TAU, K)

    x, shape = provider._build_x(S, v, TAU, K)
    chunk = (provider.chunk or provider.cfg.second_order_microbatch) or None
    g = provider.model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"), chunk=chunk)
    with torch.no_grad():
        price = provider.model(x)
    direct = {"price": price, "delta": g["delta"], "gamma": g["gamma"],
              "vega": g["vega"], "vanna": g["vanna"]}
    for nm in GREEKS:
        want = direct[nm].detach().to("cpu", torch.float64).numpy().reshape(shape)
        np.testing.assert_array_equal(out[nm], want, err_msg=nm)   # bit-equal, atol 0


# ---------------------------------------------------------------------------
# finite-difference consistency THROUGH the provider (price vs its own Greeks)
# ---------------------------------------------------------------------------

def test_fd_delta_gamma_through_provider(provider):
    rng = np.random.default_rng(1)
    S = rng.uniform(90.0, 110.0, 48)
    v = np.full_like(S, 0.04)
    h = 1e-3
    got = provider.evaluate(S, v, TAU, K)
    pp = provider.evaluate(S + h, v, TAU, K)["price"]
    pm = provider.evaluate(S - h, v, TAU, K)["price"]
    delta_fd = (pp - pm) / (2.0 * h)
    gamma_fd = (pp - 2.0 * got["price"] + pm) / h ** 2
    # rtol on the body; small atol floor guards any sign-crossing state
    np.testing.assert_allclose(delta_fd, got["delta"], rtol=1e-4,
                               atol=1e-4 * np.max(np.abs(got["delta"])))
    np.testing.assert_allclose(gamma_fd, got["gamma"], rtol=1e-4,
                               atol=1e-4 * np.max(np.abs(got["gamma"])))


# ---------------------------------------------------------------------------
# v = 0 safety: finite everywhere; v=0 rows never perturb v>0 rows (row-independent)
# ---------------------------------------------------------------------------

def test_v_zero_finite_and_no_clamp(provider):
    S = np.linspace(85.0, 115.0, 7)
    out = provider.evaluate(S, np.zeros_like(S), TAU, K)       # v in [0,0.01): mild extrapolation
    for nm in GREEKS:
        assert np.all(np.isfinite(out[nm])), nm


def test_mixed_v_zero_does_not_perturb_positive_rows(provider):
    S = np.linspace(85.0, 115.0, 7)
    v_pos = np.full_like(S, 0.04)
    out_pos = provider.evaluate(S, v_pos, TAU, K)
    v_mix = v_pos.copy()
    v_mix[::2] = 0.0                                            # even indices -> v=0
    out_mix = provider.evaluate(S, v_mix, TAU, K)
    keep = np.arange(S.size) % 2 == 1                          # odd indices stay v=0.04
    for nm in GREEKS:
        assert np.all(np.isfinite(out_mix[nm])), nm
        assert np.array_equal(out_mix[nm][keep], out_pos[nm][keep]), nm


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_deterministic(provider):
    S = np.linspace(85.0, 115.0, 9)
    v = np.linspace(0.0, 0.12, 9)
    a = provider.evaluate(S, v, TAU, K)
    b = provider.evaluate(S, v, TAU, K)
    for nm in GREEKS:
        assert np.array_equal(a[nm], b[nm]), nm


# ---------------------------------------------------------------------------
# build_providers registry
# ---------------------------------------------------------------------------

def test_build_providers_maps_names_and_oracle(ckpt_root):
    from providers import HestonCFProvider
    provs = build_providers(CONTRACT, ckpt_root, ["rung3"], SEED, R, Q)
    assert set(provs) == {"oracle", "rung3"}
    assert isinstance(provs["rung3"], PINNProvider)
    assert isinstance(provs["oracle"], HestonCFProvider)
    op = provs["oracle"].params
    for k, val in BASELINE_PARAMS.items():
        assert getattr(op, k) == val                           # oracle at theta_train baseline


def test_build_providers_include_oracle_false(ckpt_root):
    provs = build_providers(CONTRACT, ckpt_root, ["rung3"], SEED, R, Q, include_oracle=False)
    assert set(provs) == {"rung3"}


def test_build_providers_unknown_arm_raises_with_available(ckpt_root):
    with pytest.raises(FileNotFoundError) as ei:
        build_providers(CONTRACT, ckpt_root, ["standard_pinn"], SEED, R, Q)
    msg = str(ei.value)
    assert "standard_pinn" in msg                              # names the missing arm
    assert "rung3_delta_gamma_vega" in msg                     # lists the checkpoint that exists
