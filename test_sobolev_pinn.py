"""Smoke tests for the SobolevPINN single-class arm design (no full training)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import qmc

from greek_labels import bs_gamma_at_matched_iv, make_gamma_labels
from oracle import GREEK_NAMES, HestonParams, bs_call, heston_greeks_cf
from SobolevPINN import PINNConfig, SobolevPINN, autodiff_greeks, load_arm
from ude import build_model
from train_pinn import (LockedTestSet, anchors_from_contract, capacity_control_config,
                        sample_hypercube_params, saturation_sweep_configs, select_lambda_pde,
                        set_seed, train_arm)

CFG = "pinn_config.yaml"
CONTRACT = "heston_benchmark_v6.yaml"
RAW = yaml.safe_load(open(CFG))
ANCHORS = anchors_from_contract(CONTRACT)
ARMS = list(RAW["arms"])
LADDER = {"rung0_price_only": (False, False, False), "rung1_delta": (True, False, False),
          "rung2_delta_gamma": (True, True, False), "rung3_delta_gamma_vega": (True, True, True)}
RANGES = {"S": (50.0, 150.0), "K": (60.0, 140.0), "tau": (0.04, 1.0), "kappa": (1.0, 4.0),
          "theta": (0.02, 0.12), "xi": (0.20, 0.60), "rho": (-0.80, -0.20),
          "v0": (0.01, 0.12), "v": (0.01, 0.12)}


def small_cfg(**kw) -> PINNConfig:
    base = dict(n_layers=3, width=32, supervise_delta=True, supervise_gamma=True,
                supervise_vega=True)
    base.update(kw)
    return PINNConfig(**base)


def make_batch(cfg: PINNConfig, n: int = 64, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    pnames = [k for k in ("kappa", "theta", "xi", "rho", "v0") if k in cfg.inputs]
    params: dict[str, torch.Tensor] = {}
    if {"kappa", "theta", "xi"} <= set(pnames):   # v6 hypercube: Feller-constrained sampling
        pts = sample_hypercube_params({k: RANGES[k] for k in pnames}, n, seed,
                                      feller_min=RAW["hypercube_sampling"]["feller_min"],
                                      anchors=ANCHORS,
                                      excise_rel_radius=RAW["hypercube_sampling"]["excision"]["rel_radius"])
        params = {k: torch.as_tensor(pts[:, j], dtype=torch.float32)
                  for j, k in enumerate(pnames)}
    x = torch.stack([params.get(k, torch.empty(n).uniform_(*RANGES[k], generator=gen))
                     for k in cfg.inputs], 1)
    i = {name: k for k, name in enumerate(cfg.inputs)}
    S, K, tau, v = x[:, i["S"]], x[:, i["K"]], x[:, i["tau"]], x[:, i["v0" if "v0" in i else "v"]]
    price = (S - K).clamp(min=0.0) + 5.0 * v * S / 100.0
    return {"x": x, "price": price,
            "delta": torch.sigmoid((S - K) / 10.0),
            "gamma": torch.exp(-((S - K) / 20.0) ** 2) / 20.0,
            "vega": 0.5 * S * torch.sqrt(v),
            "vanna": 0.01 * torch.ones(n),
            "x_bc": x.clone(), "price_bc": price.clone()}


SCALE_TERMS = ("price", "delta", "gamma", "vega", "vanna", "bc", "pde")
SCALE_TERM_TO_LABEL = {"price": "price", "delta": "delta", "gamma": "gamma",
                       "vega": "vega", "vanna": "vanna", "bc": "price_bc"}


def realistic_scale_batch(cfg: PINNConfig, n: int = 256, seed: int = 0) -> dict[str, torch.Tensor]:
    """make_batch inputs with labels at REAL Heston magnitudes: price O(30), delta O(0.5),
    gamma O(0.02), vega O(20) — the scale gap that makes the raw gamma MSE ~6 orders
    below price/vega at equal relative error."""
    batch = make_batch(cfg, n=n, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    batch["price"] = 30.0 * (1.0 + 0.3 * torch.rand(n, generator=gen))
    batch["delta"] = 0.5 * (1.0 + 0.5 * torch.rand(n, generator=gen))
    batch["gamma"] = 0.02 * (1.0 + 0.5 * torch.rand(n, generator=gen))
    batch["vega"] = 20.0 * (1.0 + 0.5 * torch.rand(n, generator=gen))
    batch["vanna"] = 0.5 * (1.0 + 0.5 * torch.rand(n, generator=gen))
    batch["price_bc"] = batch["price"].clone()
    return batch


def test_loss_scale_normalized_terms_are_o1_comparable_at_equal_relative_error():
    """(a) At an EQUAL 5% relative error, every normalized supervised term must sit
    within 2 orders of magnitude of every other — versus ~6 orders raw (gamma vs price)."""
    cfg = small_cfg(supervise_vanna=True, lambda_vanna=1.0)
    batch = realistic_scale_batch(cfg)
    set_seed(0)
    model = SobolevPINN(cfg)
    model.loss(batch)   # first call freezes the scale buffers from THIS batch's labels
    assert bool(model.loss_scales_frozen)
    vals = {}
    for term, key in SCALE_TERM_TO_LABEL.items():
        pred = batch[key] * 1.05   # exact 5% relative error
        vals[term] = float(F.mse_loss(pred, batch[key]) / getattr(model, f"loss_scale_{term}"))
    assert max(vals.values()) / min(vals.values()) < 100.0, vals
    # counterfactual: the raw-unit gap the normalization removes is >= 4 orders
    raw_gamma = float(F.mse_loss(batch["gamma"] * 1.05, batch["gamma"]))
    raw_price = float(F.mse_loss(batch["price"] * 1.05, batch["price"]))
    assert raw_price / raw_gamma > 1e4


def test_loss_scale_buffers_identical_across_arms_and_frozen_after_first_batch():
    """(b) Scale buffers are computed flag-independently from the batch labels: two
    different arms built from the same training data must hold BIT-IDENTICAL buffers
    (arms differ only by flags, never by normalization constants), the buffers must
    serialize in state_dict, and a later batch must not move them (stationary loss)."""
    arm_a, arm_b = load_arm(CFG, "rung1_delta"), load_arm(CFG, "rung3_delta_gamma_vega")
    batch = make_batch(arm_a, n=64, seed=11)
    set_seed(0)
    ma = SobolevPINN(arm_a)
    set_seed(0)
    mb = SobolevPINN(arm_b)
    ma.loss(batch)
    mb.loss({k: v.clone() for k, v in batch.items()})
    for term in SCALE_TERMS:
        a, b = getattr(ma, f"loss_scale_{term}"), getattr(mb, f"loss_scale_{term}")
        assert torch.equal(a, b), term
        assert f"loss_scale_{term}" in ma.state_dict(), term
    assert "loss_scales_frozen" in ma.state_dict()
    # frozen after the first computation: a second, different batch changes nothing
    snap = {k: v.clone() for k, v in ma.state_dict().items() if k.startswith("loss_scale")}
    ma.loss(make_batch(arm_a, n=64, seed=99))
    for k, v in snap.items():
        assert torch.equal(ma.state_dict()[k], v), k
    # sobolev_sans_pde == lambda_pde_zero loss-dict identity survives normalization
    set_seed(2)
    m_sans = SobolevPINN(load_arm(CFG, "sobolev_sans_pde"))
    set_seed(2)
    m_zero = SobolevPINN(load_arm(CFG, "lambda_pde_zero"))
    b2 = make_batch(m_sans.cfg, n=16, seed=4)
    t_sans = m_sans.loss(b2)
    t_zero = m_zero.loss({k: v.clone() for k, v in b2.items()})
    assert t_sans.keys() == t_zero.keys() and "pde" not in t_sans
    for k in t_sans:
        assert torch.equal(t_sans[k], t_zero[k]), k


def test_gamma_term_gradients_within_orders_of_price_term_gradients():
    """(c) The whole point of the fix: parameter gradients through the normalized gamma
    term must be within a few orders of the price-term gradients — measured raw ratio
    is ~1e-8 (numerically inert); normalized it is ~2e-2."""
    def grad_norm(model, term):
        gs = torch.autograd.grad(term, list(model.parameters()),
                                 retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum(g.pow(2).sum() for g in gs if g is not None)))

    cfg = small_cfg()
    batch = realistic_scale_batch(cfg)
    set_seed(0)
    model = SobolevPINN(cfg)
    terms = model.loss(batch)
    ratio = grad_norm(model, terms["gamma"]) / grad_norm(model, terms["price"])
    assert 1e-3 < ratio < 1e3, ratio
    # and the raw-mode counterfactual really is inert
    set_seed(0)
    m_raw = SobolevPINN(small_cfg(loss_scale_mode="raw"))
    t_raw = m_raw.loss(batch)
    raw_ratio = grad_norm(m_raw, t_raw["gamma"]) / grad_norm(m_raw, t_raw["price"])
    assert raw_ratio < 1e-6, raw_ratio


def test_loss_scale_mode_raw_reproduces_legacy_loss_exactly():
    """(d) loss_scale_mode='raw' is the pre-normalization loss bit-for-bit: plain
    raw-unit MSE per term, scale buffers untouched at 1.0, nothing frozen."""
    cfg = small_cfg(loss_scale_mode="raw", supervise_vanna=True, lambda_vanna=1.0)
    batch = make_batch(cfg, n=32, seed=3)
    set_seed(5)
    model = SobolevPINN(cfg)
    terms = model.loss(batch)
    assert not bool(model.loss_scales_frozen)
    for term in SCALE_TERMS:
        assert torch.equal(getattr(model, f"loss_scale_{term}"),
                           torch.ones((), dtype=torch.float32)), term
    x = batch["x"].clone().requires_grad_(True)
    g = model.greeks(x, ("delta", "gamma", "vega", "vanna"))
    for term, key in SCALE_TERM_TO_LABEL.items():
        if term == "bc":
            continue
        assert torch.equal(terms[term], F.mse_loss(g[term], batch[key])), term
    assert torch.equal(terms["bc"], F.mse_loss(model(batch["x_bc"]), batch["price_bc"]))
    # normalized mode differs from raw ONLY by the frozen per-term scale division
    set_seed(5)
    m_norm = SobolevPINN(small_cfg(supervise_vanna=True, lambda_vanna=1.0))
    t_norm = m_norm.loss(batch)
    assert terms.keys() == t_norm.keys()
    for term in SCALE_TERMS:
        scale = getattr(m_norm, f"loss_scale_{term}")
        assert torch.allclose(t_norm[term] * scale, terms[term], rtol=1e-6), term


def test_overfit_single_batch_gradients_flow_through_gamma_and_vega():
    cfg = small_cfg()
    batch = make_batch(cfg, n=64)
    set_seed(42)
    model = SobolevPINN(cfg)
    terms = model.loss(batch)
    for term in ("gamma", "vega"):
        grads = torch.autograd.grad(terms[term], list(model.parameters()),
                                    retain_graph=True, allow_unused=True)
        assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads), term
    _, log = train_arm(cfg, batch, steps=20, lr=1e-2, seed=42)
    assert log["losses"][-1] < log["losses"][0]
    acct = log["compute"]
    assert acct["param_count"] > 0 and acct["derivative_evals"] > 0
    assert acct["wall_clock_s"] > 0 and acct["peak_memory_bytes"] > 0


def test_autodiff_gamma_and_vanna_match_finite_difference():
    def toy(x):   # V = S^3 * exp(v/2): smooth, non-trivial cross term
        return x[:, 0] ** 3 * torch.exp(0.5 * x[:, 1])

    x = torch.tensor([[0.7, 0.3], [1.3, -0.2], [2.0, 0.5]], dtype=torch.float64)
    g = autodiff_greeks(toy, x.clone().requires_grad_(True), 0, 1,
                        need=("delta", "gamma", "vega", "vanna"))
    h = 1e-3
    eS = torch.tensor([h, 0.0], dtype=torch.float64)
    eV = torch.tensor([0.0, h], dtype=torch.float64)
    gamma_fd = (toy(x + eS) - 2.0 * toy(x) + toy(x - eS)) / h ** 2
    vanna_fd = (toy(x + eS + eV) - toy(x + eS - eV)
                - toy(x - eS + eV) + toy(x - eS - eV)) / (4.0 * h ** 2)
    assert torch.allclose(g["gamma"], gamma_fd, rtol=1e-5)
    assert torch.allclose(g["vanna"], vanna_fd, rtol=1e-5)
    assert torch.allclose(g["delta"], 3.0 * x[:, 0] ** 2 * torch.exp(0.5 * x[:, 1]), rtol=1e-8)
    assert torch.allclose(g["vega"], 0.5 * toy(x), rtol=1e-8)


def test_model_autodiff_greeks_match_raw_coordinate_fd_float64():
    """THE decisive normalization test: FD on the REAL model bumping RAW columns vs
    model.greeks() autodiff, float64. If the [-1,1] input map ever leaked into the
    differentiation coordinates, gamma would be off by ~(scale_S/2)^-2 = 2500x and
    this screams; a wrong output scale would miss K_ref = 100x on every Greek."""
    cfg = small_cfg(dtype="float64")
    set_seed(3)
    model = SobolevPINN(cfg)
    x = make_batch(cfg, n=8, seed=5)["x"].to(torch.float64)
    g = model.greeks(x.clone().requires_grad_(True), need=("delta", "gamma", "vega"))

    def f(col: int, h: float) -> torch.Tensor:   # price with ONE raw column bumped
        with torch.no_grad():
            xb = x.clone()
            xb[:, col] += h
            return model(xb)

    h = 1e-3   # raw S units on S~100
    delta_fd = (f(model.i_s, h) - f(model.i_s, -h)) / (2.0 * h)
    assert torch.allclose(g["delta"], delta_fd, rtol=1e-5)

    def gamma_fd_at(hh: float) -> torch.Tensor:
        return (f(model.i_s, hh) - 2.0 * f(model.i_s, 0.0) + f(model.i_s, -hh)) / hh ** 2

    gamma_fd = (4.0 * gamma_fd_at(h) - gamma_fd_at(2.0 * h)) / 3.0   # Richardson
    assert torch.allclose(g["gamma"], gamma_fd, rtol=1e-5, atol=1e-7)

    hv = 1e-5   # raw v0 units on v0 in [0.01, 0.12]
    vega_fd = (f(model.i_v, hv) - f(model.i_v, -hv)) / (2.0 * hv)
    assert torch.allclose(g["vega"], vega_fd, rtol=1e-5)


def test_out_of_range_inputs_extrapolate_linearly_no_clamp():
    """strong_neg_corr (rho=-0.9) lies BELOW the sampled rho range [-0.8,-0.2]: genuine
    extrapolation. The normalization must map it linearly past the [-1,1] edge — never
    clamp — and the model must return finite price and Greeks there."""
    import inspect

    from SobolevPINN import MLP
    src = inspect.getsource(MLP.forward)
    assert "torch.clamp" not in src and ".clamp(" not in src   # no clamp CALL in the path
    cfg = small_cfg()
    set_seed(0)
    model = SobolevPINN(cfg)
    x = make_batch(cfg, n=4)["x"]
    x[:, model.i["rho"]] = -0.9
    xn = (x - model.net.in_lo) * model.net.in_scale - 1.0
    assert (xn[:, model.i["rho"]] < -1.0).all()
    expected = 2.0 * (-0.9 - (-0.80)) / ((-0.20) - (-0.80)) - 1.0   # = -4/3, linear past edge
    assert torch.allclose(xn[:, model.i["rho"]],
                          torch.full((4,), expected), rtol=1e-6)
    g = model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"))
    for k, t in g.items():
        assert torch.isfinite(t).all(), k
    # normalization constants are buffers: serialize in state_dict, follow .to(dtype)
    assert {"net.in_lo", "net.in_scale", "net.k_ref"} <= set(model.state_dict())


def test_pde_residual_finite_and_in_raw_units_after_normalization():
    """Residual stays in raw price/time units — normalization lives inside the net, so
    autodiff w.r.t. raw (S, v, tau) is unchanged in meaning: finite everywhere and of
    fresh-init magnitude (no scale-squared blowup from a leaked Jacobian factor)."""
    cfg = small_cfg()
    set_seed(1)
    model = SobolevPINN(cfg)
    x = make_batch(cfg, n=64)["x"].clone().requires_grad_(True)
    res = model.pde_residual(x)
    assert res.shape == (64,) and torch.isfinite(res).all()
    assert float(res.abs().mean()) < 1e4   # fresh-init residual is O(1e0-1e3) in raw units


def test_greeks_eval_chunking_matches_full_batch():
    cfg = small_cfg()
    set_seed(0)
    model = SobolevPINN(cfg)
    x = make_batch(cfg, n=17)["x"]
    full = model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"))
    chunked = model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"), chunk=5)
    for k in full:
        assert torch.allclose(full[k], chunked[k]), k


# Keys the arm-swap identity check ignores: the UDE arm's residual-correction net is an
# ALLOWED extra-parameter set (family "ude"); every price-head key must still be bit-identical.
ARM_SWAP_ALLOWLIST_PREFIXES = ("correction.",)


def _price_head_state(model) -> dict:
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(ARM_SWAP_ALLOWLIST_PREFIXES)}


def test_arm_swap_changes_only_the_loss_identical_params():
    ref = None
    for arm in ARMS:
        set_seed(42)
        # build_model routes the ude arm to UDESobolevPINN (adds correction.*); all others
        # to plain SobolevPINN. Correction keys are allow-listed OUT; the shared price head
        # (net.*, loss_scale_*) must remain bit-identical across every arm.
        head = _price_head_state(build_model(load_arm(CFG, arm)))
        if ref is None:
            ref = head
            continue
        assert head.keys() == ref.keys(), arm
        for k in ref:
            assert torch.equal(head[k], ref[k]), (arm, k)


def test_seed_determinism():
    cfg = small_cfg()
    batch = make_batch(cfg)
    m1, _ = train_arm(cfg, batch, steps=5, lr=1e-2, seed=7)
    m2, _ = train_arm(cfg, batch, steps=5, lr=1e-2, seed=7)
    m3, _ = train_arm(cfg, batch, steps=5, lr=1e-2, seed=8)
    p1, p2, p3 = (list(m.parameters()) for m in (m1, m2, m3))
    assert all(torch.equal(a, b) for a, b in zip(p1, p2))
    assert any(not torch.equal(a, c) for a, c in zip(p1, p3))


CONTRACT_METHOD_TO_ARM = {
    "baseline0_feedforward": "feedforward",
    "baseline1_standard_pinn": "standard_pinn",
    "sobolev_pinn": "rung3_delta_gamma_vega",
    "sobolev_sans_pde": "sobolev_sans_pde",
    "ladder_rung0_price": "rung0_price_only",
    "ladder_rung1_delta": "rung1_delta",
    "ladder_rung2_delta_gamma": "rung2_delta_gamma",
    "ladder_rung3_dgv": "rung3_delta_gamma_vega",
    "optional_vanna_arm": "optional_vanna_arm",
}


def test_config_coverage_every_named_arm_is_one_class():
    for arm in ARMS:
        cfg = load_arm(CFG, arm)
        model = SobolevPINN(cfg)
        assert type(model) is SobolevPINN, arm
        terms = model.loss(make_batch(cfg, n=16))
        assert torch.isfinite(terms["total"]), arm
        if arm in LADDER:
            assert (cfg.supervise_delta, cfg.supervise_gamma, cfg.supervise_vega) == LADDER[arm]
            assert cfg.use_pde and cfg.lambda_pde != 0.0
    r0, std = load_arm(CFG, "rung0_price_only"), load_arm(CFG, "standard_pinn")
    assert (std.use_pde, std.supervise_delta, std.supervise_gamma, std.supervise_vega) == \
           (r0.use_pde, r0.supervise_delta, r0.supervise_gamma, r0.supervise_vega)
    assert not load_arm(CFG, "feedforward").use_pde
    for name, sig in (("sigma_000", 0.0), ("sigma_010", 0.10), ("sigma_025", 0.25),
                      ("sigma_050", 0.50)):
        cfg = load_arm(CFG, name)
        assert cfg.label_source == "oracle" and cfg.gamma_label_noise_sigma == sig
        assert cfg.supervise_delta and cfg.supervise_vega   # A2a: Delta/Vega labels stay TRUE
    assert load_arm(CFG, "bs_gamma").label_source == "bs_gamma"
    assert load_arm(CFG, "shuffled").label_source == "shuffled"
    gp = load_arm(CFG, "gradient_penalty_only")
    assert gp.gradient_penalty and not gp.supervise_gamma and gp.label_source == "none"

    # (a) every contract method has a mapped arm that builds and trains to a finite loss.
    for method, arm in CONTRACT_METHOD_TO_ARM.items():
        assert arm in ARMS, method
        cfg = load_arm(CFG, arm)
        terms = SobolevPINN(cfg).loss(make_batch(cfg, n=16))
        assert torch.isfinite(terms["total"]), (method, arm)

    # (b) sobolev_sans_pde has no PDE term in its loss dict.
    sans_pde = load_arm(CFG, "sobolev_sans_pde")
    terms = SobolevPINN(sans_pde).loss(make_batch(sans_pde, n=16))
    assert "pde" not in terms

    # (c) rung2 has vega OFF and gamma ON.
    rung2 = load_arm(CFG, "rung2_delta_gamma")
    assert rung2.supervise_gamma and not rung2.supervise_vega

    # (d) optional_vanna_arm has supervise_vanna and a finite loss.
    vanna_arm = load_arm(CFG, "optional_vanna_arm")
    assert vanna_arm.supervise_vanna and vanna_arm.lambda_vanna > 0.0
    vanna_terms = SobolevPINN(vanna_arm).loss(make_batch(vanna_arm, n=16))
    assert torch.isfinite(vanna_terms["total"])


def test_lambda_pde_zero_is_valid_and_skips_pde_term():
    cfg = load_arm(CFG, "lambda_pde_zero")
    assert cfg.lambda_pde == 0.0
    model = SobolevPINN(cfg)
    batch = make_batch(cfg, n=16)
    terms = model.loss(batch)
    assert "pde" not in terms and torch.isfinite(terms["total"])
    _, log = train_arm(cfg, batch, steps=2, lr=1e-2, seed=42)
    assert np.isfinite(log["losses"]).all()


def test_gamma_label_sources():
    gamma = np.linspace(0.01, 0.05, 8)
    assert np.array_equal(make_gamma_labels("oracle", gamma), gamma)
    noisy = make_gamma_labels("oracle", gamma, sigma=0.25, seed=1)
    assert noisy.shape == gamma.shape and not np.array_equal(noisy, gamma)
    shuf = make_gamma_labels("shuffled", gamma, seed=1)
    assert not np.array_equal(shuf, gamma) and np.array_equal(np.sort(shuf), np.sort(gamma))
    assert make_gamma_labels("none", gamma) is None
    S, K, tau, sig, r, q = (np.array([100.0, 90.0]), np.array([100.0, 110.0]),
                            np.array([0.5, 0.25]), np.array([0.2, 0.35]), 0.02, 0.0)
    bs = bs_call(S, K, tau, sig, r, q)
    got = bs_gamma_at_matched_iv(np.asarray(bs["price"]), S, K, tau, r, q)
    assert np.allclose(got, bs["gamma"], rtol=1e-6)   # IV round-trip recovers BS Gamma


def test_oracle_vanna_matches_network_fd_convention():
    """Oracle vanna and the network-side autodiff diagnostic must share ONE
    convention — d2V/dSdv0 (dV/dv0 vega units, not sigma-vega), same sign — so
    the label and the diagnostic can never silently diverge. Verified with the
    identical 4-point cross-FD stencil that
    test_autodiff_gamma_and_vanna_match_finite_difference pins the network to."""
    assert "vanna" in GREEK_NAMES
    p = HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.50, v0=0.04)  # baseline regime
    r, q = 0.02, 0.0
    S = np.array([90.0, 105.0, 120.0])      # straddle the ATM sign change of vanna
    K = np.full(3, 100.0)
    tau = np.full(3, 0.5)
    hS, hv = 0.05, 5e-5

    def price(dS, dv):
        pp = HestonParams(p.kappa, p.theta, p.xi, p.rho, p.v0 + dv)
        return heston_greeks_cf(S + dS, K, tau, pp, r, q).price

    vanna_fd = (price(hS, hv) - price(hS, -hv) - price(-hS, hv)
                + price(-hS, -hv)) / (4.0 * hS * hv)
    got = heston_greeks_cf(S, K, tau, p, r, q).vanna
    assert np.allclose(got, vanna_fd, rtol=1e-4)
    assert np.array_equal(np.sign(got), np.sign(vanna_fd))   # convention lock


def test_hypercube_sampling_feller_constraint():
    hs = RAW["hypercube_sampling"]
    pts = sample_hypercube_params(hs["ranges"], 500, seed=42, feller_min=hs["feller_min"])
    names = list(hs["ranges"])
    assert pts.shape == (500, len(names))
    k, t, x = (pts[:, names.index(nm)] for nm in ("kappa", "theta", "xi"))
    assert (2.0 * k * t / x ** 2 >= 0.40).all()
    lo = np.array([hs["ranges"][nm][0] for nm in names])
    hi = np.array([hs["ranges"][nm][1] for nm in names])
    assert (pts >= lo).all() and (pts <= hi).all()
    with pytest.raises(RuntimeError):   # infeasible feller_min terminates with a clear error
        sample_hypercube_params(hs["ranges"], 10, seed=0, feller_min=50.0, max_rounds=5)


def test_hypercube_anchor_excision_holds_out_every_named_regime():
    hs = RAW["hypercube_sampling"]
    names = list(hs["ranges"])
    radius = hs["excision"]["rel_radius"]
    # engine copy in pinn_config.yaml must not drift from the contract (source of truth)
    cfg_anchors = {tuple(float(a[k]) for k in names) for a in hs["excision"]["anchors"].values()}
    assert cfg_anchors == {tuple(a[k] for k in names) for a in ANCHORS} and len(ANCHORS) == 5
    pts = sample_hypercube_params(hs["ranges"], 5000, seed=7, feller_min=hs["feller_min"],
                                  anchors=ANCHORS, excise_rel_radius=radius)
    assert pts.shape == (5000, len(names))
    lo = np.array([hs["ranges"][nm][0] for nm in names])
    hi = np.array([hs["ranges"][nm][1] for nm in names])
    un = (pts - lo) / (hi - lo)
    r_norm = radius * np.sqrt(len(names))
    in_cube = 0
    for a in ANCHORS:   # distance rule holds for EVERY anchor, in-cube or not
        au = (np.array([a[k] for k in names]) - lo) / (hi - lo)
        assert np.linalg.norm(un - au, axis=1).min() >= r_norm
        in_cube += bool((au >= 0.0).all() and (au <= 1.0).all())
    assert in_cube == 4   # all but strong_neg_corr (rho=-0.90) lie inside the hypercube
    k, t, x = (pts[:, names.index(nm)] for nm in ("kappa", "theta", "xi"))
    assert (2.0 * k * t / x ** 2 >= hs["feller_min"]).all()   # Feller filter still applies
    with pytest.raises(RuntimeError):   # absurd radius excises the whole cube -> clear error
        sample_hypercube_params(hs["ranges"], 10, seed=0, feller_min=hs["feller_min"],
                                anchors=ANCHORS, excise_rel_radius=5.0, max_rounds=5)


def test_hypercube_excision_off_reproduces_legacy_sampler_bitwise():
    hs = RAW["hypercube_sampling"]
    names = list(hs["ranges"])
    lo = np.array([hs["ranges"][nm][0] for nm in names])
    hi = np.array([hs["ranges"][nm][1] for nm in names])
    ik, it, ix = names.index("kappa"), names.index("theta"), names.index("xi")

    def legacy(n: int, seed: int) -> np.ndarray:   # pre-excision sampler, verbatim
        rng = np.random.default_rng(seed)
        kept, total = [], 0
        for _ in range(100):
            u = qmc.LatinHypercube(d=len(names), seed=rng).random(n)
            pts = lo + u * (hi - lo)
            pts = pts[2.0 * pts[:, ik] * pts[:, it] / pts[:, ix] ** 2 >= hs["feller_min"]]
            kept.append(pts)
            total += pts.shape[0]
            if total >= n:
                break
        return np.concatenate(kept)[:n]

    got = sample_hypercube_params(hs["ranges"], 500, seed=42, feller_min=hs["feller_min"],
                                  anchors=None)
    assert np.array_equal(got, legacy(500, 42))


def test_lambda_selection_touches_validation_only():
    guard = LockedTestSet({"x": torch.zeros(1)})
    with pytest.raises(RuntimeError):
        _ = guard.data
    out = select_lambda_pde(RAW["sweeps"]["lambda_pde"], lambda lam: (lam - 0.1) ** 2, guard)
    assert out["selected_lambda_pde"] == 0.1 and 0.0 in out["validation_scores"]
    assert guard.unlock() is not None


def test_info_matching_hooks():
    base = load_arm(CFG, "info_matched_baseline")
    im = RAW["info_matching"]
    cfgs = saturation_sweep_configs(base, im["n_price_points_multipliers"], im["cap_multiplier"])
    assert [c.n_price_points for c in cfgs] == [m * base.n_price_points
                                                for m in im["n_price_points_multipliers"]]
    with pytest.raises(AssertionError):
        saturation_sweep_configs(base, [6], im["cap_multiplier"])   # no exceeding the 5N cap
    wide = capacity_control_config(base, im["capacity_control_width_mult"])
    n_params = lambda c: sum(p.numel() for p in SobolevPINN(c).parameters())
    assert n_params(wide) > n_params(base)


def test_gamma_loss_scale_uses_gamma_ref_invariant_across_dose_arms():
    """A6 invariance: with a true-consensus gamma_ref in the batch, every
    dose-response arm (noisy / shuffled / bs_gamma) AND the gradient-penalty
    arm freeze the SAME loss_scale_gamma; without it, the noisy arm's scale
    would follow its corrupted labels (the confound this guards against)."""
    cfg0 = small_cfg()
    base = make_batch(cfg0, n=64, seed=11)
    true_gamma = base["gamma"].clone()
    noisy = true_gamma * (1.0 + 0.5 * torch.randn_like(true_gamma))

    def scale_of(cfg, batch):
        set_seed(0)
        m = SobolevPINN(cfg)
        m.loss(batch)
        return float(m.loss_scale_gamma)

    b_clean = {**base, "gamma": true_gamma, "gamma_ref": true_gamma}
    b_noisy = {**base, "gamma": noisy, "gamma_ref": true_gamma}
    s_clean = scale_of(small_cfg(), b_clean)
    s_noisy = scale_of(small_cfg(), b_noisy)
    assert s_clean == s_noisy  # bit-identical: same ref tensor drives both
    ref_scale = float(true_gamma.pow(2).mean() + 1e-12)
    assert abs(s_clean - ref_scale) < 1e-12
    # gradient-penalty arm: no "gamma" key, but gamma_ref still sets the unit
    gp_cfg = small_cfg(supervise_gamma=False, supervise_delta=False,
                       supervise_vega=False, label_source="none",
                       gradient_penalty=True)
    b_gp = {k: v for k, v in b_clean.items() if k != "gamma"}
    assert scale_of(gp_cfg, b_gp) == s_clean
    # fallback (legacy batches without gamma_ref) still uses the arm's labels
    b_legacy = {k: v for k, v in b_noisy.items() if k != "gamma_ref"}
    s_legacy = scale_of(small_cfg(), b_legacy)
    assert abs(s_legacy - float(noisy.pow(2).mean() + 1e-12)) < 1e-12