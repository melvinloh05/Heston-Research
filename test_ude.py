"""Tests for the UDE arm: learned residual on the variance drift (contract family "ude").

The load-bearing invariants: (a) the arm loads as UDESobolevPINN off flags only; (b) the price
head is byte-for-byte identical to a non-UDE arm (extra params live in .correction, counted
separately); (c) the loss is finite; (d) with g_phi zero-initialized the PDE residual reproduces
the standard SobolevPINN residual bit-for-bit — the correction is a pure, initially-inert add-on.
"""
from __future__ import annotations

import torch

from SobolevPINN import PINNConfig, SobolevPINN, load_arm
from train_pinn import set_seed
from ude import UDECorrection, UDESobolevPINN, build_model, param_counts

CFG = "pinn_config.yaml"
RANGES = {"S": (50.0, 150.0), "K": (60.0, 140.0), "tau": (0.04, 1.0), "kappa": (1.0, 4.0),
          "theta": (0.02, 0.12), "xi": (0.20, 0.60), "rho": (-0.80, -0.20), "v0": (0.01, 0.12)}


def make_batch(cfg: PINNConfig, n: int = 48, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.stack([torch.empty(n).uniform_(*RANGES[k], generator=gen) for k in cfg.inputs], 1)
    i = {name: k for k, name in enumerate(cfg.inputs)}
    S, K, v = x[:, i["S"]], x[:, i["K"]], x[:, i["v0"]]
    price = (S - K).clamp(min=0.0) + 5.0 * v * S / 100.0
    return {"x": x, "price": price}


def test_ude_arm_loads_as_correction_subclass_off_flags():
    cfg = load_arm(CFG, "ude")
    assert cfg.ude_correction and cfg.use_pde
    assert not (cfg.supervise_delta or cfg.supervise_gamma or cfg.supervise_vega)
    model = build_model(cfg)
    assert isinstance(model, UDESobolevPINN)
    assert hasattr(model, "correction") and isinstance(model.correction, UDECorrection)


def test_build_model_routes_only_ude_to_correction_class():
    assert type(build_model(load_arm(CFG, "ude"))) is UDESobolevPINN
    for arm in ("rung0_price_only", "rung3_delta_gamma_vega", "standard_pinn", "feedforward"):
        assert type(build_model(load_arm(CFG, arm))) is SobolevPINN, arm


def test_correction_net_architecture_is_1_16_16_1_tanh():
    lins = [m for m in build_model(load_arm(CFG, "ude")).correction.net if isinstance(m, torch.nn.Linear)]
    tanhs = [m for m in build_model(load_arm(CFG, "ude")).correction.net if isinstance(m, torch.nn.Tanh)]
    assert [(l.in_features, l.out_features) for l in lins] == [(1, 16), (16, 16), (16, 1)]
    assert len(tanhs) == 2


def test_price_head_params_identical_extra_params_isolated_to_correction():
    set_seed(42)
    base = SobolevPINN(load_arm(CFG, "rung0_price_only"))
    set_seed(42)
    ude = build_model(load_arm(CFG, "ude"))
    # every price-head key bit-identical (same seed; correction draws come strictly after)
    bsd = base.state_dict()
    usd = {k: v for k, v in ude.state_dict().items() if not k.startswith("correction.")}
    assert usd.keys() == bsd.keys()
    for k in bsd:
        assert torch.equal(usd[k], bsd[k]), k
    # the extra parameters live ONLY in the correction net, and it is nonempty
    pc = param_counts(ude)
    assert pc["price_head"] == sum(p.numel() for p in base.parameters())
    assert pc["correction"] > 0 and pc["total"] == pc["price_head"] + pc["correction"]
    assert param_counts(base)["correction"] == 0


def test_g_phi_is_zero_at_init_and_residual_matches_standard_bit_for_bit():
    cfg = load_arm(CFG, "ude")
    set_seed(42)
    base = SobolevPINN(cfg)          # base class ignores ude_correction -> standard drift
    set_seed(42)
    ude = build_model(cfg)           # identical price head + a zero-init correction net
    batch = make_batch(cfg)
    v = batch["x"][:, ude.i_v]
    assert torch.count_nonzero(ude.correction(v)) == 0          # g_phi(v) == 0 for all v at init
    x = batch["x"].clone().requires_grad_(True)
    xb = batch["x"].clone().requires_grad_(True)
    assert torch.equal(ude.pde_residual(x), base.pde_residual(xb))   # bit-for-bit


def test_correction_actually_enters_the_residual_when_nonzero():
    cfg = load_arm(CFG, "ude")
    set_seed(42)
    base = SobolevPINN(cfg)
    set_seed(42)
    ude = build_model(cfg)
    with torch.no_grad():            # break the zero init so g_phi(v) != 0
        ude.correction.net[-1].weight.fill_(0.1)
        ude.correction.net[-1].bias.fill_(0.05)
    x = make_batch(cfg)["x"].clone().requires_grad_(True)
    xb = x.detach().clone().requires_grad_(True)
    assert not torch.allclose(ude.pde_residual(x), base.pde_residual(xb))


def test_ude_loss_is_finite_and_backpropagates_into_correction():
    cfg = load_arm(CFG, "ude")
    model = build_model(cfg)
    # a nonzero correction so gradients can flow to it from the (initially inert) g_phi
    with torch.no_grad():
        model.correction.net[-1].weight.fill_(0.1)
    batch = make_batch(cfg)
    terms = model.loss(batch)
    assert torch.isfinite(terms["total"])
    assert "pde" in terms                        # PDE retained (partial physics)
    terms["total"].backward()
    grads = [p.grad for p in model.correction.parameters() if p.grad is not None]
    assert grads and any(torch.any(g != 0) for g in grads)
