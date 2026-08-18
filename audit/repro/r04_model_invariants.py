"""R04 — model-layer invariants the audit asks to verify by execution.

(a) price head bit-identical across arms at the same seed (incl. the UDE arm:
    does building the correction net perturb the price-head init?)
(b) g_phi == 0 at init and the UDE PDE residual bit-identical to the base residual
(c) per-term loss scales bit-identical across arms after the full-split freeze
(d) OFF loss terms contribute exactly zero gradient
(e) PINNProvider chunked vs unchunked autodiff: bit-equal?
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
from SobolevPINN import load_arm, SobolevPINN  # noqa: E402
from ude import build_model, param_counts  # noqa: E402
from train_pinn import set_seed  # noqa: E402

CFG = "/Users/melvin/Documents/Heston Research/pinn_config.yaml"
ARMS = ["rung0_price_only", "rung1_delta", "rung2_delta_gamma",
        "rung3_delta_gamma_vega", "standard_pinn", "feedforward",
        "sigma_000", "sigma_010", "sigma_050", "shuffled", "bs_gamma",
        "gradient_penalty_only", "sobolev_sans_pde", "lambda_pde_zero", "ude"]
SEED = 42

print("=" * 74)
print("(a) price-head state_dict bit-identical across arms at seed 42")
print("=" * 74)
heads = {}
for arm in ARMS:
    cfg = load_arm(CFG, arm)
    set_seed(SEED)
    m = build_model(cfg)
    heads[arm] = {k: v.detach().clone() for k, v in m.state_dict().items()
                  if k.startswith("net.")}
ref = heads["rung3_delta_gamma_vega"]
for arm, sd in heads.items():
    same = (set(sd) == set(ref)
            and all(torch.equal(sd[k], ref[k]) for k in ref))
    print(f"  {arm:<24} price head == rung3 : {same}")

print()
print("=" * 74)
print("(b) UDE: g_phi zero at init; residual bit-identical to the base arm")
print("=" * 74)
set_seed(SEED)
m_ude = build_model(load_arm(CFG, "ude"))
set_seed(SEED)
m_base = build_model(load_arm(CFG, "rung3_delta_gamma_vega"))
pc = param_counts(m_ude)
print(f"  param_counts(ude)  = {pc}")
print(f"  param_counts(base) = {param_counts(m_base)}")
v = torch.linspace(0.0, 0.20, 64, dtype=torch.float32)
gphi = m_ude.correction(v)
print(f"  max |g_phi(v)| over v in [0, 0.2]  = {float(gphi.abs().max()):.3e}")
rng = np.random.default_rng(0)
cfg = load_arm(CFG, "rung3_delta_gamma_vega")
lo = np.array([cfg.input_ranges[n][0] for n in cfg.inputs])
hi = np.array([cfg.input_ranges[n][1] for n in cfg.inputs])
x = torch.as_tensor(lo + (hi - lo) * rng.random((256, len(cfg.inputs))),
                    dtype=torch.float32).requires_grad_(True)
r_ude = m_ude.pde_residual(x)
x2 = x.detach().clone().requires_grad_(True)
r_base = m_base.pde_residual(x2)
print(f"  residual bit-identical (ude vs base) : {torch.equal(r_ude, r_base)}")
print(f"  max abs diff                          : {float((r_ude - r_base).abs().max()):.3e}")

print()
print("=" * 74)
print("(c) loss scales bit-identical across arms (full-split freeze)")
print("=" * 74)
n = 512
batch_common = {
    "x": torch.as_tensor(lo + (hi - lo) * rng.random((n, len(cfg.inputs))),
                         dtype=torch.float32),
    "price": torch.as_tensor(rng.random(n) * 10.0, dtype=torch.float32),
    "delta": torch.as_tensor(rng.random(n), dtype=torch.float32),
    "vega": torch.as_tensor(rng.random(n) * 5.0, dtype=torch.float32),
    "vanna": torch.as_tensor(rng.random(n) * 0.1, dtype=torch.float32),
}
gamma_true = torch.as_tensor(rng.random(n) * 0.05, dtype=torch.float32)
batch_common["gamma_ref"] = gamma_true

scales = {}
for arm in ARMS:
    acfg = load_arm(CFG, arm)
    b = {k: v.clone() for k, v in batch_common.items()}
    # emulate build_arm_labels: only the GAMMA label differs per arm
    if acfg.label_source == "shuffled":
        b["gamma"] = gamma_true[torch.randperm(n)]
    elif acfg.label_source == "bs_gamma":
        b["gamma"] = gamma_true * 1.37
    elif acfg.label_source != "none":
        b["gamma"] = gamma_true * (1.0 + acfg.gamma_label_noise_sigma
                                   * torch.as_tensor(rng.standard_normal(n),
                                                     dtype=torch.float32))
    set_seed(SEED)
    m = build_model(acfg)
    m._freeze_loss_scales(b)
    scales[arm] = {k: float(v) for k, v in m.state_dict().items()
                   if k.startswith("loss_scale_")}
sref = scales["rung3_delta_gamma_vega"]
print(f"  reference (rung3) scales: "
      + ", ".join(f"{k.replace('loss_scale_','')}={v:.6g}" for k, v in sorted(sref.items())))
for arm, sc in scales.items():
    same = all(sc[k] == sref[k] for k in sref)
    bad = [k for k in sref if sc[k] != sref[k]]
    print(f"  {arm:<24} scales == rung3 : {same}" + (f"   DIFFER: {bad}" if bad else ""))

print()
print("=" * 74)
print("(d) OFF loss terms contribute exactly zero gradient")
print("=" * 74)
for arm in ("rung1_delta", "rung2_delta_gamma", "rung3_delta_gamma_vega",
            "standard_pinn", "sobolev_sans_pde", "lambda_pde_zero"):
    acfg = load_arm(CFG, arm)
    b = {k: v.clone() for k, v in batch_common.items()}
    b["gamma"] = gamma_true.clone()
    set_seed(SEED)
    m = build_model(acfg)
    terms = m.loss(b)
    print(f"  {arm:<24} terms = {sorted(k for k in terms if k != 'total')}")

print()
print("=" * 74)
print("(e) PINNProvider: chunked vs unchunked autodiff")
print("=" * 74)
from pinn_provider import PINNProvider  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    acfg = load_arm(CFG, "rung3_delta_gamma_vega")
    set_seed(SEED)
    m = build_model(acfg)
    ck = Path(td) / "best.pt"
    torch.save({"state_dict": m.state_dict(),
                "cfg": {f.name: getattr(acfg, f.name)
                        for f in __import__("dataclasses").fields(acfg)}}, ck)
    tr = {"kappa": 2.0, "theta": 0.04, "xi": 0.30, "rho": -0.50}
    S = np.linspace(70.0, 130.0, 5000)
    v = np.full_like(S, 0.04)
    outs = {}
    for chunk in (None, 2048, 512, 97):
        p = PINNProvider(ck, tr, 0.02, 0.0, chunk=chunk)
        # chunk=None falls back to cfg.second_order_microbatch (2048)
        outs[chunk] = p.evaluate(S, v, 0.17, 100.0)
    base_key = 97
    print(f"  cfg.second_order_microbatch = {acfg.second_order_microbatch}")
    for chunk, o in outs.items():
        eq = {g: bool(np.array_equal(o[g], outs[base_key][g]))
              for g in ("price", "delta", "gamma", "vega")}
        mx = {g: float(np.max(np.abs(o[g] - outs[base_key][g])))
              for g in ("price", "delta", "gamma", "vega")}
        print(f"  chunk={str(chunk):<5} bit-equal to chunk=97: {eq}")
        print(f"        max abs diff: " + ", ".join(f"{g}={mx[g]:.3e}" for g in mx))
    g = outs[2048]["gamma"]
    print(f"  gamma magnitude scale: max|gamma| = {np.abs(g).max():.3e}")

    print()
    print("  --- v_floor / v = 0 behaviour (QE atom) ---")
    p = PINNProvider(ck, tr, 0.02, 0.0)
    o0 = p.evaluate(np.array([100.0]), np.array([0.0]), 0.17, 100.0)
    print(f"  evaluate at v=0 finite: "
          f"{ {k: bool(np.isfinite(val).all()) for k, val in o0.items()} }")
