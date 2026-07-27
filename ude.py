"""UDE arm: a learned residual correction to the Heston variance drift (contract family "ude").

ONE arm, partial physics. The Heston PDE is retained in full; the ONLY structural change is
inside pde_residual, where the variance drift

    kappa*(theta - v)  ->  kappa*(theta - v) + g_phi(v)

gains a small residual-correction net g_phi (1-16-16-1, tanh). The price head, the Greek
autodiff paths, the supervision terms and every loss scale are UNCHANGED and byte-for-byte
identical to the other arms — the extra parameters live in the correction net alone and are
counted separately (param_counts). g_phi is ZERO-INITIALIZED (last layer weight+bias = 0), so
at construction the residual reproduces the standard SobolevPINN residual bit-for-bit.

build_model(cfg) is the single factory: it returns UDESobolevPINN when cfg.ude_correction is
set and a plain SobolevPINN otherwise, so training and eval route the right class off flags.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from SobolevPINN import PINNConfig, SobolevPINN

_CORRECTION_WIDTH = 16   # g_phi hidden width; architecture 1-16-16-1, tanh (contract: family "ude")


class UDECorrection(nn.Module):
    """g_phi(v): 1-16-16-1 tanh residual on the (spot-variance) drift, zero at init.

    v is affinely mapped to [-1,1] on the contract v0 range before the net (same convention
    as MLP.forward) so the net sees well-conditioned inputs; the final Linear is zero-init so
    g_phi(v) == 0 exactly for all v at construction — the UDE residual then equals the standard
    Heston residual bit-for-bit until training moves the weights.
    """

    def __init__(self, cfg: PINNConfig) -> None:
        super().__init__()
        w = _CORRECTION_WIDTH
        self.net = nn.Sequential(
            nn.Linear(1, w), nn.Tanh(),
            nn.Linear(w, w), nn.Tanh(),
            nn.Linear(w, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        dt = getattr(torch, cfg.dtype)
        v_lo, v_hi = cfg.input_ranges["v0" if "v0" in cfg.input_ranges else "v"]
        self.register_buffer("v_lo", torch.tensor(float(v_lo), dtype=dt))
        self.register_buffer("v_scale", torch.tensor(2.0 / (float(v_hi) - float(v_lo)), dtype=dt))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        vn = (v - self.v_lo) * self.v_scale - 1.0        # affine to [-1,1]; NO clamp (OOD extrapolates)
        return self.net(vn.unsqueeze(-1)).squeeze(-1)


class UDESobolevPINN(SobolevPINN):
    """SobolevPINN with a learned residual on the variance drift, active ONLY in pde_residual.

    Everything the base class exposes (price head net, greeks, greeks_eval, loss, loss scales)
    is inherited unchanged; the subclass adds one module (self.correction) and overrides the
    single drift hook. The price head is built first (identical RNG draws -> bit-identical
    net.* / loss_scale_* buffers vs any other arm); the correction net draws come strictly
    after and never perturb the price head.
    """

    def __init__(self, cfg: PINNConfig) -> None:
        super().__init__(cfg)
        self.correction = UDECorrection(cfg)
        self.to(getattr(torch, cfg.dtype))

    def _variance_drift(self, v: torch.Tensor, kappa: torch.Tensor,
                        theta: torch.Tensor) -> torch.Tensor:
        # kappa*(theta - v) + g_phi(v); g_phi=0 at init -> identical to base drift bit-for-bit.
        return super()._variance_drift(v, kappa, theta) + self.correction(v)


def build_model(cfg: PINNConfig) -> SobolevPINN:
    """Factory: route the UDE arm to UDESobolevPINN, every other arm to plain SobolevPINN."""
    if getattr(cfg, "ude_correction", False):
        return UDESobolevPINN(cfg)
    return SobolevPINN(cfg)


def param_counts(model: SobolevPINN) -> dict[str, int]:
    """Split trainable params into price-head vs correction-net, for the 'logged separately'
    invariant: the price head must match a non-UDE arm; the extra params live in .correction."""
    correction = 0
    if isinstance(model, UDESobolevPINN):
        correction = int(sum(p.numel() for p in model.correction.parameters()))
    total = int(sum(p.numel() for p in model.parameters()))
    return {"price_head": total - correction, "correction": correction, "total": total}
