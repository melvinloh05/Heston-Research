"""Sobolev-PINN model class for the Heston benchmark (contract: heston_benchmark_v6.yaml).

ONE network class; every method and control arm (ladder rungs 0-3, standard_pinn,
feedforward, info-matched baseline, A6 dose-response arms, lambda_pde=0) is a PINNConfig
instance. Loss terms toggle purely by config flags — no per-method code paths.

  L = L_price + l_pde*L_pde + l_bc*L_bc + l_D*L_D + l_G*L_G + l_v*L_v [+ l_vanna*L_vanna]

Greeks come from autodiff of the price head: Delta = dV/dS, Gamma = d2V/dS2,
Vega = dV/dv (oracle convention dV/dv0; sigma-vega = 2*sqrt(v0)*dV/dv0),
vanna = d2V/dSdv (DIAGNOSTIC; a loss term only in the optional vanna arm).

Input normalization (contract ranges -> [-1,1]) and output scaling (x K_ref) live
INSIDE MLP.forward, so every autodiff path (Greeks, PDE residual, eval) differentiates
with respect to RAW coordinates and labels stay raw — no Jacobian factor appears in
any loss term or eval path.

Loss-term scale normalization (cfg.loss_scale_mode, default "label_second_moment"):
each supervised MSE term is divided by a per-term scale mean(label^2)+eps frozen from
the FIRST training batch, making every term dimensionless and O(1)-comparable. This is
load-bearing: raw-unit MSEs at EQUAL 5% relative error (baseline regime, CF oracle) are
price ~2.1e0, vega ~1.1e0, delta ~1.1e-3, gamma ~7.5e-7 — an unnormalized gamma term
sits ~6 orders below price/vega, is numerically inert, and silently nulls the
rung1->rung2 comparison. Scales are flag-independent, so arms built from the same
training data carry bit-identical buffers (arms differ ONLY by flags). "raw" restores
the legacy unnormalized loss exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "softplus": nn.Softplus}
GAMMA_LABEL_SOURCES = ("oracle", "bs_gamma", "shuffled", "none")
LOSS_SCALE_MODES = ("raw", "label_second_moment")
SCALE_EPS = 1e-12   # keeps scales strictly positive; negligible vs any real mean(label^2)
# loss term -> batch label key its scale is computed from ("pde" is handled separately)
_SCALE_LABEL_KEYS = {"price": "price", "delta": "delta", "gamma": "gamma",
                     "vega": "vega", "vanna": "vanna", "bc": "price_bc"}

# CONTRACT normalization ranges (heston_benchmark_v6.yaml) — never batch statistics:
# grid for S/K/tau, training hypercube for kappa/theta/xi/rho/v0. Shared-block YAML
# (pinn_config.yaml) carries the same values so buffers are bit-identical across arms.
DEFAULT_INPUT_RANGES: dict[str, tuple[float, float]] = {
    "S": (50.0, 150.0), "K": (60.0, 140.0), "tau": (0.04, 1.0),
    "kappa": (1.0, 4.0), "theta": (0.02, 0.12), "xi": (0.20, 0.60),
    "rho": (-0.80, -0.20), "v0": (0.01, 0.12),
    "v": (0.01, 0.12),   # option_B spot-variance input (retained-but-unused) = v0 range
}

_GRAD_CALLS = {"n": 0}  # module-wide torch.autograd.grad invocation counter (compute accounting)


@dataclass
class PINNConfig:
    """One arm = one instance; architecture identical across arms, only flags differ.

    inputs: option_A parametric hypercube [S,K,tau,kappa,theta,xi,rho,v0] — LOCKED in v6
    (Feller-constrained sampling lives in train_pinn.sample_hypercube_params). The
    option_B per-regime [S,K,tau,v] + pde_params machinery is retained but unused under v6.
    """
    inputs: tuple[str, ...] = ("S", "K", "tau", "kappa", "theta", "xi", "rho", "v0")
    input_ranges: dict = field(default_factory=lambda: dict(DEFAULT_INPUT_RANGES))
    n_layers: int = 4
    width: int = 64
    activation: str = "tanh"
    width_mult: float = 1.0            # 2x-width capacity control (A10)
    dtype: str = "float32"             # float64 available for second-derivative care
    use_pde: bool = True
    use_bc: bool = True
    supervise_delta: bool = False
    supervise_gamma: bool = False
    supervise_vega: bool = False
    supervise_vanna: bool = False      # optional vanna arm ONLY
    label_source: str = "oracle"       # GAMMA labels only: oracle | bs_gamma | shuffled | none
    gamma_label_noise_sigma: float = 0.0   # multiplicative: gamma*(1+N(0,sigma)) per v6
    gradient_penalty: bool = False     # curvature-L2 on gamma-hat, NO labels
    ude_correction: bool = False       # UDE arm ONLY: learned residual on the variance
                                       # drift INSIDE pde_residual (ude.py); price head +
                                       # supervision untouched (partial-physics family "ude")
    n_price_points: int = 4096
    lambda_pde: float = 1.0            # sweep includes 0.0 = twin-network/Sakuma config (A2)
    lambda_bc: float = 1.0
    lambda_delta: float = 1.0
    lambda_gamma: float = 1.0
    lambda_vega: float = 1.0
    lambda_vanna: float = 0.0
    loss_scale_mode: str = "label_second_moment"   # "raw" | "label_second_moment":
                                                   # per-term MSE / (mean(label^2)+eps)
    r: float = 0.02
    q: float = 0.0
    pde_params: dict = field(default_factory=dict)   # option_B: coefficients not in inputs
    second_order_microbatch: int = 0   # >0: chunk eval-time 2nd-order autodiff (memory)

    def __post_init__(self) -> None:
        assert {"S", "tau"} <= set(self.inputs) and ({"v0", "v"} & set(self.inputs))
        for n in self.inputs:   # every input needs a normalization range from the contract
            assert n in self.input_ranges, f"input_ranges missing entry for {n!r}"
            lo, hi = self.input_ranges[n]
            assert lo < hi, f"degenerate input_range for {n!r}: ({lo}, {hi})"
        assert self.label_source in GAMMA_LABEL_SOURCES
        assert self.loss_scale_mode in LOSS_SCALE_MODES
        if self.gradient_penalty:
            assert not self.supervise_gamma and self.label_source == "none"
        if self.supervise_gamma:
            assert self.label_source != "none"
        if self.gamma_label_noise_sigma > 0.0:
            assert self.label_source == "oracle"


def load_arm(cfg_path: str, arm: str) -> PINNConfig:
    """Build one arm's PINNConfig from pinn_config.yaml: shared block + per-arm flag overrides."""
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    d = {**raw["shared"], **(raw["arms"][arm] or {})}
    d["inputs"] = tuple(d["inputs"])
    if "input_ranges" in d:   # shared-block only -> bit-identical buffers across arms
        d["input_ranges"] = {k: tuple(v) for k, v in d["input_ranges"].items()}
    return PINNConfig(**d)


class MLP(nn.Module):
    """Plain feed-forward price head; architecture from config, IDENTICAL across arms.

    Normalization and output scaling live INSIDE forward so every caller (Greek
    autodiff, PDE residual, hedging eval) differentiates in RAW coordinates: inputs
    map affinely to [-1,1] (tanh-symmetric) on the CONTRACT ranges — never batch
    statistics, never clamped, so OOD points (strong_neg_corr rho=-0.9) extrapolate
    linearly past the edge — and the output is scaled by K_ref (midpoint of the K
    range). Autodiff carries both factors into every Greek and PDE term automatically;
    price and Greek labels stay in raw units. The constants are registered buffers:
    they follow .to(dtype/device) and serialize in state_dict.
    """

    def __init__(self, cfg: PINNConfig) -> None:
        super().__init__()
        w, act = int(round(cfg.width * cfg.width_mult)), _ACT[cfg.activation]
        layers: list[nn.Module] = []
        d = len(cfg.inputs)
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, w), act()]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        dt = getattr(torch, cfg.dtype)
        lo = torch.tensor([cfg.input_ranges[n][0] for n in cfg.inputs], dtype=dt)
        hi = torch.tensor([cfg.input_ranges[n][1] for n in cfg.inputs], dtype=dt)
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_scale", 2.0 / (hi - lo))
        k_lo, k_hi = cfg.input_ranges["K"]
        self.register_buffer("k_ref", torch.tensor(0.5 * (k_lo + k_hi), dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.in_lo) * self.in_scale - 1.0   # affine only — NO clamp (OOD extrapolates)
        return self.k_ref * self.net(xn).squeeze(-1)


def autodiff_greeks(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                    i_s: int, i_v: int,
                    need: tuple[str, ...] = ("delta", "gamma", "vega")) -> dict[str, torch.Tensor]:
    """Greeks of scalar-output fn(x) by autodiff; x must have requires_grad.

    delta = dV/dS, gamma = d2V/dS2, vega = dV/dv, vanna = d2V/dSdv. Second-order
    quantities use double backward (create_graph on the first pass).
    """
    out = {"price": fn(x)}
    order2 = ("gamma" in need) or ("vanna" in need)
    if not (order2 or {"delta", "vega"} & set(need)):
        return out
    g = torch.autograd.grad(out["price"].sum(), x, create_graph=True)[0]
    _GRAD_CALLS["n"] += 1
    if "delta" in need or order2:
        out["delta"] = g[:, i_s]
    if "vega" in need:
        out["vega"] = g[:, i_v]
    if order2:
        g2 = torch.autograd.grad(g[:, i_s].sum(), x, create_graph=True)[0]
        _GRAD_CALLS["n"] += 1
        if "gamma" in need:
            out["gamma"] = g2[:, i_s]
        if "vanna" in need:
            out["vanna"] = g2[:, i_v]
    return out


class SobolevPINN(nn.Module):
    """Price head + flag-assembled Sobolev loss; the single class behind every arm."""

    def __init__(self, cfg: PINNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = MLP(cfg)
        self.to(getattr(torch, cfg.dtype))
        self.i = {n: k for k, n in enumerate(cfg.inputs)}
        self.i_s = self.i["S"]
        self.i_v = self.i["v0"] if "v0" in self.i else self.i["v"]
        # Per-term loss scales: ones at construction (state_dict identical across arms),
        # frozen ONCE from the first training batch when loss_scale_mode demands it.
        dt = getattr(torch, cfg.dtype)
        for term in (*_SCALE_LABEL_KEYS, "pde"):
            self.register_buffer(f"loss_scale_{term}", torch.ones((), dtype=dt))
        self.register_buffer("loss_scales_frozen", torch.tensor(False))

    @torch.no_grad()
    def _freeze_loss_scales(self, batch: dict[str, torch.Tensor]) -> None:
        """Freeze per-term scales mean(label^2)+eps from THE FIRST TRAINING BATCH.

        Flag-independent: every label key present in the batch gets a scale (absent
        keys keep 1.0), so arms built from the same training data hold bit-identical
        buffers — arms differ only by flags, never by normalization constants. The
        PDE scale (declared choice) is mean((r*price_label)^2)+eps: the rV discount
        term of the residual evaluated on the labels — a model-free magnitude
        reference in squared residual units. Buffers serialize in state_dict and
        never move after this call, so the loss stays stationary.
        """
        for term, key in _SCALE_LABEL_KEYS.items():
            if term == "gamma" and "gamma_ref" in batch:
                # A6 dose-response arms carry CORRUPTED gamma labels; the scale must
                # come from the TRUE consensus ("gamma_ref", emitted by
                # make_labels.build_arm_labels for EVERY arm incl. label_source none)
                # or the noisy arms train with a different effective lambda_gamma and
                # the gradient-penalty arm with none at all — a weight axis silently
                # confounding the correctness axis. Falls back to the arm's own labels
                # only when no reference is provided (legacy batches).
                self.loss_scale_gamma.copy_(batch["gamma_ref"].pow(2).mean() + SCALE_EPS)
            elif key in batch:
                getattr(self, f"loss_scale_{term}").copy_(batch[key].pow(2).mean() + SCALE_EPS)
        if "price" in batch:
            self.loss_scale_pde.copy_((self.cfg.r * batch["price"]).pow(2).mean() + SCALE_EPS)
        self.loss_scales_frozen.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def greeks(self, x: torch.Tensor,
               need: tuple[str, ...] = ("delta", "gamma", "vega")) -> dict[str, torch.Tensor]:
        """Differentiable Greeks of the price head (x must have requires_grad)."""
        return autodiff_greeks(self.net, x, self.i_s, self.i_v, need)

    @torch.enable_grad()
    def greeks_eval(self, x: torch.Tensor,
                    need: tuple[str, ...] = ("delta", "gamma", "vega"),
                    chunk: int | None = None) -> dict[str, torch.Tensor]:
        """Detached (price, Greeks) interface for the hedging engine.

        Chunks the batch (cfg.second_order_microbatch) to bound double-backward peak
        memory — dominant near the Feller boundary (feller_violating_volvol 0.44,
        near_feller 1.04) where dense low-v sampling meets steep d2V/dv2.
        """
        n = chunk or self.cfg.second_order_microbatch or x.shape[0]
        outs = [autodiff_greeks(self.net, xc.detach().clone().requires_grad_(True),
                                self.i_s, self.i_v, need) for xc in x.split(n)]
        return {k: torch.cat([o[k].detach() for o in outs]) for k in outs[0]}

    def _coef(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Per-point PDE coefficient: input column (option_A) or cfg.pde_params (option_B)."""
        if name in self.i:
            return x[:, self.i[name]]
        return torch.full_like(x[:, 0], float(self.cfg.pde_params[name]))

    def pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Heston PDE residual via autodiff; dV/dt = -dV/dtau (tau = time to maturity).

        Three autograd.grad calls (one first-order, two second-order double backwards);
        the second-order passes dominate peak memory — worst in the Feller-boundary
        regimes (feller_violating_volvol ratio 0.44, near_feller 1.04).
        """
        i_s, i_v, i_t = self.i_s, self.i_v, self.i["tau"]
        V = self.net(x)
        g = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        V_s, V_v, V_tau = g[:, i_s], g[:, i_v], g[:, i_t]
        g2s = torch.autograd.grad(V_s.sum(), x, create_graph=True)[0]
        V_ss, V_sv = g2s[:, i_s], g2s[:, i_v]
        V_vv = torch.autograd.grad(V_v.sum(), x, create_graph=True)[0][:, i_v]
        _GRAD_CALLS["n"] += 3
        S, v = x[:, i_s], x[:, i_v]
        kappa, theta, xi, rho = (self._coef(x, n) for n in ("kappa", "theta", "xi", "rho"))
        r, q = self.cfg.r, self.cfg.q
        return (-V_tau + 0.5 * v * S ** 2 * V_ss + rho * xi * v * S * V_sv
                + 0.5 * xi ** 2 * v * V_vv + (r - q) * S * V_s
                + self._variance_drift(v, kappa, theta) * V_v - r * V)

    def _variance_drift(self, v: torch.Tensor, kappa: torch.Tensor,
                        theta: torch.Tensor) -> torch.Tensor:
        """Heston variance drift kappa*(theta - v). Overridable hook: the UDE arm
        (ude.UDESobolevPINN) adds a learned residual g_phi(v) here — nowhere else in
        the residual — so the standard drift is recovered bit-for-bit when g_phi=0."""
        return kappa * (theta - v)

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Assemble L from cfg flags; returns per-term dict including 'total'.

        Under loss_scale_mode="label_second_moment" every term in the dict is already
        scale-normalized (dimensionless); lambdas weight the NORMALIZED terms, so sweep
        semantics are unchanged and lambda_pde=0.0 still skips the PDE term entirely.
        "raw" bypasses normalization bit-for-bit (legacy behavior).
        """
        cfg = self.cfg
        if cfg.loss_scale_mode == "label_second_moment" and not bool(self.loss_scales_frozen):
            self._freeze_loss_scales(batch)
        nrm = ((lambda mse, term: mse / getattr(self, f"loss_scale_{term}"))
               if cfg.loss_scale_mode == "label_second_moment"
               else (lambda mse, term: mse))
        x = batch["x"].clone().requires_grad_(True)
        need = ["price"]
        if cfg.supervise_delta:
            need.append("delta")
        if cfg.supervise_gamma or cfg.gradient_penalty:
            need.append("gamma")
        if cfg.supervise_vega:
            need.append("vega")
        if cfg.supervise_vanna:
            need.append("vanna")
        g = self.greeks(x, tuple(need))
        terms = {"price": nrm(F.mse_loss(g["price"], batch["price"]), "price")}
        total = terms["price"]
        if cfg.use_pde and cfg.lambda_pde != 0.0:   # lambda_pde=0.0 valid: term skipped, identically zero
            xp = batch.get("x_pde", batch["x"]).clone().requires_grad_(True)
            terms["pde"] = nrm(self.pde_residual(xp).pow(2).mean(), "pde")
            total = total + cfg.lambda_pde * terms["pde"]
        if cfg.use_bc and "x_bc" in batch:
            terms["bc"] = nrm(F.mse_loss(self.net(batch["x_bc"]), batch["price_bc"]), "bc")
            total = total + cfg.lambda_bc * terms["bc"]
        if cfg.supervise_delta:
            terms["delta"] = nrm(F.mse_loss(g["delta"], batch["delta"]), "delta")
            total = total + cfg.lambda_delta * terms["delta"]
        if cfg.supervise_gamma:   # batch["gamma"] pre-built per label_source (greek_labels.py)
            terms["gamma"] = nrm(F.mse_loss(g["gamma"], batch["gamma"]), "gamma")
            total = total + cfg.lambda_gamma * terms["gamma"]
        elif cfg.gradient_penalty:   # curvature-L2, NO labels (A6 pure-regularizer reference);
            # same gamma scale as the supervised term, so the A6 comparison stays in one unit
            terms["gamma_penalty"] = nrm(g["gamma"].pow(2).mean(), "gamma")
            total = total + cfg.lambda_gamma * terms["gamma_penalty"]
        if cfg.supervise_vega:
            terms["vega"] = nrm(F.mse_loss(g["vega"], batch["vega"]), "vega")
            total = total + cfg.lambda_vega * terms["vega"]
        if cfg.supervise_vanna:   # optional vanna arm ONLY; off in the base class defaults
            terms["vanna"] = nrm(F.mse_loss(g["vanna"], batch["vanna"]), "vanna")
            total = total + cfg.lambda_vanna * terms["vanna"]
        terms["total"] = total
        return terms