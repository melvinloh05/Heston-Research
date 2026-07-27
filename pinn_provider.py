"""PINN (price, Greeks) provider for the hedging engine + arm/provider registry.

Mirrors providers.HestonCFProvider's GreekProvider contract: evaluate(S, v, tau, K)
returns {price, delta, gamma, vega, vanna} broadcast to S.shape, finite for all
v >= 0 (QE paths hit v = 0 exactly). Unlike HestonCFProvider (analytic theta_train
Heston Greeks), this provider's Greeks are AUTODIFF of a TRAINED SobolevPINN price
head loaded from best.pt: Delta = dV/dS, Gamma = d2V/dS2, Vega = dV/dv0,
vanna = d2V/dSdv0 (oracle convention; sigma-vega is 2*sqrt(v0)*dV/dv0 downstream).

FROZEN train-regime parameters: the network's Option-A input vector is
[S, K, tau, kappa, theta, xi, rho, v0]. At hedging time the four Heston parameter
dims (kappa, theta, xi, rho) are PINNED to theta_train (the baseline regime every
arm is evaluated under); only S and v0 vary along the path. Per-point
v0 := max(v, v_floor) is the pathwise state, so v in [0, 0.01) is MILD extrapolation
below the training range — left UN-CLAMPED by design (MLP.forward normalizes affinely
with no clamp, so OOD points extrapolate linearly past the edge; covered by the
no-clamp v=0 test).

eval_dtype: double-backward Gamma in float32 carries a noise floor that can rival the
small-Gamma wings, so the WHOLE model is cast to eval_dtype (default float64) at load.
At ~10k path states the float64 cost is negligible and the Greeks are clean.

Determinism: the model has no stochastic modules (asserted at load), runs in eval, and
greeks_eval is pure forward + autodiff, so identical inputs give bit-equal outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from oracle import HestonParams
from providers import HestonCFProvider
from SobolevPINN import PINNConfig
from ude import build_model

# the four Heston-parameter input dims pinned to theta_train at hedging time
_FROZEN_PARAMS = ("kappa", "theta", "xi", "rho")

# engine method name -> checkpoint arm directory (identity for names not listed:
# feedforward, sigma_*, shuffled, bs_gamma, gradient_penalty_only, info_matched_baseline)
_ARM_DIR = {
    "standard_pinn": "standard_pinn",
    "rung1": "rung1_delta",
    "rung2": "rung2_delta_gamma",
    "rung3": "rung3_delta_gamma_vega",
    "sans_pde": "sobolev_sans_pde",
}

# mode-dependent / stochastic modules that would break eval reproducibility
_STOCHASTIC = (torch.nn.modules.dropout._DropoutNd,
               torch.nn.modules.batchnorm._BatchNorm,
               torch.nn.modules.instancenorm._InstanceNorm)


def _cfg_from_dict(d: dict) -> PINNConfig:
    """Rebuild a PINNConfig from a checkpoint's asdict cfg. Tuples survive the
    torch.save pickle, but coerce inputs / input_ranges defensively (exactly as
    SobolevPINN.load_arm does) so a json-normalized cfg round-trips too."""
    d = dict(d)
    d["inputs"] = tuple(d["inputs"])
    if "input_ranges" in d:
        d["input_ranges"] = {k: tuple(v) for k, v in d["input_ranges"].items()}
    return PINNConfig(**d)


class PINNProvider:
    """GreekProvider backed by a trained SobolevPINN checkpoint (best.pt).

    The checkpoint carries the arm's PINNConfig (asdict) and state_dict; the model is
    rebuilt, loaded, cast to eval_dtype, and frozen in eval mode. evaluate feeds the
    hedging engine's path states through the network with the Heston parameter dims
    pinned to `train_regime` (theta_train) and returns detached numpy float64 Greeks.
    """

    def __init__(self, checkpoint_path, train_regime: dict, r: float, q: float,
                 eval_dtype: str = "float64", v_floor: float = 1e-6,
                 chunk: int | None = None, device: str = "cpu") -> None:
        missing = [k for k in _FROZEN_PARAMS if k not in train_regime]
        assert not missing, f"train_regime missing frozen params {missing}"

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.cfg: PINNConfig = _cfg_from_dict(ckpt["cfg"])
        self.eval_dtype = (getattr(torch, eval_dtype) if isinstance(eval_dtype, str)
                           else eval_dtype)
        self.device = torch.device(device)

        model = build_model(self.cfg)                 # built in cfg.dtype (matches state_dict);
        model.load_state_dict(ckpt["state_dict"])     # UDE arm carries correction.* keys too
        model.to(self.eval_dtype).to(self.device).eval()
        stochastic = sorted({type(m).__name__ for m in model.modules()
                             if isinstance(m, _STOCHASTIC)})
        assert not stochastic, (f"checkpoint model has stochastic modules {stochastic}; "
                                "eval is not deterministic")
        self.model = model

        self.train_regime = {k: float(train_regime[k]) for k in _FROZEN_PARAMS}
        self.r = float(r)
        self.q = float(q)
        self.v_floor = float(v_floor)
        self.chunk = chunk
        self.inputs = self.cfg.inputs

    def _build_x(self, S, v, tau: float, K: float) -> tuple[torch.Tensor, tuple]:
        """Assemble the network input matrix (n, len(inputs)) in cfg.inputs order.

        S/v0 are the pathwise state (v0 := max(v, v_floor)); K and tau are scalar
        broadcasts; the four Heston parameter dims are frozen at train_regime. Returns
        (x tensor in eval_dtype on device, output shape) so evaluate can reshape."""
        S, v = np.broadcast_arrays(np.asarray(S, float), np.asarray(v, float))
        shape = S.shape
        n = S.size
        vfloored = np.maximum(v.ravel(), self.v_floor)
        col = {"S": S.ravel(), "K": np.full(n, float(K)), "tau": np.full(n, float(tau))}
        for p in _FROZEN_PARAMS:
            col[p] = np.full(n, self.train_regime[p])
        cols = []
        for name in self.inputs:
            if name in ("v0", "v"):          # pathwise spot variance (Option A: v0)
                cols.append(vfloored)
            elif name in col:
                cols.append(col[name])
            else:
                raise KeyError(f"PINNProvider cannot build input column {name!r} "
                               f"(inputs={self.inputs})")
        x = np.stack(cols, axis=1)
        return torch.as_tensor(x, dtype=self.eval_dtype, device=self.device), shape

    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float, K: float) -> dict:
        """(price, delta, gamma, vega, vanna) at path states (S, max(v, v_floor), tau, K),
        each broadcast to S.shape as numpy float64. Greeks via double-backward autodiff
        (greeks_eval handles enable_grad); price via a separate no-grad forward."""
        x, shape = self._build_x(S, v, tau, K)
        chunk = self.chunk or self.cfg.second_order_microbatch
        g = self.model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"),
                                   chunk=chunk or None)
        with torch.no_grad():
            price = self.model(x)
        out = {"price": price, "delta": g["delta"], "gamma": g["gamma"],
               "vega": g["vega"], "vanna": g["vanna"]}
        return {k: t.detach().to("cpu", torch.float64).numpy().reshape(shape)
                for k, t in out.items()}


def build_providers(cfg, ckpt_root, arms: list[str], seed: int, r: float, q: float,
                    include_oracle: bool = True) -> dict:
    """Map engine method names to providers for a hedging cell.

    `cfg` is the benchmark contract (heston_benchmark_v6.yaml, dict or path); the
    theta_train / train_regime parameters come from cfg.regimes.baseline. Returns
    {"oracle": HestonCFProvider(theta_train), "<engine name>": PINNProvider(...)} with
    each PINN checkpoint read from ckpt_root/<arm dir>/s<seed>/best.pt (engine name ->
    arm dir via _ARM_DIR; identity otherwise). A missing checkpoint raises loudly,
    listing every best.pt that DOES exist under ckpt_root.
    """
    if isinstance(cfg, (str, Path)):
        import yaml
        cfg = yaml.safe_load(Path(cfg).read_text())
    baseline = cfg["regimes"]["baseline"]
    train_regime = {k: float(baseline[k]) for k in _FROZEN_PARAMS}
    ckpt_root = Path(ckpt_root)

    providers: dict = {}
    if include_oracle:
        params = HestonParams(**{k: float(baseline[k])
                                 for k in ("kappa", "theta", "xi", "rho", "v0")})
        providers["oracle"] = HestonCFProvider(params, r, q)

    for name in arms:
        arm_dir = _ARM_DIR.get(name, name)
        path = ckpt_root / arm_dir / f"s{seed}" / "best.pt"
        if not path.exists():
            available = (sorted(str(p.relative_to(ckpt_root))
                                for p in ckpt_root.glob("*/s*/best.pt"))
                         if ckpt_root.exists() else [])
            raise FileNotFoundError(
                f"no checkpoint for arm {name!r} (dir {arm_dir!r}, seed {seed}) at {path}. "
                f"Available best.pt under {ckpt_root}: {available or '<none>'}")
        providers[name] = PINNProvider(path, train_regime, r, q)

    return providers
