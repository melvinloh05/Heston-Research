"""Training loop, compute accounting, validation-only lambda selection, A10 sweep hooks.

Smoke scope: drives SobolevPINN for tests and short runs; full training is launched
from (pinn_config.yaml + contract + seed) with the same entry points.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import sys
import time
import warnings
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import yaml
from scipy.stats import qmc

import SobolevPINN as sp
from SobolevPINN import PINNConfig, SobolevPINN
from ude import build_model, param_counts


def set_seed(seed: int) -> None:
    """Seed torch/numpy/random AND force deterministic kernels for a reproducible run.

    Beyond the RNG seeds, enables torch deterministic algorithms (warn_only, so ops
    without a deterministic implementation degrade to a warning rather than raising)
    and pins cuDNN to deterministic/non-benchmark. On CUDA, sets CUBLAS_WORKSPACE_CONFIG
    when unset (required for deterministic cuBLAS GEMMs). These make two same-seed runs
    bit-identical on CPU (the config_hash / best.pt determinism guarantee).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def peak_rss_bytes() -> int:
    """Process peak RSS in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024


def train_arm(cfg: PINNConfig, batch: dict[str, torch.Tensor], steps: int, lr: float,
              seed: int) -> tuple[SobolevPINN, dict]:
    """Train one arm on one batch; returns (model, log with losses + compute accounting)."""
    set_seed(seed)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    grad0, t0 = sp._GRAD_CALLS["n"], time.perf_counter()
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        terms = model.loss(batch)
        terms["total"].backward()
        opt.step()
        losses.append(float(terms["total"].detach()))
    pc = param_counts(model)
    log = {"losses": losses, "seed": seed, "cfg": cfg,
           "compute": {"wall_clock_s": time.perf_counter() - t0,
                       "param_count": pc["total"],
                       "price_head_param_count": pc["price_head"],
                       "correction_param_count": pc["correction"],
                       "derivative_evals": sp._GRAD_CALLS["n"] - grad0,
                       "peak_memory_bytes": peak_rss_bytes()}}
    return model, log


class LockedTestSet:
    """Guard asserting test data is untouched during lambda selection; unlock() to use."""

    def __init__(self, data) -> None:
        self._data, self.locked = data, True

    def unlock(self):
        self.locked = False
        return self._data

    @property
    def data(self):
        if self.locked:
            raise RuntimeError("test set accessed during lambda selection")
        return self._data


def select_lambda_pde(candidates: Iterable[float], fit_and_val_score: Callable[[float], float],
                      test_set: LockedTestSet) -> dict:
    """Pick lambda_pde on VALIDATION ONLY (v6 lambda_discipline); logs selected value."""
    if not test_set.locked:
        raise RuntimeError("test set unlocked before lambda selection")
    scores = {float(lam): float(fit_and_val_score(float(lam))) for lam in candidates}
    if not test_set.locked:
        raise RuntimeError("test set touched during lambda selection")
    selected = min(scores, key=scores.get)
    return {"selected_lambda_pde": selected, "validation_scores": scores}


HESTON_PARAM_NAMES = ("kappa", "theta", "xi", "rho", "v0")


def anchors_from_contract(contract_path: str) -> list[dict]:
    """All named eval regimes from the contract as excision anchors (kappa..v0 only).

    Includes every regime under `regimes:` — anchors outside the sampling hypercube
    (strong_neg_corr, rho=-0.90) are harmless: only the intersection of their ball
    with the cube is excised.
    """
    regimes = yaml.safe_load(open(contract_path))["regimes"]
    return [{k: float(reg[k]) for k in HESTON_PARAM_NAMES} for reg in regimes.values()]


def sample_hypercube_params(ranges: dict[str, tuple[float, float]], n: int, seed: int,
                            feller_min: float = 0.40, method: str = "latin_hypercube",
                            max_rounds: int = 100, anchors: list[dict] | None = None,
                            excise_rel_radius: float = 0.10) -> np.ndarray:
    """v6 Option A training-parameter sampler (columns in `ranges` key order).

    Draws (kappa, theta, xi, rho, v0) in `ranges` by seeded Latin hypercube (or uniform)
    and REJECTS/resamples points failing either filter:
    1. Feller: 2*kappa*theta/xi^2 < feller_min (hypercube corners reach ~0.11, where
       the oracle is untrustworthy).
    2. Anchor excision (only when `anchors` is not None): in RANGE-NORMALIZED
       coordinates u = (x - lo)/(hi - lo) per dimension, a sample is rejected iff its
       EUCLIDEAN distance to any anchor (same normalization) is
       < excise_rel_radius * sqrt(d), i.e. an L2 ball whose radius is excise_rel_radius
       of the unit-cube diagonal. Each anchor is a dict containing every key of
       `ranges`; anchors outside the hypercube still excise where their ball
       intersects it.

    Raises RuntimeError after max_rounds if the acceptance region is too small —
    never loops forever. With anchors=None the output is bit-identical to the
    pre-excision sampler for the same seed.
    """
    names = list(ranges)
    ik, it, ix = names.index("kappa"), names.index("theta"), names.index("xi")
    lo = np.array([ranges[k][0] for k in names])
    hi = np.array([ranges[k][1] for k in names])
    anc_u = None
    if anchors is not None:
        anc = np.array([[a[k] for k in names] for a in anchors], dtype=float)
        anc_u = (anc - lo) / (hi - lo)
        r2_ex = excise_rel_radius ** 2 * len(names)   # (excise_rel_radius * sqrt(d))^2
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    total = 0
    for _ in range(max_rounds):
        if method == "latin_hypercube":
            u = qmc.LatinHypercube(d=len(names), seed=rng).random(n)
        else:
            u = rng.random((n, len(names)))
        pts = lo + u * (hi - lo)
        mask = 2.0 * pts[:, ik] * pts[:, it] / pts[:, ix] ** 2 >= feller_min
        if anc_u is not None:
            # normalize from pts (not u) so returned points satisfy the distance
            # property exactly when recomputed from raw coordinates
            un = (pts - lo) / (hi - lo)
            d2 = ((un[:, None, :] - anc_u[None, :, :]) ** 2).sum(axis=-1)
            mask &= (d2 >= r2_ex).all(axis=1)
        pts = pts[mask]
        kept.append(pts)
        total += pts.shape[0]
        if total >= n:
            break
    else:
        excise = "" if anchors is None else f", excise_rel_radius={excise_rel_radius}"
        raise RuntimeError(f"feller_min={feller_min}{excise}: fewer than {n} admissible "
                           f"samples after {max_rounds} rounds — acceptance region too small")
    out = np.concatenate(kept)[:n]
    assert (2.0 * out[:, ik] * out[:, it] / out[:, ix] ** 2 >= feller_min).all()
    return out


def saturation_sweep_configs(base: PINNConfig, multipliers: Iterable[int],
                             cap_multiplier: int) -> list[PINNConfig]:
    """A10 info-matched baseline: grow n_price_points until plateau; cap 5N, no hardcoded 3N."""
    ms = list(multipliers)
    assert max(ms) <= cap_multiplier, "sweep exceeds the contract cap (5N)"
    return [replace(base, n_price_points=m * base.n_price_points) for m in ms]


def info_matched_budget(n_ladder: int, accounting: str, labels_per_point: int = 4,
                        fd_stencil_size: int = 5) -> int:
    """Price-point budget matching the ladder's label information under the chosen accounting."""
    assert accounting in ("raw_scalar", "fd_equivalent")
    return n_ladder * (labels_per_point if accounting == "raw_scalar" else fd_stencil_size)


def capacity_control_config(base: PINNConfig, width_mult: float = 2.0) -> PINNConfig:
    """2x-width retrain hook for the saturated baseline (A10 capacity control)."""
    return replace(base, width_mult=width_mult)


# ============================================================================
# TASK 1 — data plumbing: split-restricted arm view, minibatching, PDE points
# ============================================================================

def cuda_max_memory_bytes(device) -> int | None:
    """torch.cuda.max_memory_allocated on a CUDA device, else None (CPU runs)."""
    if torch.device(device).type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def sample_pde_points(n: int, seed: int, ranges: dict[str, tuple[float, float]],
                      inputs: Iterable[str], *, anchors: list[dict] | None = None,
                      feller_min: float = 0.40, excise_rel_radius: float = 0.10,
                      method: str = "latin_hypercube") -> np.ndarray:
    """Fresh PDE collocation batch, columns in `inputs` order (float64, shape (n, d)).

    Grid coordinates (S, K, tau) are drawn UNIFORMLY inside the contract grid ranges;
    the Heston parameter dims (kappa, theta, xi, rho, v0) come from
    sample_hypercube_params with the SAME Feller filter and anchor excision as the
    training/label samplers — so collocation never lands in a masked-out or held-out
    region. Two independent seeded streams (the hypercube's own, and [seed, 8191] for
    the grid coords) keep the draw deterministic given `seed`. Called once per EPOCH.
    """
    inputs = tuple(inputs)
    pnames = [k for k in HESTON_PARAM_NAMES if k in inputs]
    prm = sample_hypercube_params({k: ranges[k] for k in pnames}, n, seed,
                                  feller_min=feller_min, method=method, anchors=anchors,
                                  excise_rel_radius=excise_rel_radius)
    cols: dict[str, np.ndarray] = {k: prm[:, j] for j, k in enumerate(pnames)}
    rng = np.random.default_rng([int(seed), 8191])   # independent of the hypercube stream
    for k in inputs:
        if k not in cols:
            lo, hi = ranges[k]
            cols[k] = rng.uniform(lo, hi, n)
    return np.stack([cols[k] for k in inputs], axis=1)


class ArmDataset:
    """Split-restricted, minibatchable view of one arm's labels (from build_arm_labels).

    Wraps make_labels.build_arm_labels (mask-filtered consensus rows for `arm_cfg`) and
    keeps only the rows whose PARAMETER POINT falls in the requested split. The split is
    read from the P8 train_val artifact's "split" key (0=train, 1=val, one entry per
    hypercube parameter point); if absent it falls back to a by-parameter-point split at
    the contract's training_parameterization.sampling.val_param_frac, with rng [seed, 21]
    (the make_datasets rule) and warns. Splitting by parameter
    POINT (never by (S, K, tau) row) is leakage-safe: a point's rows share its price
    surface.

    Every emitted minibatch carries build_arm_labels' keys VERBATIM — including gamma_ref,
    which is load-bearing for the loss-scale freeze (the true-consensus gamma unit shared
    across the dose-response ladder). Minibatches are seeded permutations, one fresh
    permutation per epoch.
    """

    def __init__(self, labels_npz: str, arm_cfg: PINNConfig, split: str | int, *,
                 seed: int = 42, warn: bool = True,
                 contract_path: str = "heston_benchmark_v6.yaml") -> None:
        from make_labels import build_arm_labels   # lazy: avoids train_pinn<->make_labels cycle

        assert split in ("train", "val", 0, 1), f"split must be train/val/0/1, got {split!r}"
        want = 0 if split in ("train", 0) else 1
        full = build_arm_labels(labels_npz, arm_cfg, seed=seed)
        d = np.load(labels_npz, allow_pickle=False)
        n_points, n_skt = d["mask_any"].shape
        keep = ~d["mask_any"].ravel()                       # C-order over (point, skt)
        retained_point = np.flatnonzero(keep) // n_skt      # param-point index per retained row
        split_arr = self._resolve_split(d, n_points, seed, warn, contract_path)
        sel = split_arr[retained_point] == want
        idx = torch.as_tensor(np.flatnonzero(sel), dtype=torch.long)
        self.data = {k: v.index_select(0, idx).contiguous() for k, v in full.items()}
        self.split = "train" if want == 0 else "val"
        self.n_rows = int(idx.numel())
        self.n_points = int(n_points)

    @staticmethod
    def _resolve_split(d, n_points: int, seed: int, warn: bool,
                        contract_path: str = "heston_benchmark_v6.yaml") -> np.ndarray:
        """The P8 split vector (0=train, 1=val), or the contract-declared fallback."""
        files = d.files if hasattr(d, "files") else d
        if "split" in files:
            return np.asarray(d["split"]).astype(np.int64).ravel()
        val_param_frac = float(yaml.safe_load(open(contract_path))
                               ["training_parameterization"]["sampling"]["val_param_frac"])
        if warn:
            warnings.warn(
                f"labels artifact has no 'split' key; falling back to a "
                f"{val_param_frac:.0%}-by-parameter-point split (contract "
                "training_parameterization.sampling.val_param_frac) with rng [seed, 21]",
                RuntimeWarning)
        n_val = min(max(int(round(val_param_frac * n_points)), 1), n_points - 1)
        val_idx = np.random.default_rng([int(seed), 21]).choice(n_points, n_val, replace=False)
        split = np.zeros(n_points, dtype=np.int64)
        split[val_idx] = 1
        return split

    def iter_minibatches(self, batch_size: int, seed: int, epoch: int):
        """Yield per-batch-sliced dicts over a fresh seeded permutation for this epoch."""
        perm = np.random.default_rng([int(seed), 13, int(epoch)]).permutation(self.n_rows)
        for start in range(0, self.n_rows, batch_size):
            idx = torch.as_tensor(perm[start:start + batch_size], dtype=torch.long)
            yield {k: v.index_select(0, idx) for k, v in self.data.items()}


# ============================================================================
# TASK 2 — training config, the loop, compute accounting and config hashing
# ============================================================================

@dataclass
class TrainConfig:
    """Full-run training schedule (pinn_config.yaml `training:` block); tests override."""
    steps: int = 20000
    batch: int = 1024
    n_pde: int = 4096
    lr: float = 3.0e-3
    schedule: str = "cosine"
    cosine_to: float = 3.0e-5
    val_every: int = 500
    patience: int = 10
    grad_clip: float = 1.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def pde_sampling_spec(contract_path: str, pinn_cfg_path: str):
    """(ranges, anchors, feller_min, excise_rel_radius) for sample_pde_points.

    Grid ranges (S, K, tau) from the contract grid; parameter ranges from the contract
    sampling block; anchors from the contract regimes (source of truth); feller_min and
    excision radius from the pinn-config hypercube_sampling block.
    """
    contract = yaml.safe_load(open(contract_path))
    pinn = yaml.safe_load(open(pinn_cfg_path))
    hyc = pinn.get("hypercube_sampling", {})
    gc = contract["grid"]
    ranges = {ax: (float(gc[ax]["min"]), float(gc[ax]["max"])) for ax in ("S", "K", "tau")}
    samp = contract["training_parameterization"]["sampling"]["ranges"]
    for k in HESTON_PARAM_NAMES:
        ranges[k] = (float(samp[k][0]), float(samp[k][1]))
    anchors = anchors_from_contract(contract_path) if "regimes" in contract else None
    feller_min = float(hyc.get("feller_min", 0.40))
    excise = float(hyc.get("excision", {}).get("rel_radius", 0.10))
    return ranges, anchors, feller_min, excise


def _eval_val(model: SobolevPINN, val_batch: dict[str, torch.Tensor]) -> tuple[float, dict]:
    """Validation objective = the ARM'S OWN normalized loss total on the val split.

    Module-level (monkeypatchable in tests). Uses the arm's frozen loss scales; no
    optimizer step. Deterministic given the model and val batch.
    """
    terms = model.loss(val_batch)
    return float(terms["total"].detach()), {k: float(v.detach()) for k, v in terms.items()}


def train_model(cfg: PINNConfig, train_ds: ArmDataset, val_ds: ArmDataset,
                tcfg: TrainConfig, seed: int, *, device: str = "cpu",
                pde_ranges: dict | None = None, pde_anchors: list[dict] | None = None,
                feller_min: float = 0.40, excise_rel_radius: float = 0.10,
                early_stop: bool = True) -> tuple[SobolevPINN, dict, dict, dict]:
    """Train one arm; return (model_at_last_step, best_state, last_state, runlog).

    - FIRST-BATCH SCALE FREEZE uses the FULL training split (not a minibatch), so the
      per-term loss scales are bit-identical across arms AND across minibatch orders.
    - PDE collocation is resampled FRESH PER EPOCH (Feller + anchor excision).
    - best_state = the best-validation checkpoint (arm-fair early-stopping SELECTION on
      the arm's own normalized total). When early_stop, the loop also TERMINATES after
      `patience` non-improving val checks; --matched-epochs (early_stop=False) runs the
      full fixed `steps` so last_state is a genuine matched-epochs checkpoint.
    """
    set_seed(seed)
    dev = torch.device(device)
    dt = getattr(torch, cfg.dtype)
    model = build_model(cfg).to(dev)
    train_data = {k: v.to(dev) for k, v in train_ds.data.items()}
    val_batch = {k: v.to(dev) for k, v in val_ds.data.items()}
    # FULL-split, once — order-independent scales (do NOT let a minibatch set them).
    if cfg.loss_scale_mode == "label_second_moment":
        model._freeze_loss_scales(train_data)

    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, tcfg.steps),
                                                        eta_min=tcfg.cosine_to)
             if tcfg.schedule == "cosine" else None)
    use_pde = cfg.use_pde and cfg.lambda_pde != 0.0
    if use_pde and pde_ranges is None:
        raise ValueError("use_pde arm needs pde_ranges (see pde_sampling_spec)")
    if use_pde and pde_anchors is None:
        # CLAUDE.md invariant: anchor excision is part of EVERY training-sampling call;
        # PDE collocation must not land in a held-out named-regime ball.
        raise ValueError("use_pde arm needs pde_anchors — never sample collocation "
                         "points without excising the named eval anchors")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    grad0, t0 = sp._GRAD_CALLS["n"], time.perf_counter()

    best = {"val": float("inf"), "step": -1, "state": None}
    val_curve: list[dict] = []
    step, epoch, since_improve, stopped_early = 0, 0, 0, False
    while step < tcfg.steps and not stopped_early:
        x_pde = None
        if use_pde:
            eseed = int(np.random.default_rng([int(seed), 555, epoch]).integers(1, 2 ** 31 - 1))
            x_pde = torch.as_tensor(
                sample_pde_points(tcfg.n_pde, eseed, pde_ranges, cfg.inputs,
                                  anchors=pde_anchors, feller_min=feller_min,
                                  excise_rel_radius=excise_rel_radius),
                dtype=dt, device=dev)
        for mb in train_ds.iter_minibatches(tcfg.batch, seed, epoch):
            mb = {k: v.to(dev) for k, v in mb.items()}
            if x_pde is not None:
                mb["x_pde"] = x_pde
            opt.zero_grad(set_to_none=True)
            terms = model.loss(mb)
            terms["total"].backward()
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if step % tcfg.val_every == 0 or step >= tcfg.steps:
                v_total, v_terms = _eval_val(model, val_batch)
                val_curve.append({
                    "step": step, "val_total": v_total, "val_terms": v_terms,
                    "train_total": float(terms["total"].detach()),
                    "train_terms": {k: float(v.detach()) for k, v in terms.items()},
                    "lr": (sched.get_last_lr()[0] if sched is not None else tcfg.lr)})
                if v_total < best["val"]:
                    best = {"val": v_total, "step": step,
                            "state": {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}}
                    since_improve = 0
                else:
                    since_improve += 1
                    if early_stop and since_improve >= tcfg.patience:
                        stopped_early = True
            if step >= tcfg.steps or stopped_early:
                break
        epoch += 1

    if best["state"] is None:   # steps < val_every: snapshot the final model as best
        v_total, _ = _eval_val(model, val_batch)
        best = {"val": v_total, "step": step,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    pc = param_counts(model)
    compute = {"wall_clock_s": time.perf_counter() - t0,
               "param_count": pc["total"],
               "price_head_param_count": pc["price_head"],
               "correction_param_count": pc["correction"],
               "derivative_evals": sp._GRAD_CALLS["n"] - grad0,
               "peak_memory_bytes": peak_rss_bytes(),
               "n_pde_per_epoch": tcfg.n_pde if use_pde else 0,
               "epochs": epoch, "steps": step, "stopped_early": stopped_early}
    cuda_peak = cuda_max_memory_bytes(dev)
    if cuda_peak is not None:
        compute["cuda_max_memory_bytes"] = cuda_peak
    # 'last' is always the final model. It is ALSO 'matched_epochs' only when the
    # step budget was actually reached: with early_stop on (train.py's DEFAULT,
    # `early_stop = not --matched-epochs`) the loop breaks at the patience step, and
    # labelling that checkpoint matched_epochs would let the contract's
    # compute_accounting.report_both table compare arms that stopped at different
    # steps as though their budgets were matched (audit T2). The early-stop DEFAULT
    # is deliberately unchanged — that is a run-book decision, not a code one.
    last_ck = {"step": step, "path": "last.pt",
               "val_total": (val_curve[-1]["val_total"] if val_curve else best["val"]),
               "reached_max_steps": step >= tcfg.steps}
    checkpoints = {
        "best": {"step": best["step"], "val_total": best["val"], "path": "best.pt"},
        "last": last_ck}
    if last_ck["reached_max_steps"]:
        checkpoints["matched_epochs"] = dict(last_ck)
    runlog = {
        "val_curve": val_curve,
        "checkpoints": checkpoints,
        "compute": compute,
        "n_train_rows": train_ds.n_rows, "n_val_rows": val_ds.n_rows}
    return model, best["state"], last_state, runlog


def _cfg_dict(cfg: PINNConfig) -> dict:
    """PINNConfig as a JSON-normalized dict (tuples->lists) for stable hashing/serializing."""
    return json.loads(json.dumps(asdict(cfg), sort_keys=True, default=str))


def config_hash(cfg: PINNConfig, seed: int, data_manifest_sha: str | None,
                lambdas_sha: str | None) -> str:
    """sha256 over (resolved PINNConfig dict + data manifest sha + lambdas sha + seed)."""
    payload = {"cfg": _cfg_dict(cfg), "seed": int(seed),
               "data_manifest_sha256": data_manifest_sha, "lambdas_sha256": lambdas_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path: str | None) -> str | None:
    """Hex sha256 of a file's bytes, or None when the path is missing/None."""
    if path and Path(path).exists():
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return None


def data_manifest_sha(data_path: str) -> str | None:
    """The data artifact's manifest_sha256 (train_val/labels manifest) or a byte hash fallback."""
    p = Path(data_path)
    for man in (p.with_name("train_val_manifest.json"), p.with_name("manifest.json")):
        if man.exists():
            try:
                return json.loads(man.read_text()).get("manifest_sha256")
            except (OSError, json.JSONDecodeError):
                pass
    return file_sha256(data_path)


# ============================================================================
# TASK 4 — VALIDATION-ONLY joint lambda selection (extends select_lambda_pde)
# ============================================================================

def select_lambdas(candidates_pde: Iterable[float], candidates_gamma: Iterable[float],
                   candidates_vega: Iterable[float], *,
                   fit_and_val_score: Callable[[float, float, float], float],
                   test_set: LockedTestSet, out_path: str | None = None,
                   config_hash: str | None = None, lambda_delta: float = 1.0) -> dict:
    """Pick (lambda_pde, lambda_gamma, lambda_vega) on VALIDATION ONLY.

    A generic guarded grid search. The PROTOCOL that decides which arm scores which axis
    lives in train.py::_run_select_lambdas, which calls this TWICE per the contract's
    `lambda_selection` (1-D lambda_pde on the baseline source arm, then 2-D
    (lambda_gamma, lambda_vega) on the rung3 source arm at that fixed lambda_pde) — do not
    collapse those two calls back into one joint search (contract Q3).

    Grid-searches the three candidate axes, scoring each combo with `fit_and_val_score`
    (a SHORT rung3 fit on the training split -> mean normalized validation RMSE of
    price/delta/gamma/vega; lower is better). The held-out anchor grids stay behind
    `test_set` (LockedTestSet) the entire time — accessing them raises, exactly like
    select_lambda_pde. lambda_delta is FIXED at 1.0: delta labels are already
    O(1)-comparable after loss-scale normalization and delta is not a swept axis in the
    contract. Writes lambdas_selected.yaml when out_path is given.
    """
    if not test_set.locked:
        raise RuntimeError("test set unlocked before lambda selection")
    table: list[dict] = []
    best: dict | None = None
    for lp in candidates_pde:
        for lg in candidates_gamma:
            for lv in candidates_vega:
                score = float(fit_and_val_score(float(lp), float(lg), float(lv)))
                if not test_set.locked:
                    raise RuntimeError("test set touched during lambda selection")
                row = {"lambda_pde": float(lp), "lambda_gamma": float(lg),
                       "lambda_vega": float(lv), "score": score}
                table.append(row)
                if best is None or score < best["score"]:
                    best = row
    if best is None:
        raise ValueError("empty candidate grid — nothing to select")
    result = {"lambda_pde": best["lambda_pde"], "lambda_gamma": best["lambda_gamma"],
              "lambda_vega": best["lambda_vega"], "lambda_delta": float(lambda_delta),
              "scores_table": table, "config_hash": config_hash}
    if out_path is not None:
        Path(out_path).write_text(yaml.safe_dump(result, sort_keys=False))
    return result