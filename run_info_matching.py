"""run_info_matching.py — A10 information-matching saturation-curve driver.

Answers the "3N/5N dissolved" clause of the contract's `information_matching` block:
grow the price-only `info_matched_baseline`'s price-point budget until its VALIDATION
Greek accuracy PLATEAUS, cap 5N, then re-run the saturated rung at 2x width so a flat
capacity-control point kills the "you were just optimization-limited" objection.

WHAT THE CURVE IS (and is NOT). The saturation curve

    plateau bounds what THIS architecture/protocol extracts from prices, NOT an
    information bound

(the wording is the contract's, read verbatim from `information_matching.wording`). It
is a statement about the estimator, not about the prices: a rung3 point still buys
supervised Delta/Gamma/Vega labels that the price-only baseline must reconstruct.

MECHANISM (all declared, nothing resampled):
  * The rows are the FROZEN training rows of the P8 label artifact. Multiplier m uses the
    first m*N of ONE seeded permutation of that split (subsample_train) — so the m=2 set
    CONTAINS the m=1 set (a genuine "grow the budget" ladder) and NO new labels are ever
    generated. When m*N exceeds the frozen row count the rung saturates at the whole split
    (honest: you cannot manufacture information you did not freeze).
  * Budget is logged BOTH ways per rung via train_pinn.info_matched_budget — the
    fd_equivalent (5-pt stencil) and raw_scalar (4 labels/point) resolution versions the
    contract asks for; `info_matching.accounting` names the headline one.
  * PLATEAU RULE (declared here and echoed into the CSV/paragraph): the plateau is the
    SMALLEST m >= 2 whose relative improvement of the mean-over-seeds validation Gamma
    rel-RMSE versus m-1 is < PLATEAU_TOL (2%). If no rung clears the bar the plateau is
    the 5N cap and `plateau_reached` is False.
  * CAPACITY CONTROL: the plateau-m config is retrained at width_mult=2.0
    (train_pinn.capacity_control_config), SAME seeds, SAME subsample; both points are
    reported.

Outputs (out_dir): saturation_curve.csv (seed-aggregated headline), saturation_curve_per_seed.csv
(per run, with compute accounting + config hash), saturation_curve.png, and
saturation_summary.txt (the contract-worded paragraph). Mirrors the repo's per_seed/agg
+ figure house style (see run_hedging / eval_greeks / exhibits).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from pinn_provider import _ARM_DIR                 # arm -> ckpt dir map the full-sweep loader uses
from SobolevPINN import load_arm
from train_pinn import (ArmDataset, TrainConfig, capacity_control_config, config_hash,
                        data_manifest_sha, pde_sampling_spec, saturation_sweep_configs,
                        train_model)

# The plateau tolerance is fixed by this study, not a swept knob — declared once here and
# echoed into every artifact so the rule is legible from the outputs alone.
PLATEAU_TOL = 0.02                                  # < 2% mean-Gamma-RMSE gain => plateau
SATURATED_ARM = "info_matched_baseline"             # price-only; n_price_points is the sweep axis
_VAL_GREEKS = ("price", "delta", "gamma", "vega")   # gamma drives the plateau; rest are context


# ---------------------------------------------------------------------------
# frozen-row subsampling (NEVER resamples labels — same artifact, declared)
# ---------------------------------------------------------------------------

def subsample_train(train_ds: ArmDataset, n_rows: int, seed: int) -> ArmDataset:
    """A budget-`n_rows` view of an ArmDataset's frozen training rows (seeded, NESTED).

    Takes the first `n_rows` of ONE seed-derived permutation of the split, so for a fixed
    seed the multiplier ladder is nested (larger budgets ADD rows, never swap them) and two
    calls with the same (dataset, n_rows, seed) are bit-identical. Capped at the frozen row
    count — no new labels are invented. Returns a shallow copy sharing everything but the
    row-sliced `.data`/`.n_rows` (so iter_minibatches / train_model see the smaller budget).
    """
    n_keep = min(int(n_rows), train_ds.n_rows)
    perm = np.random.default_rng([int(seed), 40961]).permutation(train_ds.n_rows)
    idx = torch.as_tensor(perm[:n_keep], dtype=torch.long)
    sub = copy.copy(train_ds)
    sub.data = {k: v.index_select(0, idx).contiguous() for k, v in train_ds.data.items()}
    sub.n_rows = int(idx.numel())
    return sub


# ---------------------------------------------------------------------------
# validation Greek RMSE (against the TRUE consensus; gamma vs gamma_ref)
# ---------------------------------------------------------------------------

def val_greek_rmse(model, val_ds: ArmDataset, device: str = "cpu") -> dict:
    """{greek: {rmse, rel_rmse}} of a fitted arm on the val split's true-consensus labels.

    Gamma is scored against gamma_ref (the FROZEN true-consensus gamma, per the loss-scale
    invariant) — never a per-arm/noised label. rel_rmse = rmse / rms(true); the plateau
    reads the gamma rel_rmse (rms(true) is constant across the sweep, so the plateau's
    RELATIVE improvement is identical whether rmse is absolute or relative).
    """
    x = val_ds.data["x"].to(device)
    g = model.greeks_eval(x, need=("delta", "gamma", "vega"))
    with torch.no_grad():
        price = model(x).detach()
    pred = {"price": price.cpu(), "delta": g["delta"].cpu(),
            "gamma": g["gamma"].cpu(), "vega": g["vega"].cpu()}
    out = {}
    for q in _VAL_GREEKS:
        true = (val_ds.data["gamma_ref"] if q == "gamma" else val_ds.data[q]).cpu()
        err = pred[q] - true
        rmse = float(torch.sqrt(torch.mean(err ** 2)))
        rms = float(torch.sqrt(torch.mean(true ** 2)))
        out[q] = {"rmse": rmse, "rel_rmse": (rmse / rms if rms > 0 else float("nan"))}
    return out


# ---------------------------------------------------------------------------
# the plateau rule (pure; unit-tested on synthetic curves)
# ---------------------------------------------------------------------------

def plateau_multiplier(multipliers, mean_gamma_rmse, tol: float = PLATEAU_TOL) -> dict:
    """Smallest m >= 2 whose mean Gamma-RMSE gain over m-1 is < `tol`, else the cap.

    `multipliers` are the ladder's m values (ascending), `mean_gamma_rmse[i]` the
    mean-over-seeds validation Gamma (rel-)RMSE at multipliers[i]. Relative improvement is
    (prev - cur) / prev; a non-improving (flat OR worse) rung trips the plateau. Returns
    {plateau_multiplier, plateau_index, plateau_reached, rel_improvement} where
    rel_improvement is the per-rung list (NaN at the first rung, no predecessor).
    """
    ms = list(multipliers)
    r = [float(x) for x in mean_gamma_rmse]
    assert len(ms) == len(r) and len(ms) >= 1, "multipliers/rmse length mismatch"
    rel = [float("nan")]
    plateau_i = None
    for i in range(1, len(ms)):
        prev, cur = r[i - 1], r[i]
        impr = (prev - cur) / prev if prev > 0 else float("nan")
        rel.append(impr)
        if plateau_i is None and (not math.isfinite(impr) or impr < tol):
            plateau_i = i
    if plateau_i is None:                       # never plateaued within the cap
        plateau_i, reached = len(ms) - 1, False
    else:
        reached = True
    return {"plateau_multiplier": ms[plateau_i], "plateau_index": plateau_i,
            "plateau_reached": reached, "rel_improvement": rel}


# ---------------------------------------------------------------------------
# CSV / paragraph helpers
# ---------------------------------------------------------------------------

def _agg(values) -> tuple[float, float]:
    """(mean, sample-std ddof=1) over finite entries; std=0 for one seed (eval_greeks rule)."""
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


def _write_csv(path, cols, rows) -> None:
    """Fixed-header CSV; non-finite floats blanked (matches eval_greeks._write_csv)."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean = {}
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float) and not math.isfinite(v):
                    v = ""
                clean[c] = v
            w.writerow(clean)


PERSEED_COLS = ["multiplier", "width_mult", "seed", "config_hash", "n_price_points",
                "n_train_rows", "fd_equivalent_budget", "raw_scalar_budget",
                "gamma_rmse", "gamma_rel_rmse", "delta_rel_rmse", "vega_rel_rmse",
                "price_rel_rmse", "wall_clock_s", "param_count", "derivative_evals",
                "peak_memory_bytes", "epochs", "steps"]

AGG_COLS = ["multiplier", "width_mult", "n_price_points", "n_train_rows_mean",
            "fd_equivalent_budget", "raw_scalar_budget", "accounting", "n_seeds",
            "gamma_rel_rmse_mean", "gamma_rel_rmse_std", "gamma_rmse_mean",
            "delta_rel_rmse_mean", "vega_rel_rmse_mean", "price_rel_rmse_mean",
            "rel_improvement_vs_prev", "is_plateau", "plateau_multiplier"]


def _plot_saturation(agg_rows, plat, out_path) -> str:
    """Mean validation Gamma rel-RMSE vs multiplier (width 1.0 curve) + the 2x-width point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = sorted((r for r in agg_rows if r["width_mult"] == 1.0), key=lambda r: r["multiplier"])
    xs = [r["multiplier"] for r in base]
    ys = [r["gamma_rel_rmse_mean"] for r in base]
    es = [r["gamma_rel_rmse_std"] for r in base]
    cap = [r for r in agg_rows if r["width_mult"] == 2.0]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label="width_mult 1.0")
    pm = plat["plateau_multiplier"]
    ax.axvline(pm, ls="--", color="0.5",
               label=f"plateau m={pm}" + ("" if plat["plateau_reached"] else " (cap; not reached)"))
    for r in cap:
        ax.errorbar([r["multiplier"]], [r["gamma_rel_rmse_mean"]], yerr=[r["gamma_rel_rmse_std"]],
                    marker="s", capsize=3, color="C3", label="width_mult 2.0 (capacity control)")
    ax.set_xlabel("price-point budget multiplier m  (n_price_points = m x N)")
    ax.set_ylabel("mean validation Gamma rel-RMSE")
    ax.set_title("A10 information-matching saturation curve")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def build_paragraph(contract: dict, plat: dict, agg_rows, accounting: str) -> str:
    """The reported paragraph — plateau finding + capacity control, ending on the contract's
    EXACT wording (read from information_matching.wording, the source of truth)."""
    wording = contract["information_matching"]["wording"]
    pm = plat["plateau_multiplier"]
    lut = {(r["multiplier"], r["width_mult"]): r for r in agg_rows}
    w1 = lut.get((pm, 1.0))
    w2 = lut.get((pm, 2.0))
    reached = ("reached" if plat["plateau_reached"]
               else "NOT reached within the 5N cap; the plateau is pinned at the cap")
    lines = [
        f"Information-matching saturation sweep for `{SATURATED_ARM}` (price-only baseline). "
        f"Growing the price-point budget m x N and scoring validation Gamma rel-RMSE, the "
        f"plateau rule (smallest m>=2 with < {PLATEAU_TOL:.0%} mean-Gamma-RMSE gain over m-1) "
        f"is {reached} at m = {pm}"
        + (f" (mean Gamma rel-RMSE {w1['gamma_rel_rmse_mean']:.4g})." if w1 else "."),
    ]
    if w2 is not None and w1 is not None:
        w1r, w2r = w1["gamma_rel_rmse_mean"], w2["gamma_rel_rmse_mean"]
        delta = (w1r - w2r) / w1r if w1r > 0 else float("nan")
        flat = math.isfinite(delta) and delta < PLATEAU_TOL
        verdict = ("still flat at 2x width (< {:.0%} gain) — the optimization-limited "
                   "objection dies").format(PLATEAU_TOL) if flat else (
                   "IMPROVED at 2x width ({:+.1%}) — capacity, not information, was binding "
                   "at the plateau".format(delta))
        lines.append(
            f"Capacity control: retraining the saturated m={pm} config at 2x width gives mean "
            f"Gamma rel-RMSE {w2r:.4g} vs {w1r:.4g} ({delta:+.1%}); {verdict}.")
    lines.append(f"Budget reported both ways ({accounting} is the headline): fd_equivalent "
                 "(5-pt stencil) and raw_scalar (4 labels/rung3-point) resolution versions "
                 "are in saturation_curve.csv.")
    lines.append(f'Interpretation (contract wording, verbatim): "{wording}".')
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# plateau-m checkpoint persistence (feeds run_hedging's full-sweep loader)
# ---------------------------------------------------------------------------
#
# The full sweep (run_hedging.run_full_sweep -> pinn_provider.build_providers) hedges
# `info_matched_baseline` from a checkpoint at ckpt_root/<arm dir>/s<seed>/best.pt, where
# <arm dir> = _ARM_DIR.get(name, name) (identity for this arm). That loader rebuilds the
# model from ckpt["cfg"] and load_state_dict(ckpt["state_dict"]) (the SHAPE gate) after a
# weights_only=False torch.load; so we persist the SAME dict schema train.py writes
# (state_dict + cfg asdict + frozen loss scales + config_hash/seed/arm — the sha/DGP audit
# fields). The persisted model is the PLATEAU-m rung at WIDTH 1.0 (never the 2x-width
# capacity-control rung), and its best-VALIDATION state (never the last step). A per-seed
# sidecar JSON records how the budget was chosen, so the choice is auditable off-line.

BUDGET_SIDECAR = "info_matched_budget.json"     # next to best.pt; the auditable budget record


def _ckpt_arm_dir() -> str:
    """The checkpoint sub-directory the full-sweep loader reads for this arm (identity here,
    but resolved through the loader's own _ARM_DIR map so the two never silently diverge)."""
    return _ARM_DIR.get(SATURATED_ARM, SATURATED_ARM)


def _sanitize(o):
    """Recursively map non-finite floats to None so the sidecar is strict, parseable JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _budget_sidecar(seed, pm, N, plat, sat_curve, plateau_row, tol, cap, accounting,
                    chash, dsha) -> dict:
    """The auditable per-seed budget record written beside best.pt.

    Carries the SELECTED multiplier, the full seed-aggregated saturation curve, the declared
    plateau criterion, and — when the plateau was never reached within the cap — a LOUD note
    that the budget is pinned at the cap (NOT the N budget)."""
    reached = bool(plat["plateau_reached"])
    return {
        "arm": SATURATED_ARM,
        "seed": int(seed),
        "checkpoint": "best.pt",
        "checkpoint_is_best_validation": True,      # best_state, never last-step
        "width_mult": 1.0,                          # the hedged rung; NOT the 2x capacity control
        "selected_multiplier": int(pm),
        "n_price_points": int(pm * N),
        "N": int(N),
        "cap_multiplier": int(cap),
        "plateau_reached": reached,
        "plateau_pinned_at_cap": (not reached),
        "plateau_tol": float(tol),
        "plateau_criterion": (
            f"smallest m>=2 whose mean-over-seeds validation Gamma rel-RMSE improvement over "
            f"m-1 is < {tol:.2%}; else pinned at the {cap}N cap (plateau_reached=false)"),
        "accounting": accounting,
        "n_train_rows": int(plateau_row["n_train_rows"]),
        "fd_equivalent_budget": int(plateau_row["fd_equivalent_budget"]),
        "raw_scalar_budget": int(plateau_row["raw_scalar_budget"]),
        "config_hash": chash,
        "data_manifest_sha256": dsha,
        "saturation_curve": sat_curve,
        "note": (
            f"PLATEAU NOT REACHED within the {cap}N cap: budget is PINNED at the cap ({cap}N), "
            f"NOT the N budget — treat the plateau as a lower bound." if not reached else
            "plateau reached; budget is the smallest saturating multiplier"),
    }


def _persist_plateau_checkpoints(ckpt_root, seeds, pm, cfg_by_m, best_states, dsha,
                                 N, plat, sat_curve, plateau_rows, tol, cap,
                                 accounting) -> dict:
    """Write ckpt_root/<arm dir>/s<seed>/{best.pt, info_matched_budget.json} for every seed.

    Persists the plateau-m WIDTH-1.0 best-validation state (schema-identical to train.py's
    best.pt so pinn_provider.build_providers loads it unchanged) plus the auditable sidecar.
    Requires best_states[(pm, seed)] to exist for every seed — we cache every width-1.0 rung
    during the sweep precisely so the plateau rung is always in hand and we NEVER retrain or
    silently fall back to the N budget."""
    arm_dir = _ckpt_arm_dir()
    cfg = cfg_by_m[pm]                              # plateau rung at width 1.0
    cfg_dict = asdict(cfg)
    saved: dict = {}
    for seed in seeds:
        key = (pm, seed)
        assert key in best_states, (                # no fallback: the plateau rung MUST be cached
            f"plateau rung m={pm} seed={seed} has no cached best_state; refusing to "
            "fall back to the N budget")
        state = best_states[key]
        chash = config_hash(cfg, seed, dsha, None)
        scales = {k: v for k, v in state.items() if k.startswith("loss_scale")}
        d = Path(ckpt_root) / arm_dir / f"s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "cfg": cfg_dict, "scales": scales,
                    "config_hash": chash, "seed": int(seed), "arm": SATURATED_ARM},
                   d / "best.pt")
        sidecar = _budget_sidecar(seed, pm, N, plat, sat_curve, plateau_rows[seed],
                                  tol, cap, accounting, chash, dsha)
        (d / BUDGET_SIDECAR).write_text(json.dumps(_sanitize(sidecar), indent=2, default=str))
        saved[int(seed)] = {"best_pt": str(d / "best.pt"),
                            "sidecar": str(d / BUDGET_SIDECAR)}
    return saved


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

def run_saturation_sweep(data: str, seeds, out_dir: str, *,
                         pinn_cfg: str = "pinn_config.yaml",
                         contract: str = "heston_benchmark_v6.yaml",
                         device: str = "cpu", steps: int | None = None,
                         multipliers=None, base_n_price_points: int | None = None,
                         tol: float = PLATEAU_TOL, train_overrides: dict | None = None,
                         ckpt_root: str | None = None) -> dict:
    """Run the full A10 saturation sweep + capacity control; write the CSVs, PNG, paragraph,
    AND persist the plateau-m checkpoint the downstream full sweep hedges.

    For every (multiplier m, seed) the frozen train split is subsampled to m*N rows
    (subsample_train) and `info_matched_baseline` is trained; validation Greek RMSE is
    recorded. The plateau-m config is then retrained at 2x width for the same seeds. The
    plateau-m WIDTH-1.0 best-validation checkpoint is written per seed to
    ckpt_root/<arm dir>/s<seed>/best.pt (+ an auditable budget sidecar) in exactly the layout
    pinn_provider.build_providers reads (run_hedging.run_full_sweep otherwise dies with
    FileNotFoundError). Returns a summary dict (paths, plateau, per-seed + agg rows, paragraph,
    checkpoints).

    `ckpt_root` is where the per-seed checkpoints land (defaults to `out_dir`; real runs point
    it at the shared grid root so info_matched_baseline sits beside the other arms).
    `base_n_price_points` overrides N (the config default is 4096; tests shrink it so the
    subsample actually bites on a tiny artifact). `steps` overrides the training budget.
    """
    contract_d = yaml.safe_load(Path(contract).read_text())
    pinn_raw = yaml.safe_load(Path(pinn_cfg).read_text())
    im = pinn_raw.get("info_matching", {})
    cap = int(im.get("cap_multiplier", 5))
    mults = list(multipliers if multipliers is not None
                 else im.get("n_price_points_multipliers", [1, 2, 3, 4, 5]))
    accounting = im.get("accounting", "fd_equivalent")
    labels_per_point = int(im.get("labels_per_point", 4))
    fd_stencil = int(im.get("fd_stencil_size", 5))
    cap_width = float(im.get("capacity_control_width_mult", 2.0))
    seeds = [int(s) for s in seeds]

    base = load_arm(pinn_cfg, SATURATED_ARM)
    if base_n_price_points is not None:
        base = replace(base, n_price_points=int(base_n_price_points))
    N = base.n_price_points

    tcfg = TrainConfig.from_dict(pinn_raw.get("training", {}))
    if train_overrides:                                 # test/CI knob (n_pde, batch, ...)
        tcfg = replace(tcfg, **train_overrides)
    if steps is not None:
        tcfg = replace(tcfg, steps=int(steps))
    ranges, anchors, feller_min, excise = pde_sampling_spec(contract, pinn_cfg)
    dsha = data_manifest_sha(data)

    # saturation_sweep_configs enforces the 5N cap (raises if any m > cap) — reused here.
    sweep_cfgs = saturation_sweep_configs(base, mults, cap)

    # val split is arm-independent for a price-only baseline; build once per seed.
    val_by_seed = {s: ArmDataset(data, base, "val", seed=s) for s in seeds}
    train_by_seed = {s: ArmDataset(data, base, "train", seed=s) for s in seeds}

    def _one(cfg, seed, n_target):
        train_ds = subsample_train(train_by_seed[seed], n_target, seed)
        model, best_state, _, runlog = train_model(
            cfg, train_ds, val_by_seed[seed], tcfg, seed, device=device,
            pde_ranges=ranges, pde_anchors=anchors, feller_min=feller_min,
            excise_rel_radius=excise, early_stop=True)
        model.load_state_dict(best_state)
        greeks = val_greek_rmse(model, val_by_seed[seed], device)
        comp = runlog["compute"]
        row = {"seed": seed, "n_price_points": cfg.n_price_points,
               "n_train_rows": train_ds.n_rows,
               "config_hash": config_hash(cfg, seed, dsha, None),
               "fd_equivalent_budget": train_ds.n_rows * fd_stencil,
               "raw_scalar_budget": train_ds.n_rows * labels_per_point,
               "gamma_rmse": greeks["gamma"]["rmse"],
               "gamma_rel_rmse": greeks["gamma"]["rel_rmse"],
               "delta_rel_rmse": greeks["delta"]["rel_rmse"],
               "vega_rel_rmse": greeks["vega"]["rel_rmse"],
               "price_rel_rmse": greeks["price"]["rel_rmse"],
               "wall_clock_s": comp["wall_clock_s"], "param_count": comp["param_count"],
               "derivative_evals": comp["derivative_evals"],
               "peak_memory_bytes": comp["peak_memory_bytes"],
               "epochs": comp["epochs"], "steps": comp["steps"]}
        return row, best_state

    # cache every width-1.0 rung's best_state (keyed (m, seed)) so the plateau rung — whichever
    # it turns out to be — is always in hand to persist, without retraining or falling back.
    best_states: dict = {}
    cfg_by_m = dict(zip(mults, sweep_cfgs))
    perseed: list[dict] = []
    for m, cfg in zip(mults, sweep_cfgs):
        for seed in seeds:
            row, best_state = _one(cfg, seed, m * N)
            row.update({"multiplier": m, "width_mult": 1.0})
            perseed.append(row)
            best_states[(m, seed)] = best_state

    # aggregate the width-1.0 ladder, then apply the plateau rule to mean Gamma rel-RMSE
    agg = _aggregate(perseed, accounting)
    curve = sorted((r for r in agg if r["width_mult"] == 1.0), key=lambda r: r["multiplier"])
    plat = plateau_multiplier([r["multiplier"] for r in curve],
                              [r["gamma_rel_rmse_mean"] for r in curve], tol)

    # capacity control: retrain the plateau rung at 2x width, same seeds/subsample
    pm = plat["plateau_multiplier"]
    cap_cfg = capacity_control_config(replace(base, n_price_points=pm * N), cap_width)
    for seed in seeds:
        row, _ = _one(cap_cfg, seed, pm * N)        # capacity control is reported, never hedged
        row.update({"multiplier": pm, "width_mult": cap_width})
        perseed.append(row)

    agg = _aggregate(perseed, accounting)
    for r in agg:                                   # annotate the plateau onto the curve
        idx = next((i for i, c in enumerate(curve) if c["multiplier"] == r["multiplier"]), None)
        r["rel_improvement_vs_prev"] = (plat["rel_improvement"][idx]
                                        if (idx is not None and r["width_mult"] == 1.0)
                                        else float("nan"))
        r["is_plateau"] = bool(r["width_mult"] == 1.0 and r["multiplier"] == pm)
        r["plateau_multiplier"] = pm

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"per_seed": out / "saturation_curve_per_seed.csv",
             "curve": out / "saturation_curve.csv",
             "png": out / "saturation_curve.png",
             "summary": out / "saturation_summary.txt"}
    _write_csv(paths["per_seed"], PERSEED_COLS, perseed)
    _write_csv(paths["curve"], AGG_COLS, agg)
    _plot_saturation(agg, plat, paths["png"])
    paragraph = build_paragraph(contract_d, plat, agg, accounting)
    paths["summary"].write_text(paragraph + "\n")

    # persist the plateau-m WIDTH-1.0 best.pt (+ budget sidecar) per seed, in the layout the
    # full-sweep loader reads. If the plateau was never reached, pm is already the cap (5N):
    # we save that 5N model and shout — never a silent fall-back to the N budget.
    sat_curve = [{"multiplier": c["multiplier"], "n_price_points": c["n_price_points"],
                  "gamma_rel_rmse_mean": c["gamma_rel_rmse_mean"],
                  "gamma_rel_rmse_std": c["gamma_rel_rmse_std"],
                  "rel_improvement_vs_prev": plat["rel_improvement"][i],
                  "n_seeds": c["n_seeds"]} for i, c in enumerate(curve)]
    plateau_rows = {r["seed"]: r for r in perseed
                    if r["multiplier"] == pm and r["width_mult"] == 1.0}
    ckpt_root = out_dir if ckpt_root is None else ckpt_root
    checkpoints = _persist_plateau_checkpoints(
        ckpt_root, seeds, pm, cfg_by_m, best_states, dsha, N, plat, sat_curve,
        plateau_rows, tol, cap, accounting)
    if not plat["plateau_reached"]:
        warnings.warn(
            f"[info-matching] plateau NOT reached within the {cap}N cap: persisting the "
            f"CAP ({cap}N) model as info_matched_baseline (budget pinned at the cap, NOT N).",
            RuntimeWarning)

    print(f"[info-matching] N={N}, multipliers={mults} (cap {cap}), seeds={seeds}; "
          f"plateau m={pm} ({'reached' if plat['plateau_reached'] else 'CAP, not reached'}). "
          f"Wrote {', '.join(p.name for p in paths.values())} to {out}; "
          f"persisted {len(checkpoints)} best.pt under "
          f"{Path(ckpt_root) / _ckpt_arm_dir()}/s<seed>/.")
    return {"paths": {k: str(v) for k, v in paths.items()},
            "plateau": plat, "per_seed": perseed, "agg": agg,
            "paragraph": paragraph, "N": N, "multipliers": mults, "seeds": seeds,
            "checkpoints": checkpoints, "ckpt_root": str(ckpt_root)}


def _aggregate(perseed, accounting) -> list[dict]:
    """Seed-aggregate per-seed rows into the curve (one row per (multiplier, width_mult))."""
    by: dict = {}
    for r in perseed:
        by.setdefault((r["multiplier"], r["width_mult"]), []).append(r)
    rows = []
    for (m, w), rs in sorted(by.items()):
        g_mean, g_std = _agg([r["gamma_rel_rmse"] for r in rs])
        grmse_mean, _ = _agg([r["gamma_rmse"] for r in rs])
        d_mean, _ = _agg([r["delta_rel_rmse"] for r in rs])
        v_mean, _ = _agg([r["vega_rel_rmse"] for r in rs])
        p_mean, _ = _agg([r["price_rel_rmse"] for r in rs])
        nrows_mean, _ = _agg([r["n_train_rows"] for r in rs])
        rows.append({"multiplier": m, "width_mult": w, "n_price_points": rs[0]["n_price_points"],
                     "n_train_rows_mean": nrows_mean, "accounting": accounting,
                     "fd_equivalent_budget": rs[0]["fd_equivalent_budget"],
                     "raw_scalar_budget": rs[0]["raw_scalar_budget"], "n_seeds": len(rs),
                     "gamma_rel_rmse_mean": g_mean, "gamma_rel_rmse_std": g_std,
                     "gamma_rmse_mean": grmse_mean, "delta_rel_rmse_mean": d_mean,
                     "vega_rel_rmse_mean": v_mean, "price_rel_rmse_mean": p_mean})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="P8 train/val label artifact (.npz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt-root", default=None,
                    help="root for the persisted plateau-m checkpoints "
                         "(<ckpt_root>/info_matched_baseline/s<seed>/best.pt; the shared grid "
                         "root the full sweep reads). Defaults to --out-dir.")
    ap.add_argument("--pinn-cfg", default="pinn_config.yaml")
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (default: global_seed + 0..seeds_min-1)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="override the training step budget")
    ap.add_argument("--multipliers", default=None, help="comma-separated m override")
    ap.add_argument("--base-n", type=int, default=None,
                    help="override N (config default 4096); mostly for tests/small artifacts")
    return ap


def main(argv=None) -> dict:
    args = _build_cli().parse_args(argv)
    contract_d = yaml.safe_load(Path(args.contract).read_text())
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        meta = contract_d["benchmark"]["meta"] if "benchmark" in contract_d else contract_d["meta"]
        g, n = int(meta["global_seed"]), int(meta["seeds_min"])
        seeds = [g + i for i in range(n)]
    mults = [int(m) for m in args.multipliers.split(",")] if args.multipliers else None
    return run_saturation_sweep(
        args.data, seeds, args.out_dir, pinn_cfg=args.pinn_cfg, contract=args.contract,
        device=args.device, steps=args.steps, multipliers=mults,
        base_n_price_points=args.base_n, ckpt_root=args.ckpt_root)


if __name__ == "__main__":
    main()
