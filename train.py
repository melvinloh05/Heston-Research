"""train.py — single-arm training entry for the v6 Sobolev-PINN benchmark.

    python train.py --arm rung3_delta_gamma_vega --seed 42 --data <train_val_labels.npz>
        --pinn-cfg pinn_config.yaml --lambdas lambdas_selected.yaml
        --out results/grid/<arm>/s42 [--device cuda|cpu] [--steps N] [--matched-epochs]

One arm = one PINNConfig (SobolevPINN); the loss is assembled from flags only. Data comes
from the P8 train/val label artifact via train_pinn.ArmDataset (split 0=train, 1=val).
lambdas_selected.yaml (when present) overrides the swept lambdas; a missing file warns and
uses the pinn-config defaults. Every run writes:

    best.pt          best-validation checkpoint (state_dict + cfg dict + frozen scales)
    last.pt          matched-epochs checkpoint (the fixed-step model; report_both)
    runlog.json      config hash, compute accounting, per-term loss curve, checkpoints
    loss_curves.csv  per-term train/val losses at each validation check

--pilot runs a SHORT fit (steps=2000 unless --steps overrides) and prints
`sigma_gamma_pilot` — the oracle-headroom-gate input (PINN gamma error scale).
--select-lambdas runs the VALIDATION-ONLY joint lambda search instead of a training run.
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from SobolevPINN import PINNConfig, load_arm
from train_pinn import (ArmDataset, LockedTestSet, TrainConfig, config_hash,
                        data_manifest_sha, file_sha256, pde_sampling_spec,
                        select_lambdas, train_model)

# per-term loss columns for the loss-curve CSV (superset; absent terms stay blank)
_TERM_COLS = ["price", "pde", "delta", "gamma", "gamma_penalty", "vega", "vanna", "bc"]


def _apply_lambdas(cfg: PINNConfig, lam: dict) -> PINNConfig:
    """Override the SWEPT lambdas from lambdas_selected.yaml, preserving OFF cells.

    lambda_pde is overridden only for arms that actually use the residual (use_pde and
    lambda_pde != 0) so the Sakuma/DML-no-PDE arms keep their pinned 0; lambda_gamma only
    when gamma is supervised or penalized; lambda_vega only when vega is supervised.
    lambda_delta is left at its config value (fixed 1.0; not a swept axis).
    """
    upd: dict[str, float] = {}
    if "lambda_pde" in lam and cfg.use_pde and cfg.lambda_pde != 0.0:
        upd["lambda_pde"] = float(lam["lambda_pde"])
    if "lambda_gamma" in lam and (cfg.supervise_gamma or cfg.gradient_penalty):
        upd["lambda_gamma"] = float(lam["lambda_gamma"])
    if "lambda_vega" in lam and cfg.supervise_vega:
        upd["lambda_vega"] = float(lam["lambda_vega"])
    return replace(cfg, **upd) if upd else cfg


def _resolve_steps(args, tcfg: TrainConfig) -> int:
    """--steps overrides everything; --pilot defaults to 2000; else the config's steps."""
    if args.steps is not None:
        return int(args.steps)
    if args.pilot:
        return 2000
    return tcfg.steps


def _write_loss_curve_csv(path: Path, val_curve: list[dict]) -> None:
    """Per-term train/val losses at each validation check (two rows per check)."""
    cols = ["step", "split", "total", "lr"] + _TERM_COLS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in val_curve:
            for split, terms in (("train", e["train_terms"]), ("val", e["val_terms"])):
                row = {"step": e["step"], "split": split,
                       "total": terms.get("total"), "lr": e["lr"]}
                row.update({k: terms.get(k) for k in _TERM_COLS})
                w.writerow(row)


def _gamma_truth(val_ds: ArmDataset, prefer_ref: bool) -> tuple[torch.Tensor, str]:
    """Ground truth for the pilot gamma RMSE.

    gamma_ref is the frozen true-consensus label (identical across dose-response arms);
    val_ds.data["gamma"] is the arm's OWN label, which is deliberately corrupted for the
    sigma_* noise arms, so it must never be preferred over gamma_ref when both are present.
    prefer_ref=False reproduces the old (buggy) priority, kept only for before/after logging.
    """
    has_ref = "gamma_ref" in val_ds.data
    has_label = "gamma" in val_ds.data
    if prefer_ref:
        return (val_ds.data["gamma_ref"], "gamma_ref") if has_ref \
            else (val_ds.data["gamma"], "gamma (arm label; gamma_ref absent)")
    return (val_ds.data["gamma"], "gamma (arm label)") if has_label \
        else (val_ds.data["gamma_ref"], "gamma_ref (arm label absent)")


def _pilot_gamma_rmse(model, val_ds: ArmDataset, device: str,
                       prefer_ref: bool = True) -> tuple[float, float, str]:
    """Relative gamma RMSE of the fitted model vs ground-truth gamma (gamma_ref by default)."""
    x = val_ds.data["x"].to(device)
    gh = model.greeks_eval(x, need=("gamma",))["gamma"].detach().cpu()
    true, source = _gamma_truth(val_ds, prefer_ref)
    rmse = float(torch.sqrt(torch.mean((gh - true) ** 2)))
    rms = float(torch.sqrt(torch.mean(true ** 2)))
    return rmse, (rmse / rms if rms > 0.0 else float("nan")), source


def _val_greek_score(model, val_ds: ArmDataset, device: str) -> float:
    """Mean normalized validation RMSE over price/delta/gamma/vega (lambda-selection score)."""
    x = val_ds.data["x"].to(device)
    g = model.greeks_eval(x, need=("delta", "gamma", "vega"))
    scores = []
    for q in ("price", "delta", "gamma", "vega"):
        pred = g[q].detach().cpu()
        true = val_ds.data["gamma_ref"] if q == "gamma" else val_ds.data[q]
        rms = float(torch.sqrt(torch.mean(true ** 2))) or 1.0
        scores.append(float(torch.sqrt(torch.mean((pred - true) ** 2))) / rms)
    return float(np.mean(scores))


def _parse_floats(s: str | None) -> list[float] | None:
    return [float(x) for x in s.split(",")] if s else None


def _run_select_lambdas(args) -> dict:
    """VALIDATION-ONLY joint (lambda_pde, lambda_gamma, lambda_vega) search -> yaml."""
    pinn_raw = yaml.safe_load(open(args.pinn_cfg))
    tcfg = replace(TrainConfig.from_dict(pinn_raw.get("training", {})),
                   steps=args.steps if args.steps is not None else 4000)
    ranges, anchors, feller_min, excise = pde_sampling_spec(args.contract, args.pinn_cfg)
    base = load_arm(args.pinn_cfg, "rung3_delta_gamma_vega")
    train_ds = ArmDataset(args.data, base, "train", seed=args.seed)
    val_ds = ArmDataset(args.data, base, "val", seed=args.seed)
    guard = LockedTestSet(args.anchor_grids or "<<held-out anchor grids>>")
    cpde = _parse_floats(args.cand_pde) or pinn_raw.get("sweeps", {}).get(
        "lambda_pde", [0.0, 0.01, 0.1, 1.0])
    cg = _parse_floats(args.cand_gamma) or [0.3, 1.0, 3.0]   # playbook pre-reg {0.3,1,3}
    cv = _parse_floats(args.cand_vega) or [0.3, 1.0, 3.0]     # playbook pre-reg {0.3,1,3}

    def fit_and_val_score(lp: float, lg: float, lv: float) -> float:
        cfg = replace(base, lambda_pde=lp, lambda_gamma=lg, lambda_vega=lv)
        model, best_state, _, _ = train_model(
            cfg, train_ds, val_ds, tcfg, args.seed, device=args.device,
            pde_ranges=ranges, pde_anchors=anchors, feller_min=feller_min,
            excise_rel_radius=excise, early_stop=False)
        model.load_state_dict(best_state)
        return _val_greek_score(model, val_ds, args.device)

    chash = config_hash(base, args.seed, data_manifest_sha(args.data), None)
    out_path = str(args.out) if args.out else None
    res = select_lambdas(cpde, cg, cv, fit_and_val_score=fit_and_val_score,
                         test_set=guard, out_path=out_path, config_hash=chash)
    print(f"selected lambda_pde={res['lambda_pde']} lambda_gamma={res['lambda_gamma']} "
          f"lambda_vega={res['lambda_vega']} (lambda_delta fixed 1.0); "
          f"scored {len(res['scores_table'])} combos on validation")
    if out_path:
        print(f"wrote {out_path}")
    return res


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="rung3_delta_gamma_vega")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", required=True, help="P8 train/val label artifact (.npz)")
    ap.add_argument("--pinn-cfg", default="pinn_config.yaml")
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--lambdas", default="lambdas_selected.yaml")
    ap.add_argument("--out", required=True, help="output dir (training) / yaml (--select-lambdas)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="override the config step budget")
    ap.add_argument("--matched-epochs", action="store_true",
                    help="disable early stop; run the full fixed-step matched-epochs budget")
    ap.add_argument("--pilot", action="store_true",
                    help="short fit; print sigma_gamma_pilot (oracle-headroom-gate input)")
    ap.add_argument("--select-lambdas", action="store_true",
                    help="run VALIDATION-ONLY joint lambda selection instead of training")
    ap.add_argument("--anchor-grids", default=None, help="held-out anchor grids (kept locked)")
    ap.add_argument("--cand-pde", default=None, help="comma-separated lambda_pde candidates")
    ap.add_argument("--cand-gamma", default=None, help="comma-separated lambda_gamma candidates")
    ap.add_argument("--cand-vega", default=None, help="comma-separated lambda_vega candidates")
    return ap


def main(argv=None) -> dict:
    args = _build_parser().parse_args(argv)
    if args.select_lambdas:
        return _run_select_lambdas(args)

    cfg = load_arm(args.pinn_cfg, args.arm)
    lam_path = Path(args.lambdas)
    lam_sha = None
    if lam_path.exists():
        lam = yaml.safe_load(lam_path.read_text()) or {}
        cfg = _apply_lambdas(cfg, lam)
        lam_sha = file_sha256(args.lambdas)
    else:
        warnings.warn(f"lambdas file {args.lambdas} not found; using pinn-config defaults "
                      "(lambda_pde/gamma/vega unchanged)", RuntimeWarning)

    pinn_raw = yaml.safe_load(open(args.pinn_cfg))
    tcfg = TrainConfig.from_dict(pinn_raw.get("training", {}))
    tcfg = replace(tcfg, steps=_resolve_steps(args, tcfg))
    train_ds = ArmDataset(args.data, cfg, "train", seed=args.seed)
    val_ds = ArmDataset(args.data, cfg, "val", seed=args.seed)
    ranges, anchors, feller_min, excise = pde_sampling_spec(args.contract, args.pinn_cfg)

    model, best_state, last_state, runlog = train_model(
        cfg, train_ds, val_ds, tcfg, args.seed, device=args.device,
        pde_ranges=ranges, pde_anchors=anchors, feller_min=feller_min,
        excise_rel_radius=excise, early_stop=not args.matched_epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dsha = data_manifest_sha(args.data)
    chash = config_hash(cfg, args.seed, dsha, lam_sha)
    scales = {k: v for k, v in best_state.items() if k.startswith("loss_scale")}
    cfg_dict = asdict(cfg)
    for name, state in (("best.pt", best_state), ("last.pt", last_state)):
        torch.save({"state_dict": state, "cfg": cfg_dict, "scales": scales,
                    "config_hash": chash, "seed": args.seed, "arm": args.arm}, out / name)

    runlog.update({
        "arm": args.arm, "seed": args.seed, "device": args.device,
        "config_hash": chash, "cfg": cfg_dict,
        "data": {"path": args.data, "manifest_sha256": dsha},
        "lambdas": {"path": args.lambdas if lam_path.exists() else None, "sha256": lam_sha},
        "training": asdict(tcfg), "matched_epochs_mode": bool(args.matched_epochs)})

    if args.pilot:
        # BEFORE: reproduces the prior bug exactly (last-step model, arm-label priority) so the
        # magnitude of the correction is visible before rerunning any gate decision on it.
        sigma_pre, rel_pre, src_pre = _pilot_gamma_rmse(model, val_ds, args.device,
                                                         prefer_ref=False)
        model.load_state_dict(best_state)
        sigma, rel, src = _pilot_gamma_rmse(model, val_ds, args.device, prefer_ref=True)
        runlog["sigma_gamma_pilot"] = sigma
        runlog["sigma_gamma_pilot_relative"] = rel
        runlog["sigma_gamma_pilot_source"] = src
        print(f"sigma_gamma_pilot BEFORE fix (last-step model, {src_pre}) = "
              f"{sigma_pre:.6g} (relative {rel_pre:.6g})")
        print(f"sigma_gamma_pilot AFTER  fix (best-step model, {src}) = "
              f"{sigma:.6g} (relative {rel:.6g})")

    _write_loss_curve_csv(out / "loss_curves.csv", runlog["val_curve"])
    (out / "runlog.json").write_text(json.dumps(runlog, indent=2, default=str))
    ck = runlog["checkpoints"]
    print(f"wrote {out/'best.pt'} (best step {ck['best']['step']}), "
          f"{out/'last.pt'} (matched_epochs step {ck['matched_epochs']['step']}), "
          f"{out/'runlog.json'}, {out/'loss_curves.csv'}")
    return runlog


if __name__ == "__main__":
    main()
