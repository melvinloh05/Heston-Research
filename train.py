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
--select-lambdas runs the VALIDATION-ONLY STAGED lambda search instead of a training run:
lambda_pde on the contract's `lambda_selection.lambda_pde.source_arm` (the baseline), then
(lambda_gamma, lambda_vega) on the rung3 source arm at that fixed lambda_pde. The emitted
lambdas_selected.yaml records each value's SOURCE ARM (contract Q3).
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
    """VALIDATION-ONLY STAGED lambda search -> lambdas_selected.yaml (contract Q3).

    The contract's `lambda_selection` section pre-registers WHICH ARM each shared lambda
    is sourced from; this function implements exactly that, in two stages:

      stage 1  lambda_pde, 1-D over its candidates, scored on
               `lambda_selection.lambda_pde.source_arm` (standard_pinn) — the arm for
               which the PDE residual is the ONLY structural signal. Selecting it on the
               treatment arm and then applying it to the baseline handicaps the baseline;
               sourcing it here is the conservative direction wrt the hypothesis.
      stage 2  (lambda_gamma, lambda_vega), 2-D at the stage-1 lambda_pde, scored on
               `lambda_selection.lambda_{gamma,vega}.source_arm` (rung3).

    SHARING lambda_pde across arms is unchanged and deliberate (one model class, identical
    architecture/ansatz — see methods.ansatz_control); only its SOURCE moved. Both stages
    touch train/val only, both run behind the same LockedTestSet, and both SELECTED values
    are written with their SOURCE ARM so provenance lives in the artifact rather than in a
    reader's memory of which arm train.py happened to default to.
    """
    pinn_raw = yaml.safe_load(open(args.pinn_cfg))
    contract = yaml.safe_load(Path(args.contract).read_text())
    lam_sel = contract["lambda_selection"]
    pde_arm = str(lam_sel["lambda_pde"]["source_arm"])
    gv_arm = str(lam_sel["lambda_gamma"]["source_arm"])
    if gv_arm != str(lam_sel["lambda_vega"]["source_arm"]):
        raise ValueError("lambda_gamma and lambda_vega must share a source arm; the 2-D "
                         "stage fits ONE model per (gamma, vega) combo")
    tcfg = replace(TrainConfig.from_dict(pinn_raw.get("training", {})),
                   steps=args.steps if args.steps is not None else 4000)
    ranges, anchors, feller_min, excise = pde_sampling_spec(args.contract, args.pinn_cfg)
    guard = LockedTestSet(args.anchor_grids or "<<held-out anchor grids>>")
    cpde = _parse_floats(args.cand_pde) or lam_sel["lambda_pde"]["candidates"]
    cg = _parse_floats(args.cand_gamma) or lam_sel["lambda_gamma"]["candidates"]
    cv = _parse_floats(args.cand_vega) or lam_sel["lambda_vega"]["candidates"]

    def _arm_data(arm: str):
        cfg = load_arm(args.pinn_cfg, arm)
        return (cfg, ArmDataset(args.data, cfg, "train", seed=args.seed),
                ArmDataset(args.data, cfg, "val", seed=args.seed))

    def _score(cfg, train_ds, val_ds) -> float:
        model, best_state, _, _ = train_model(
            cfg, train_ds, val_ds, tcfg, args.seed, device=args.device,
            pde_ranges=ranges, pde_anchors=anchors, feller_min=feller_min,
            excise_rel_radius=excise, early_stop=False)
        model.load_state_dict(best_state)
        return _val_greek_score(model, val_ds, args.device)

    # ---- stage 1: lambda_pde on the BASELINE arm (gamma/vega are OFF there) ----
    pde_cfg, pde_train, pde_val = _arm_data(pde_arm)

    def score_pde(lp: float, _lg: float, _lv: float) -> float:
        return _score(replace(pde_cfg, lambda_pde=lp), pde_train, pde_val)

    stage1 = select_lambdas(cpde, [1.0], [1.0], fit_and_val_score=score_pde,
                            test_set=guard)
    lambda_pde = float(stage1["lambda_pde"])

    # ---- stage 2: (lambda_gamma, lambda_vega) on rung3 at that fixed lambda_pde ----
    gv_cfg, gv_train, gv_val = _arm_data(gv_arm)

    def score_gv(lp: float, lg: float, lv: float) -> float:
        return _score(replace(gv_cfg, lambda_pde=lp, lambda_gamma=lg, lambda_vega=lv),
                      gv_train, gv_val)

    stage2 = select_lambdas([lambda_pde], cg, cv, fit_and_val_score=score_gv,
                            test_set=guard)

    chash = config_hash(gv_cfg, args.seed, data_manifest_sha(args.data), None)
    res = {
        "lambda_pde": lambda_pde,
        "lambda_gamma": float(stage2["lambda_gamma"]),
        "lambda_vega": float(stage2["lambda_vega"]),
        "lambda_delta": float(stage2["lambda_delta"]),
        # provenance: WHICH ARM each value was selected on (contract lambda_selection)
        "sources": {"lambda_pde": pde_arm, "lambda_gamma": gv_arm,
                    "lambda_vega": gv_arm, "lambda_delta": "fixed_not_swept"},
        "selection": {
            "protocol": "staged: 1-D lambda_pde on the source arm, then 2-D "
                        "(lambda_gamma, lambda_vega) at that fixed lambda_pde",
            "tune_on": str(lam_sel.get("tune_on", "validation_only")),
            "candidates": {"lambda_pde": [float(x) for x in cpde],
                           "lambda_gamma": [float(x) for x in cg],
                           "lambda_vega": [float(x) for x in cv]},
            "seed": int(args.seed), "steps": int(tcfg.steps)},
        # stage 1 varied ONLY lambda_pde; the joint grid's placeholder columns are dropped
        "scores_table_pde": [{"lambda_pde": r["lambda_pde"], "score": r["score"]}
                             for r in stage1["scores_table"]],
        "scores_table": stage2["scores_table"],
        "config_hash": chash,
    }
    if args.out:
        Path(args.out).write_text(yaml.safe_dump(res, sort_keys=False))
    print(f"selected lambda_pde={res['lambda_pde']} (on {pde_arm}, "
          f"{len(cpde)} candidates) lambda_gamma={res['lambda_gamma']} "
          f"lambda_vega={res['lambda_vega']} (on {gv_arm}, "
          f"{len(res['scores_table'])} combos) (lambda_delta fixed 1.0); "
          f"validation only")
    if args.out:
        print(f"wrote {args.out}")
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
    ap.add_argument("--show-prefix-bug", action="store_true",
                    help="also print the deliberately-reproduced pre-fix sigma_gamma_pilot "
                         "(last-step model, arm label). It is ALWAYS in the runlog; it is off "
                         "stdout by default because it is the wrong number to feed the gate")
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
        # It is KEPT OUT of default stdout (audit T1): it is computed on an unconverged model
        # and so is typically LARGER, and feeding it to the gate would make the gate look more
        # favourable than it is. Default stdout carries only the correct value; the wrong one
        # lives in the runlog under an unmistakable key and behind --show-prefix-bug.
        sigma_pre, rel_pre, src_pre = _pilot_gamma_rmse(model, val_ds, args.device,
                                                         prefer_ref=False)
        model.load_state_dict(best_state)
        sigma, rel, src = _pilot_gamma_rmse(model, val_ds, args.device, prefer_ref=True)
        runlog["sigma_gamma_pilot"] = sigma
        runlog["sigma_gamma_pilot_relative"] = rel
        runlog["sigma_gamma_pilot_source"] = src
        runlog["sigma_gamma_pilot_prefix_bug"] = sigma_pre
        runlog["sigma_gamma_pilot_prefix_bug_relative"] = rel_pre
        if args.show_prefix_bug:
            print(f"sigma_gamma_pilot BEFORE fix (last-step model, {src_pre}) = "
                  f"{sigma_pre:.6g} (relative {rel_pre:.6g})  [DO NOT FEED TO THE GATE]")
        print(f"sigma_gamma_pilot AFTER  fix (best-step model, {src}) = "
              f"{sigma:.6g} (relative {rel:.6g})")
        print("hand off to the gate WITHOUT retyping it:\n"
              f"    python gate_headroom.py --sigma-gamma-from-runlog {out/'runlog.json'}")

    _write_loss_curve_csv(out / "loss_curves.csv", runlog["val_curve"])
    (out / "runlog.json").write_text(json.dumps(runlog, indent=2, default=str))
    ck = runlog["checkpoints"]
    # 'matched_epochs' is present only when the step budget was reached (audit T2);
    # say which of the two this last.pt actually is.
    last_label = "matched_epochs" if "matched_epochs" in ck else "last, EARLY-STOPPED"
    print(f"wrote {out/'best.pt'} (best step {ck['best']['step']}), "
          f"{out/'last.pt'} ({last_label} step {ck['last']['step']}), "
          f"{out/'runlog.json'}, {out/'loss_curves.csv'}")
    return runlog


if __name__ == "__main__":
    main()
