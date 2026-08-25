"""lambda_sweep_hedge.py — CODE_AUDIT_2026-08-20 action 2, second half.

Hedges each lambda_pde rung of the sweep at the CONFIRMATORY CELL and emits the
headline contrast as a function of lambda_pde.

WHAT QUESTION THIS ANSWERS. The registered confirmatory contrast (rung3 vs
standard_pinn) is +31.50% at zero cost with lambda_pde = 0.01 and +2.95% with
lambda_pde = 0.0 — an 11x swing driven by a shared hyperparameter that was selected on
a region the headline is not measured in, at the lower boundary of its candidate grid.
Between those two endpoints the contract's grid has no candidates at all. This script
fills the interval and reports the contrast at every rung, so the paper can say how much
of the effect is supervision and how much is where lambda_pde happened to land.

PROTOCOL. Identical to run_hedging.run_confirmatory in every respect except the arm list
(only the two arms the sweep retrained) and the checkpoint root:
  combined perturbation, magnitudes {0.0 (in-model), 1.0 (misspecified)},
  the contract's registered tc tiers, meta.seeds_confirmatory_cell seeds,
  persist_pnl forced True (the pooled paired bootstrap needs per-path PnL),
  the SAME frozen CRN path banks every other run uses.

The two endpoints are NOT re-hedged — they already exist and are read in place:
  lambda_pde = 0.01  -> results/hedging_atc/confirmatory        (registered)
  lambda_pde = 0.0   -> results/hedging_robustness/confirmatory (contract robustness row)

STATUS: sensitivity analysis. The registered lambda_pde stays 0.01, the registered
verdicts are computed at that value only, and nothing here is a re-scoring.

    python lambda_sweep_hedge.py --hedge-only     # run the hedging, skip the curve
    python lambda_sweep_hedge.py --curve-only     # rebuild the curve from existing runs
    python lambda_sweep_hedge.py                  # both
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path

import analyze_results as ar
import Hedging_backtest as hb
import run_hedging as rh
from lambda_sweep import ARMS, NEW_LAMBDAS, lam_tag

ENGINE_ARMS = ["standard_pinn", "rung3"]          # engine names for the two swept arms

#: rungs that already have a hedging run on disk — read, never recomputed
EXISTING = {
    0.0: "results/hedging_robustness/confirmatory",
    0.01: "results/hedging_atc/confirmatory",
}


def hedge_one(lam: float, sweep_root: Path, out_root: Path, contract: str,
              engine: str) -> dict:
    """Confirmatory hedging for one lambda_pde rung (resume-safe via run_hedging)."""
    cfg = hb.resolve_config(contract, engine)
    meta = cfg["benchmark"]["meta"]
    g, n_seeds = int(meta["global_seed"]), int(meta["seeds_confirmatory_cell"])
    prog = copy.deepcopy(cfg)
    prog["derived"]["seeds"] = [g + i for i in range(n_seeds)]
    prog["engine"]["risk"]["persist_pnl"] = True
    mis = prog["benchmark"]["hedging_simulation"]["misspecification"]
    mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
    mis["cross_model"] = []
    prog["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]
    ckpt_root = sweep_root / "grid" / lam_tag(lam)
    run_root = str(out_root / lam_tag(lam) / "confirmatory")
    return rh._run_program(prog, ckpt_root, run_root, ENGINE_ARMS, "confirmatory")


def _pnl_dir(run_root: str) -> str:
    return os.path.join(run_root, ar.PNL_DIR)


def contrast(run_root: str, tc: float, slug_filter) -> dict:
    """Pooled-stratified paired CVaR diff rung3 - standard_pinn, plus the seed companion.

    Both conventions, always — the contract requires the pooled statistic with seed
    variance reported beside it (tail_claim_requires), and the 2026-08-18 convention
    audit fixed that reporting them separately is what let a claim rest on whichever
    interval favoured it.
    """
    res = ar.paired_ci_from_npz(_pnl_dir(run_root), "rung3", "standard_pinn", tc,
                                slug_filter=slug_filter)
    p, s = res["pooled"], res["per_seed_summary"]
    n = int(res["n_seeds"])
    half = (1.96 * s["diff_seed_std"] / math.sqrt(n)) if n > 1 else float("nan")
    return {"n_seeds": n,
            "pooled_diff": p["diff"], "pooled_lo": p["ci_lo"], "pooled_hi": p["ci_hi"],
            "pooled_rel": p["rel_improvement"],
            "pooled_excludes_zero": bool(p["ci_lo"] > 0 or p["ci_hi"] < 0),
            "seed_diff": s["diff_mean"], "seed_std": s["diff_seed_std"],
            "seed_lo": s["diff_mean"] - half, "seed_hi": s["diff_mean"] + half,
            "seed_robust": bool(math.isfinite(half)
                                and (s["diff_mean"] - half) * (s["diff_mean"] + half) > 0),
            "seed_rel_mean": s["rel_improvement_mean"],
            "seed_rel_std": s["rel_improvement_seed_std"]}


def build_curve(lams, sweep_out: Path, contract: str, engine: str,
                out_dir: Path) -> list[dict]:
    """One row per (lambda_pde, magnitude, tc): the contrast under both conventions."""
    th = hb.contract_thresholds(hb.resolve_config(contract, engine))
    tiers = [float(t) for t in th["tc_tiers"]]
    rows: list[dict] = []
    for lam in lams:
        run_root = EXISTING.get(lam) or str(sweep_out / lam_tag(lam) / "confirmatory")
        if not os.path.isdir(_pnl_dir(run_root)):
            print(f"  lambda={lam:g}: no PnL bank at {run_root} — skipped")
            continue
        for label, filt in (("misspec", ar._MISSPEC_FILTER), ("in_model", ar._INMODEL_FILTER)):
            for tc in tiers:
                try:
                    c = contrast(run_root, tc, filt)
                except FileNotFoundError:
                    continue
                rows.append({"lambda_pde": lam, "cell": label, "tc": tc,
                             "source": run_root, "registered": lam == 0.01, **c})
        print(f"  lambda={lam:g}: {run_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "headline_vs_lambda_pde.csv"
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p} ({len(rows)} rows)")
    return rows


def render(rows: list[dict], out_dir: Path) -> str:
    """Markdown table: the headline as a function of lambda_pde, both conventions."""
    lams = sorted({r["lambda_pde"] for r in rows})
    L = ["# Headline contrast vs lambda_pde (CODE_AUDIT_2026-08-20 action 2)", "",
         "rung3 - standard_pinn at the confirmatory cell. Negative CVaR diff = rung3 better;",
         "`rel` is the pre-registered relative improvement. `pooled` is the registered",
         "statistic (paired bootstrap over CRN paths); `seed` is its mandatory companion",
         "(does the effect replicate across training runs). SENSITIVITY ANALYSIS — the",
         "registered lambda_pde is 0.01 and no registered verdict reads any other rung.", ""]
    for cell in ("misspec", "in_model"):
        sub = [r for r in rows if r["cell"] == cell]
        if not sub:
            continue
        tiers = sorted({r["tc"] for r in sub})
        L += [f"## {cell} cell", "",
              "| lambda_pde | " + " | ".join(f"tc={t:g}: rel (pooled CI)" for t in tiers)
              + " | seed-robust @tc=0 |",
              "|---" * (len(tiers) + 2) + "|"]
        for lam in lams:
            cells = []
            sr = ""
            for t in tiers:
                r = next((x for x in sub if x["lambda_pde"] == lam and x["tc"] == t), None)
                if r is None:
                    cells.append("—")
                    continue
                mark = "" if r["pooled_excludes_zero"] else " *ns*"
                cells.append(f"{r['pooled_rel']:+.4f} [{r['pooled_lo']:+.3f}, {r['pooled_hi']:+.3f}]{mark}")
                if t == 0.0:
                    sr = "yes" if r["seed_robust"] else "no"
            tag = " **(registered)**" if lam == 0.01 else ""
            L.append(f"| {lam:g}{tag} | " + " | ".join(cells) + f" | {sr} |")
        L.append("")
    p = out_dir / "headline_vs_lambda_pde.md"
    p.write_text("\n".join(L) + "\n")
    print(f"wrote {p}")
    return str(p)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-root", default="results/lambda_sweep")
    ap.add_argument("--out-root", default="results/lambda_sweep/hedging")
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--engine", default="hedging_config.yaml")
    ap.add_argument("--lambdas", default=None)
    ap.add_argument("--hedge-only", action="store_true")
    ap.add_argument("--curve-only", action="store_true")
    args = ap.parse_args(argv)

    new = ([float(x) for x in args.lambdas.split(",")] if args.lambdas
           else list(NEW_LAMBDAS))
    sweep_root, out_root = Path(args.sweep_root), Path(args.out_root)

    if not args.curve_only:
        for lam in new:
            root = sweep_root / "grid" / lam_tag(lam)
            missing = [f"{a}/s{s}" for a in ARMS for s in range(42, 52)
                       if not (root / a / f"s{s}" / "best.pt").exists()]
            if missing:
                print(f"lambda={lam:g}: SKIPPED — {len(missing)} checkpoints missing "
                      f"(first: {missing[0]})")
                continue
            print(f"lambda={lam:g}: hedging {root} ...", flush=True)
            hedge_one(lam, sweep_root, out_root, args.contract, args.engine)

    if args.hedge_only:
        return {}
    all_lams = sorted(set(new) | set(EXISTING))
    rows = build_curve(all_lams, out_root, args.contract, args.engine,
                       Path(args.sweep_root))
    md = render(rows, Path(args.sweep_root)) if rows else None
    json.dump({"lambdas": all_lams, "n_rows": len(rows)},
              open(Path(args.sweep_root) / "curve_manifest.json", "w"), indent=2)
    return {"rows": rows, "markdown": md}


if __name__ == "__main__":
    main()
