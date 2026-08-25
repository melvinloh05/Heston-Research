"""lambda_sweep.py — CODE_AUDIT_2026-08-20 action 2: the lambda_pde sensitivity sweep.

WHY THIS EXISTS. The audit established three facts about the shared PDE weight:

  1. lambda_pde was SELECTED on the hypercube validation split crossed with the FULL
     (S, K, tau) grid, where the residual HELPS (standard_pinn delta RMSE 0.0418 vs
     feedforward 0.0607). The hedging headline is measured in a narrow near-ATM,
     short-tau slice, where the same choice HURTS (0.0989 vs 0.0484). Anchors are
     excised from training AND validation, so the selection is blind by construction
     to the region that decides the headline.
  2. The registered candidate grid [0.0, 0.01, 0.1, 1.0] SELECTED ITS OWN LOWER
     BOUNDARY: 0.01 beat 0.0 by 3.4% on validation (0.1697 vs 0.1757) while everything
     larger degraded sharply. The interval (0, 0.01) was never explored, and that is
     exactly where the 43% hedging swing between the two endpoints lives.
  3. loss_scale_pde = mean((r*price)^2) has rms 0.527 against the residual's own
     dominant term rms(0.5*v*S^2*Gamma) = 4.674 — the normalizer is 8.9x too small in
     rms, 79x in squared units, so a nominal lambda carries ~79x the weight a
     scale-matched normalization would give it.

This module fills the unexplored interval. It is a SENSITIVITY ANALYSIS, not a
re-selection: the registered lambda_pde stays 0.01 and the registered verdicts are
untouched. What it produces is the headline contrast as a FUNCTION of lambda_pde, so
the paper can state how much of the +31.5% zero-cost effect is a property of Sobolev
supervision and how much is a property of where the baseline's shared hyperparameter
happened to land.

PROTOCOL — identical to the production grid in every respect except lambda_pde:
  same data artifact, same pinn_config, same 20000 matched-epoch steps, same
  batch/lr/schedule/grad_clip, same device (cpu — the production grid ran on cpu,
  runlog.compute.wall_clock_s ~ 410s/run), same seeds. Only the `--lambdas` file
  differs, and only in its lambda_pde field; lambda_gamma/lambda_vega/lambda_delta
  keep their selected values so nothing else moves.

SAFETY. Writes ONLY under out_root (default results/lambda_sweep/). Never touches
results/grid, results/grid_robustness, data/frozen, or any registered artifact. The
ledger makes the sweep resumable: an interrupted run is retried, a completed run is
skipped, and a checkpoint is only marked done after it loads.

    python lambda_sweep.py --dry-run          # plan + time estimate, no work
    python lambda_sweep.py --workers 3        # run it
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
DATA = "data/frozen/v6-labels-20260812/train_val/train_val_labels.npz"
BASE_LAMBDAS = "lambdas_selected.yaml"

#: the unexplored interval below the registered 0.01, plus a decade above 0 to
#: bracket it. 0.0 and 0.01 are NOT re-trained — they already exist as
#: results/grid_robustness (lambda_pde=0) and results/grid (lambda_pde=0.01).
NEW_LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3)

#: the confirmatory contrast: baseline arm and treatment arm, both retrained at each
#: lambda_pde (the robustness-row protocol — retraining only one side would compare
#: arms fitted under different objectives).
ARMS = ("standard_pinn", "rung3_delta_gamma_vega")

SEEDS = tuple(range(42, 52))            # meta.seeds_confirmatory_cell = 10

LEDGER_COLS = ["lam_tag", "lambda_pde", "arm", "seed", "status", "wall_clock_s",
               "best_step", "val_total", "returncode", "note"]


def lam_tag(lam: float) -> str:
    """Filesystem-safe tag for a lambda value ('1e-04' -> 'lam1e-04')."""
    return "lam" + f"{lam:g}".replace(".", "p").replace("-", "m")


def write_lambdas_file(lam: float, dest: Path, base: str = BASE_LAMBDAS) -> Path:
    """A lambdas yaml identical to the selected one except lambda_pde.

    Carries a provenance block so a reader of the artifact can see this is a
    sensitivity fit and not a re-selection.
    """
    d = yaml.safe_load(Path(base).read_text())
    d = dict(d)
    d["lambda_pde"] = float(lam)
    d["_provenance"] = {
        "origin": base,
        "modified_field": "lambda_pde",
        "registered_value": yaml.safe_load(Path(base).read_text())["lambda_pde"],
        "sweep_value": float(lam),
        "purpose": "CODE_AUDIT_2026-08-20 action 2 — lambda_pde sensitivity analysis",
        "not_a_reselection": True,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(d, sort_keys=False))
    return dest


def _ledger_done(path: Path) -> dict:
    """{(lam_tag, arm, seed): row} for rows already marked ok."""
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                out[(row["lam_tag"], row["arm"], int(row["seed"]))] = row
    return out


def _append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLS})


def _checkpoint_ok(d: Path) -> bool:
    """A run counts as done only when best.pt LOADS and carries a state_dict."""
    p = d / "best.pt"
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        import torch
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        return "state_dict" in ck and "cfg" in ck
    except Exception:
        return False


def train_one(lam: float, arm: str, seed: int, out_root: Path, lam_file: Path,
              threads: int, device: str = "cpu") -> dict:
    """One matched-epochs fit, mirroring infra/modal_app._train_remote_body."""
    tag = lam_tag(lam)
    dest = out_root / "grid" / tag / arm / f"s{seed}"
    if _checkpoint_ok(dest):
        return {"lam_tag": tag, "lambda_pde": lam, "arm": arm, "seed": seed,
                "status": "ok", "note": "already present", "returncode": 0}
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    cmd = [sys.executable, "train.py", "--arm", arm, "--seed", str(seed),
           "--data", DATA, "--lambdas", str(lam_file), "--out", str(dest),
           "--device", device, "--matched-epochs"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    ok = proc.returncode == 0 and _checkpoint_ok(dest)
    row = {"lam_tag": tag, "lambda_pde": lam, "arm": arm, "seed": seed,
           "status": "ok" if ok else "failed", "wall_clock_s": round(wall, 1),
           "returncode": proc.returncode,
           "note": "" if ok else proc.stderr.strip().splitlines()[-1][:300] if proc.stderr.strip() else "no best.pt"}
    rl = dest / "runlog.json"
    if rl.exists():
        try:
            r = json.loads(rl.read_text())
            row["best_step"] = r.get("checkpoints", {}).get("best", {}).get("step", "")
            row["val_total"] = r.get("checkpoints", {}).get("best", {}).get("val_total", "")
        except Exception:
            pass
    return row


def build_plan(lambdas=NEW_LAMBDAS, arms=ARMS, seeds=SEEDS) -> list:
    return [(lam, arm, seed) for lam in lambdas for arm in arms for seed in seeds]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default="results/lambda_sweep")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent fits; each gets --threads BLAS threads")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lambdas", default=None,
                    help="comma-separated lambda_pde values (default: the audit's grid)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds")
    ap.add_argument("--minutes-per-run", type=float, default=6.9,
                    help="production runlog wall_clock_s ~ 410s")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    lams = ([float(x) for x in args.lambdas.split(",")] if args.lambdas
            else list(NEW_LAMBDAS))
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS))
    out_root = Path(args.out_root)
    ledger = out_root / "train_ledger.csv"
    plan = build_plan(lams, ARMS, seeds)
    done = _ledger_done(ledger)
    todo = [p for p in plan if (lam_tag(p[0]), p[1], p[2]) not in done]

    est_h = len(todo) * args.minutes_per_run / 60.0 / max(1, args.workers)
    print(f"lambda_pde sweep: {len(lams)} lambdas x {len(ARMS)} arms x {len(seeds)} seeds "
          f"= {len(plan)} fits; {len(done)} already done, {len(todo)} to run")
    print(f"  lambdas : {lams}")
    print(f"  arms    : {list(ARMS)}")
    print(f"  seeds   : {seeds[0]}..{seeds[-1]}")
    print(f"  out     : {out_root}/grid/<lam_tag>/<arm>/s<seed>/best.pt")
    print(f"  estimate: ~{est_h:.1f} h wall at {args.workers} workers "
          f"({args.minutes_per_run:.1f} min/fit)")
    if args.dry_run:
        print("DRY RUN — nothing executed.")
        return {"plan": plan, "todo": todo, "est_hours": est_h}

    lam_files = {lam: write_lambdas_file(lam, out_root / "lambdas" / f"{lam_tag(lam)}.yaml")
                 for lam in lams}
    t0 = time.perf_counter()
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(train_one, lam, arm, seed, out_root, lam_files[lam],
                          args.threads, args.device): (lam, arm, seed)
                for lam, arm, seed in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            _append_ledger(ledger, row)
            n_ok += row["status"] == "ok"
            n_fail += row["status"] != "ok"
            el = (time.perf_counter() - t0) / 60.0
            eta = el / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {row['lam_tag']:>9} {row['arm']:<24} s{row['seed']} "
                  f"{row['status']:<7} {row.get('wall_clock_s','')}s | elapsed {el:.0f}m eta {eta:.0f}m",
                  flush=True)
    print(f"DONE: {n_ok} ok, {n_fail} failed in {(time.perf_counter()-t0)/60:.1f} min")
    return {"ok": n_ok, "failed": n_fail, "ledger": str(ledger)}


if __name__ == "__main__":
    main()
