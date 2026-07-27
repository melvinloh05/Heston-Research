"""infra/digest.py — nightly one-page digest over the grid + hedging ledger.

Reads results/grid/**/runlog.json (per-arm/seed training runs, written by
train.py) and results/hedging/run_ledger.csv (per-cell hedging status, written
by run_hedging.py) and writes a single markdown page: one row per training run
(arm, seed, status, headline loss, wall clock, anomalies) and one section for
the hedging cells, with an anomaly rollup at the top.

Anomalies flagged:
  loss_nan            a NaN in the best-checkpoint val_total or any loss curve entry
  early_stop_step<N   early-stopped before EARLY_STOP_STEP_FLOOR (=1000) steps
  missing_checkpoint  runlog present but best.pt is absent in the run dir

Pure over the filesystem; no network, no secrets.

    python infra/digest.py                               # -> results/digest.md
    python infra/digest.py --grid-root results/grid \
        --ledger results/hedging/run_ledger.csv --out results/digest.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

GRID_ROOT = "results/grid"
LEDGER_PATH = "results/hedging/run_ledger.csv"
DIGEST_OUT = "results/digest.md"
EARLY_STOP_STEP_FLOOR = 1000

# ---------------------------------------------------------------------------
# anomaly detection
# ---------------------------------------------------------------------------

def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _runlog_has_nan(runlog: dict) -> bool:
    """True if the best-checkpoint total or any loss-curve total/term is NaN."""
    if _is_nan(runlog.get("checkpoints", {}).get("best", {}).get("val_total")):
        return True
    for entry in runlog.get("val_curve", []) or []:
        if _is_nan(entry.get("val_total")) or _is_nan(entry.get("train_total")):
            return True
        for terms in (entry.get("val_terms"), entry.get("train_terms")):
            if any(_is_nan(v) for v in (terms or {}).values()):
                return True
    return False


def detect_anomalies(runlog: dict, run_dir: str | Path) -> list[str]:
    """The three anomaly tags for one run (empty list == clean)."""
    anomalies: list[str] = []
    if _runlog_has_nan(runlog):
        anomalies.append("loss_nan")
    comp = runlog.get("compute", {}) or {}
    steps = comp.get("steps")
    if comp.get("stopped_early") and isinstance(steps, int) and steps < EARLY_STOP_STEP_FLOOR:
        anomalies.append(f"early_stop_step{steps}")
    best_rel = runlog.get("checkpoints", {}).get("best", {}).get("path", "best.pt")
    if not (Path(run_dir) / best_rel).exists():
        anomalies.append("missing_checkpoint")
    return anomalies

# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------

def scan_grid(grid_root: str | Path = GRID_ROOT) -> list[dict]:
    """One record per results/grid/**/runlog.json (unreadable logs are surfaced)."""
    records: list[dict] = []
    root = Path(grid_root)
    for rl_path in sorted(root.glob("**/runlog.json")):
        run_dir = rl_path.parent
        try:
            runlog = json.loads(rl_path.read_text())
        except (json.JSONDecodeError, OSError):
            records.append({"arm": run_dir.parent.name, "seed": None,
                            "status": "unreadable", "headline_loss": None,
                            "wall_clock_s": None, "anomalies": ["unreadable_runlog"],
                            "run_dir": str(run_dir)})
            continue
        anomalies = detect_anomalies(runlog, run_dir)
        records.append({
            "arm": runlog.get("arm", run_dir.parent.name),
            "seed": runlog.get("seed"),
            "status": "anomaly" if anomalies else "ok",
            "headline_loss": runlog.get("checkpoints", {}).get("best", {}).get("val_total"),
            "wall_clock_s": (runlog.get("compute", {}) or {}).get("wall_clock_s"),
            "anomalies": anomalies,
            "run_dir": str(run_dir)})
    return records


def scan_ledger(ledger_path: str | Path = LEDGER_PATH) -> list[dict]:
    """Per-cell hedging rows from run_ledger.csv (empty list if absent)."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return [{"run": row.get("run"), "cell": row.get("cell"), "seed": row.get("seed"),
                 "status": row.get("status"), "wall_clock_s": row.get("wall_clock_s"),
                 "n_rows": row.get("n_rows")}
                for row in csv.DictReader(fh)]

# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt(x, sig: int = 4) -> str:
    if x is None or x == "":
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "NaN"
    return f"{xf:.{sig}g}"


def render_digest(grid_records: list[dict], ledger_records: list[dict],
                  now: datetime | None = None) -> str:
    """Render the one-page markdown digest from scanned records."""
    now = now or datetime.now(timezone.utc)
    flagged = [r for r in grid_records if r["anomalies"]]
    lines = [f"# Nightly digest — {now:%Y-%m-%d %H:%M UTC}", "",
             f"**Grid runs:** {len(grid_records)} &middot; "
             f"**with anomalies:** {len(flagged)} &middot; "
             f"**hedging cells:** {len(ledger_records)}", ""]

    # anomaly rollup
    if flagged:
        lines += ["## ⚠ Anomalies", ""]
        for r in flagged:
            seed = r["seed"] if r["seed"] is not None else "—"
            lines.append(f"- `{r['arm']}` s{seed}: {', '.join(r['anomalies'])}")
        lines.append("")
    else:
        lines += ["_No training-run anomalies._", ""]

    # grid table — anomalous runs first, then arm/seed order
    lines += ["## Grid training runs", "",
              "| arm | seed | status | headline loss | wall clock (s) | anomalies |",
              "|---|---|---|---|---|---|"]
    for r in sorted(grid_records,
                    key=lambda r: (not r["anomalies"], r["arm"], str(r["seed"]))):
        seed = r["seed"] if r["seed"] is not None else "—"
        anoms = ", ".join(r["anomalies"]) if r["anomalies"] else "—"
        lines.append(f"| {r['arm']} | {seed} | {r['status']} | "
                     f"{_fmt(r['headline_loss'])} | {_fmt(r['wall_clock_s'])} | {anoms} |")
    lines.append("")

    # hedging cells
    lines += ["## Hedging cells (run_ledger)", ""]
    if ledger_records:
        lines += ["| run | cell | seed | status | wall clock (s) | n_rows |",
                  "|---|---|---|---|---|---|"]
        for r in ledger_records:
            lines.append(f"| {r['run']} | {r['cell']} | {r['seed']} | {r['status']} | "
                         f"{_fmt(r['wall_clock_s'])} | {r['n_rows']} |")
    else:
        lines.append("_No hedging ledger found._")
    lines.append("")
    return "\n".join(lines)


def build_digest(grid_root: str | Path = GRID_ROOT,
                 ledger_path: str | Path = LEDGER_PATH,
                 now: datetime | None = None) -> str:
    """Scan both sources and return the rendered digest markdown."""
    return render_digest(scan_grid(grid_root), scan_ledger(ledger_path), now=now)


def write_digest(out_path: str | Path = DIGEST_OUT, grid_root: str | Path = GRID_ROOT,
                 ledger_path: str | Path = LEDGER_PATH,
                 now: datetime | None = None) -> str:
    """Write the digest markdown to out_path and return the text."""
    md = build_digest(grid_root, ledger_path, now=now)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    return md

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> str:
    ap = argparse.ArgumentParser(description="nightly grid + hedging digest")
    ap.add_argument("--grid-root", default=GRID_ROOT)
    ap.add_argument("--ledger", default=LEDGER_PATH)
    ap.add_argument("--out", default=DIGEST_OUT)
    args = ap.parse_args(argv)
    write_digest(args.out, args.grid_root, args.ledger)
    print(f"wrote {args.out}")
    return args.out


if __name__ == "__main__":
    main()
