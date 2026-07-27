"""Tests for infra/digest.py — the digest over synthetic runlogs must flag each
of the three anomaly types (loss NaN / early-stop at step<1000 / missing
checkpoint) and leave a clean run unflagged, plus surface the hedging ledger.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from infra import digest


def _write_runlog(run_dir: Path, runlog: dict, with_best: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runlog.json").write_text(json.dumps(runlog))
    if with_best:
        (run_dir / "best.pt").write_bytes(b"stub")


def _runlog(arm: str, seed: int, *, best_val=0.01, steps=8000,
            stopped_early=False, val_curve=None) -> dict:
    return {
        "arm": arm, "seed": seed,
        "checkpoints": {"best": {"step": 4000, "val_total": best_val, "path": "best.pt"},
                        "matched_epochs": {"step": steps, "path": "last.pt"}},
        "compute": {"wall_clock_s": 123.4, "steps": steps, "stopped_early": stopped_early},
        "val_curve": val_curve if val_curve is not None else
        [{"step": 4000, "val_total": best_val, "train_total": best_val,
          "val_terms": {"total": best_val}, "train_terms": {"total": best_val}}],
    }


def _make_grid(root: Path) -> None:
    # clean
    _write_runlog(root / "rung3_delta_gamma_vega" / "s42",
                  _runlog("rung3_delta_gamma_vega", 42))
    # loss NaN (in the best checkpoint and the curve)
    nan = float("nan")
    _write_runlog(root / "rung2_delta_gamma" / "s42",
                  _runlog("rung2_delta_gamma", 42, best_val=nan,
                          val_curve=[{"step": 500, "val_total": nan, "train_total": nan,
                                      "val_terms": {"total": nan}, "train_terms": {}}]))
    # early stop before 1000 steps
    _write_runlog(root / "rung1_delta" / "s42",
                  _runlog("rung1_delta", 42, steps=750, stopped_early=True))
    # missing checkpoint (runlog present, no best.pt)
    _write_runlog(root / "standard_pinn" / "s42",
                  _runlog("standard_pinn", 42), with_best=False)


def _make_ledger(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "cell", "seed", "wall_clock_s",
                                           "n_rows", "status"])
        w.writeheader()
        w.writerow({"run": "confirmatory", "cell": "combined_m1.0", "seed": 42,
                    "wall_clock_s": 12.5, "n_rows": 30, "status": "ok"})


def test_scan_grid_flags_each_anomaly(tmp_path):
    root = tmp_path / "grid"
    _make_grid(root)
    records = {(r["arm"], r["seed"]): r for r in digest.scan_grid(root)}

    assert records[("rung3_delta_gamma_vega", 42)]["anomalies"] == []
    assert records[("rung3_delta_gamma_vega", 42)]["status"] == "ok"

    assert "loss_nan" in records[("rung2_delta_gamma", 42)]["anomalies"]

    early = records[("rung1_delta", 42)]["anomalies"]
    assert any(a.startswith("early_stop_step") for a in early)
    assert "early_stop_step750" in early

    assert "missing_checkpoint" in records[("standard_pinn", 42)]["anomalies"]


def test_clean_run_not_flagged_for_normal_early_stop(tmp_path):
    root = tmp_path / "grid"
    # early-stopped but well past the 1000-step floor -> not an anomaly
    _write_runlog(root / "rung0_price_only" / "s42",
                  _runlog("rung0_price_only", 42, steps=6000, stopped_early=True))
    rec = digest.scan_grid(root)[0]
    assert rec["anomalies"] == []


def test_headline_loss_and_wall_clock_reported(tmp_path):
    root = tmp_path / "grid"
    _write_runlog(root / "rung3_delta_gamma_vega" / "s42",
                  _runlog("rung3_delta_gamma_vega", 42, best_val=0.0042))
    rec = digest.scan_grid(root)[0]
    assert rec["headline_loss"] == 0.0042
    assert rec["wall_clock_s"] == 123.4


def test_scan_ledger(tmp_path):
    ledger = tmp_path / "hedging" / "run_ledger.csv"
    _make_ledger(ledger)
    rows = digest.scan_ledger(ledger)
    assert len(rows) == 1 and rows[0]["cell"] == "combined_m1.0"
    assert rows[0]["status"] == "ok"
    assert digest.scan_ledger(tmp_path / "nope.csv") == []


def test_render_and_write_digest(tmp_path):
    root = tmp_path / "grid"
    ledger = tmp_path / "hedging" / "run_ledger.csv"
    _make_grid(root)
    _make_ledger(ledger)
    out = tmp_path / "digest.md"
    md = digest.write_digest(out, grid_root=root, ledger_path=ledger)

    assert out.read_text() == md
    assert "# Nightly digest" in md
    # anomaly rollup names the three broken runs
    assert "loss_nan" in md and "missing_checkpoint" in md and "early_stop_step750" in md
    # both tables present
    assert "## Grid training runs" in md
    assert "## Hedging cells" in md and "combined_m1.0" in md
    # clean run present, marked ok
    assert "rung3_delta_gamma_vega" in md


def test_unreadable_runlog_surfaced(tmp_path):
    root = tmp_path / "grid"
    d = root / "rung1_delta" / "s42"
    d.mkdir(parents=True)
    (d / "runlog.json").write_text("{not json")
    rec = digest.scan_grid(root)[0]
    assert rec["status"] == "unreadable"
    assert "unreadable_runlog" in rec["anomalies"]
