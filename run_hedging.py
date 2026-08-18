"""run_hedging.py — confirmatory-cell-first driver for the delta-only hedging
program, plus the validation-only no-trade-band selector.

Orchestration only: the hedging mechanics, the CRN/path banks, the PnL
decomposition and the risk statistics all live in Hedging_backtest; the trained
(price, Greeks) arms come from pinn_provider.build_providers. This module adds
three things ON TOP of the engine, touching neither:

  run_confirmatory  the (combined perturbation) x {0.0, 1.0} x {all tc} cell at
                    10 seeds, hedged by oracle + standard_pinn + the supervision
                    ladder (rung1/rung2/rung3 — the order attribution lives here
                    too). persist_pnl FORCED True; outputs under
                    out_dir/confirmatory/.
  run_full_sweep    every arm (incl. info_matched_baseline, gradient_penalty_only,
                    the smoothed standard_pinn variant), the full perturbation +
                    Bates + Merton program at 5 seeds, persist_pnl True.
  select_band_width the no-trade band chosen on VALIDATION (magnitude-0.5 cells
                    ONLY — the control's hyperparameter is never tuned on the
                    confirmatory magnitude-1.0 cell).

PER-SEED PROVIDERS: PINN checkpoints live at ckpt_root/<arm>/s<seed>/best.pt, so
providers are rebuilt per seed — the engine's own seed loop assumes seed-agnostic
providers, so this driver enumerates cells with Hedging_backtest._iter_sim_cells
(seed outer) and calls the engine one cell at a time with derived seeds trimmed
to [seed] ("seeds plumbing"; the engine is untouched). One cell = one unit of
the run_ledger.csv, so a completed (cell, seed) is skipped on resume.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time

import numpy as np
import yaml

import Hedging_backtest as hb
from pinn_provider import build_providers

# supervision ladder + the confirmatory contrast arm (order attribution at the cell)
_CONFIRMATORY_ARMS = ["standard_pinn", "rung1", "rung2", "rung3"]
# full program: the ladder + sans-PDE factorial arm + the two matched-baseline
# controls + the six gamma-label-noise dose-response arms (identity _ARM_DIR
# fallback: checkpoint dir == arm name); the smoothed standard_pinn variant is
# added by the engine itself.
ALL_ARMS = ["standard_pinn", "rung1", "rung2", "rung3", "sans_pde",
            # baseline 0 = the factorial's supervision-OFF x PDE-OFF cell; hedged
            # so the 2x2 has all four corners on the primary metric (D1).
            "feedforward",
            "gradient_penalty_only", "info_matched_baseline",
            "sigma_000", "sigma_010", "sigma_025", "sigma_050",
            "shuffled", "bs_gamma"]

_OUT_NAME = "headline_delta_only"            # matches Hedging_backtest.run_headline
_LEDGER_COLS = ["run", "cell", "seed", "wall_clock_s", "n_rows", "status"]

# ---------------------------------------------------------------------------
# nightly ledger (append-only, resumable) + resume-safe row master
# ---------------------------------------------------------------------------

def _ledger_done(path: str) -> set:
    """Completed (cell, seed) pairs from a run_ledger.csv (status == 'ok')."""
    done: set = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add((row["cell"], int(row["seed"])))
    return done


def _append_ledger(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_LEDGER_COLS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def _append_rows_csv(path: str, rows: list[dict]) -> None:
    """Append per-cell metric rows (incl. the hidden gap_* carriers) to a
    resume-safe master, keeping the on-disk header order stable across resumes."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fieldnames = header
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _coerce(val: str):
    """CSV string -> int/float/str for re-aggregation ('' stays blank)."""
    if val == "":
        return ""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _read_rows_typed(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]

# ---------------------------------------------------------------------------
# per-cell cfg trimming (one cell of _iter_sim_cells at one seed)
# ---------------------------------------------------------------------------

def _cell_cfg(cfg: dict, tag: dict, seed: int) -> dict:
    """A resolved-cfg copy trimmed so Hedging_backtest._iter_sim_cells yields
    EXACTLY the cell described by `tag`, at derived seeds [seed]. The engine is
    then invoked one cell at a time; the slug (and hence the pnl npz name and the
    ledger key) is identical to the untrimmed program's, so resume aligns."""
    c = copy.deepcopy(cfg)
    c["derived"]["seeds"] = [int(seed)]
    mis = c["benchmark"]["hedging_simulation"]["misspecification"]
    src_perts = cfg["benchmark"]["hedging_simulation"]["misspecification"]["perturbations"]
    if tag["sweep"] == "perturbation":
        mis["perturbations"] = {tag["direction"]: src_perts[tag["direction"]]}
        mis["cross_model"] = []
        c["engine"]["misspecification"]["magnitudes"] = [tag["magnitude"]]
    elif tag["sweep"] == "bates":
        mis["perturbations"] = {}
        mis["cross_model"] = ["bates_severity_sweep"]
        sw = c["benchmark"]["models"]["bates"]["jump_severity_sweep"]
        sw["lambda_j"] = [tag["lambda_j"]]
        sw["sigma_j"] = [tag["sigma_j"]]
    elif tag["sweep"] == "merton":
        mis["perturbations"] = {}
        mis["cross_model"] = ["merton"]
    else:
        raise ValueError(f"unknown sweep {tag['sweep']!r}")
    return c

# ---------------------------------------------------------------------------
# no-trade band plumbing (TASK 3 output consumed here)
# ---------------------------------------------------------------------------

def _apply_selected_band(cfg: dict, band_yaml: str | None) -> dict:
    """When band_yaml exists, override engine smoothing_baseline.no_trade_band
    with the validation-selected width (select_band_width output). Absent -> the
    hedging_config default (0.02) is unchanged. The band is a HEDGE-TIME control
    hyperparameter tuned on magnitude-0.5 validation cells, never here."""
    if not band_yaml or not os.path.exists(band_yaml):
        return cfg
    sel = yaml.safe_load(open(band_yaml))
    cfg["engine"]["smoothing_baseline"]["no_trade_band"] = float(sel["no_trade_band"])
    return cfg

# ---------------------------------------------------------------------------
# shared program driver
# ---------------------------------------------------------------------------

def _run_program(prog: dict, ckpt_root, run_root: str, arms: list[str],
                 run_label: str) -> dict:
    """Enumerate `prog`'s cells (seed outer), skip completed (cell, seed) pairs
    via the ledger, and run each remaining cell one at a time with that seed's
    providers. Appends a ledger row and the cell's metric rows per completed
    cell (both resume-safe); rewrites the complete per_seed/agg CSVs from the
    row master at the end. `prog` must already carry its seeds, cell trims and
    persist_pnl. Returns a run summary (ran / skipped cells, seeds, paths)."""
    os.makedirs(run_root, exist_ok=True)
    ledger_path = os.path.join(run_root, "run_ledger.csv")
    master_path = os.path.join(run_root, "_rows_master.csv")
    contract = prog["benchmark"]
    r, q = contract["grid"]["r"], contract["grid"]["q"]
    done = _ledger_done(ledger_path)
    ran, skipped = [], []
    providers_cache: dict[int, dict] = {}
    for seed, tag, _dgp_kind, _dgp in hb._iter_sim_cells(prog):
        slug = hb._cell_slug(tag, seed)
        if (slug, seed) in done:
            skipped.append((slug, seed))
            continue
        if seed not in providers_cache:
            providers_cache.clear()                 # hold one seed's arms at a time
            providers_cache[seed] = build_providers(contract, ckpt_root, arms,
                                                    seed, r, q)
        cell_cfg = _cell_cfg(prog, tag, seed)
        t0 = time.perf_counter()
        rows = hb.run_headline(cell_cfg, providers_cache[seed], run_root)
        wall = time.perf_counter() - t0
        _append_rows_csv(master_path, rows)
        _append_ledger(ledger_path, {"run": run_label, "cell": slug,
                                     "seed": int(seed), "wall_clock_s": round(wall, 4),
                                     "n_rows": len(rows), "status": "ok"})
        done.add((slug, seed))
        ran.append((slug, seed))

    # PROVENANCE (audit R1): the engine writes resolved_config.yaml into this
    # shared run_root on every per-cell call, each with a TRIMMED config (one
    # direction, one magnitude, one seed), so what survived the loop described
    # the LAST CELL rather than the run. Re-write it from the untrimmed program
    # so the surviving copy is the one a reproduction attempt needs.
    hb.log_resolved_config(prog, run_root)

    master = _read_rows_typed(master_path)
    if master:
        gap_floor = float(prog["engine"]["risk"].get("gap_floor_frac", 0.01))
        clean = [{k: v for k, v in row.items() if not str(k).startswith("_")}
                 for row in master]
        hb.write_rows_csv(clean, os.path.join(run_root, f"{_OUT_NAME}_per_seed.csv"))
        hb.write_rows_csv(hb.aggregate_over_seeds(master, gap_floor),
                          os.path.join(run_root, f"{_OUT_NAME}_agg.csv"))
    return {"run": run_label, "run_root": run_root,
            "ran": ran, "skipped": skipped,
            "n_ran": len(ran), "n_skipped": len(skipped),
            "seeds": list(prog["derived"]["seeds"]),
            "ledger": ledger_path, "pnl_dir": os.path.join(run_root, f"pnl_{_OUT_NAME}")}


def _print_disk_estimate(prog: dict, arms: list[str]) -> int:
    cells = sum(1 for _ in hb._iter_sim_cells(prog))
    tiers = len(prog["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"])
    n_paths = int(prog["engine"]["simulation"]["n_paths"])
    applies = prog["engine"].get("smoothing_baseline", {}).get("applies_to", [])
    n_methods = 1 + len(arms) + sum(1 for a in arms if a in applies)  # +oracle +smoothed
    est = cells * n_methods * tiers * n_paths * 8
    print(f"[run_full_sweep] persist_pnl disk estimate: {cells} cells x {n_methods} "
          f"methods x {tiers} tc tiers x {n_paths} paths x float64 "
          f"~= {est / 1e6:.1f} MB uncompressed (npz keeps it compressed on disk)")
    return est

# ---------------------------------------------------------------------------
# TASK 2 — confirmatory-cell-first + full-sweep runners
# ---------------------------------------------------------------------------

def run_confirmatory(cfg: dict, ckpt_root, out_dir: str,
                     band_yaml: str | None = None) -> dict:
    """Confirmatory cell first: combined perturbation at magnitudes {0.0, 1.0}
    (0.0 = the in-model Sakuma-null consistency cell), all three tc tiers,
    seeds = global_seed + 0..(seeds_confirmatory_cell - 1) [= 10], hedged by
    oracle + standard_pinn + rung1/rung2/rung3. persist_pnl FORCED True; outputs
    under out_dir/confirmatory/."""
    meta = cfg["benchmark"]["meta"]
    g, n_seeds = int(meta["global_seed"]), int(meta["seeds_confirmatory_cell"])
    prog = copy.deepcopy(cfg)
    prog["derived"]["seeds"] = [g + i for i in range(n_seeds)]   # engine untouched
    prog["engine"]["risk"]["persist_pnl"] = True
    mis = prog["benchmark"]["hedging_simulation"]["misspecification"]
    mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
    mis["cross_model"] = []
    prog["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]
    _apply_selected_band(prog, band_yaml)
    return _run_program(prog, ckpt_root, os.path.join(out_dir, "confirmatory"),
                        _CONFIRMATORY_ARMS, "confirmatory")


def run_full_sweep(cfg: dict, ckpt_root, out_dir: str, arms: list[str] | None = None,
                   band_yaml: str | None = None) -> dict:
    """Full program: every arm (default ALL_ARMS incl. info_matched_baseline and
    gradient_penalty_only; the engine adds the smoothed standard_pinn variant),
    the full perturbation + Bates severity + Merton sweep at the default 5 seeds,
    persist_pnl True (a disk estimate is printed). Outputs under
    out_dir/full_sweep/."""
    arms = list(ALL_ARMS if arms is None else arms)
    prog = copy.deepcopy(cfg)
    prog["engine"]["risk"]["persist_pnl"] = True
    _apply_selected_band(prog, band_yaml)
    _print_disk_estimate(prog, arms)
    return _run_program(prog, ckpt_root, os.path.join(out_dir, "full_sweep"),
                        arms, "full_sweep")

# ---------------------------------------------------------------------------
# TASK 3 — no-trade band width on validation (roadmap M4 step 3)
# ---------------------------------------------------------------------------

def select_band_width(cfg: dict, ckpt_root,
                      widths=(0.005, 0.01, 0.02, 0.04), tc: float = 0.01,
                      out_dir: str | None = None,
                      arms: tuple[str, ...] = ("standard_pinn",)) -> dict:
    """Choose the no-trade band width on VALIDATION cells only.

    Evaluated ONLY on magnitude-0.5 cells (the mid severity point) — EXPLICITLY
    not the confirmatory magnitude-1.0 cells, so the smoothing control's own
    hyperparameter is never tuned on the confirmatory data it will be judged
    against. Criterion: minimum CVaR95 at `tc` (= 0.01) of standard_pinn_smoothed,
    averaged per seed then over seeds.

    PER-SEED PROVIDERS: PINN checkpoints live at ckpt_root/<arm>/s<seed>/best.pt,
    so providers are REBUILT for every engine seed and the engine is invoked one
    seed at a time (derived seeds trimmed to [seed]) — exactly as _run_program
    does. Reusing a single providers dict across seeds would hedge every seed
    with seed 0's network, faking the seed diversity of the selection. `arms`
    names the band's target ('standard_pinn'; the band is applied to its
    positions); 'oracle' (premium/gap ceiling) and the float64 double-backward
    Greeks come free from build_providers, identical to the runners' use of it.

    Writes band_selected.yaml {no_trade_band, table, ...} under out_dir when
    given; the runners consume it via band_yaml=. Returns the selection dict.
    """
    base = copy.deepcopy(cfg)
    base["benchmark"]["hedging_simulation"]["misspecification"]["cross_model"] = []
    base["engine"]["misspecification"]["magnitudes"] = [0.5]      # mid severity ONLY
    base["engine"]["risk"]["persist_pnl"] = False
    level = base["engine"]["risk"]["cvar_level"]
    contract = base["benchmark"]
    r, q = contract["grid"]["r"], contract["grid"]["q"]
    arms = list(arms)

    # cvar per (width, seed); providers rebuilt per seed and held one seed's arms
    # at a time (mirrors _run_program's construction + teardown).
    per_seed: dict[float, dict] = {float(w): {} for w in widths}
    consumed_mags: set = set()
    for seed in base["derived"]["seeds"]:
        providers = build_providers(contract, ckpt_root, arms, seed, r, q)
        c_seed = copy.deepcopy(base)
        c_seed["derived"]["seeds"] = [int(seed)]                  # engine untouched
        for w in widths:
            c = copy.deepcopy(c_seed)
            c["engine"]["smoothing_baseline"]["no_trade_band"] = float(w)
            rows = hb.run_headline(c, providers, out_dir=None)
            consumed = [row for row in rows
                        if row["method"] == "standard_pinn_smoothed" and row["tc"] == tc]
            assert consumed, (
                f"no standard_pinn_smoothed rows at tc={tc}: arms must include "
                "'standard_pinn' and engine smoothing_baseline must apply to it")
            # falsifier guard: the band is tuned ONLY on the mid-severity cell
            assert all(row["magnitude"] == 0.5 for row in consumed), \
                "select_band_width consumed a non-0.5-magnitude row"
            consumed_mags.update(row["magnitude"] for row in consumed)
            per_seed[float(w)].setdefault(int(seed), []).extend(
                float(row["cvar"]) for row in consumed)
        del providers                                            # hold one seed at a time
    table = {w: float(np.mean([np.mean(v) for v in seeds.values()]))
             for w, seeds in per_seed.items()}
    best = min(table, key=table.get)
    sel = {"no_trade_band": float(best),
           "criterion": f"min_cvar{level}_at_tc{tc}_on_magnitude0.5_validation_cells",
           "magnitudes_consumed": sorted(consumed_mags),
           "table": {float(k): float(v) for k, v in table.items()}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "band_selected.yaml")
        with open(path, "w") as fh:
            yaml.safe_dump(sel, fh, sort_keys=False)
        sel["path"] = path
    return sel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="delta-only hedging runner")
    ap.add_argument("mode", choices=["confirmatory", "full_sweep"])
    ap.add_argument("--ckpt-root", required=True, help="root of <arm>/s<seed>/best.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--engine", default="hedging_config.yaml")
    ap.add_argument("--band-yaml", default=None,
                    help="band_selected.yaml from select_band_width (optional)")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = _build_cli().parse_args(argv)
    cfg = hb.resolve_config(args.contract, args.engine)
    if args.mode == "confirmatory":
        return run_confirmatory(cfg, args.ckpt_root, args.out_dir, args.band_yaml)
    return run_full_sweep(cfg, args.ckpt_root, args.out_dir, band_yaml=args.band_yaml)


if __name__ == "__main__":
    print(main())
