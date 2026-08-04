"""Fix batch 3, ITEM 2 — re-measure the gate's `clipped_frac` at PRODUCTION scale.

The region-of-validity bound `oracle_headroom_gate.region_of_validity.clipped_frac_max`
(0.25, AM2-3c) was fitted to a SMOKE-SIZE measurement: `audit/fixlog/g2_measured.txt`,
field mode, n_paths=256, 2 seeds, and only the OLD ladder (0.1, 0.2, 0.4, 0.8). Two of the
four rungs the amendment declares (0.05, 0.15) were never measured at all — they were
BOUNDED by their measured neighbours via the monotonicity argument in `spacing_rationale`.

This script measures the DECLARED ladder (decision 0.05/0.10/0.15/0.20 + diagnostic 0.40)
at the production path count (`hedging_config.yaml simulation.n_paths` = 10000) and the
contract's confirmatory seed count (`meta.seeds_confirmatory_cell` = 10), ONE SEED PER RUN
so the seed-to-seed spread of `clipped_frac` — explicitly called unknown in the amendment's
§3.3 caveat — is measurable rather than pooled away.

It is run BEFORE any pilot fit exists, so no rung's measured value can be chosen with
knowledge of where `sigma_gamma_pilot` lands.

    python audit/repro/e2_g2_production_scale.py            # writes audit/fixlog/g2_production_scale.txt

Field mode only (the PRIMARY corruption model; the iid arm is contrast-only and does not
enter the region-of-validity bound). Nothing here writes to data/ or results/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import Hedging_backtest as hb                                    # noqa: E402
import gate_headroom as gh                                       # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixlog" / "g2_production_scale.txt"
SCRATCH = os.environ.get("G2_SCRATCH", "/tmp/g2_production_scale")

# the 256-path / 2-seed numbers this re-measurement is compared against
SMOKE = {0.1: 0.061364, 0.2: 0.215253, 0.4: 0.779978, 0.8: 0.959984}


def main() -> None:
    cfg0 = hb.resolve_config()
    th = hb.contract_thresholds(cfg0)
    ladder = [(sr, True) for sr in th["gate_sigma_rel_decision"]] + \
             [(sr, False) for sr in th["gate_sigma_rel_diagnostic"]]
    n_paths = int(cfg0["engine"]["simulation"]["n_paths"])
    n_seeds = int(th["seeds_confirmatory_cell"])
    g0 = int(th["global_seed"])
    seeds = [g0 + i for i in range(n_seeds)]
    frac_max = float(cfg0["benchmark"]["oracle_headroom_gate"]
                     ["region_of_validity"]["clipped_frac_max"])

    per_seed: dict[float, list[float]] = {sr: [] for sr, _ in ladder}
    spread: dict[tuple, list[float]] = {}
    t0 = time.time()
    for i, seed in enumerate(seeds):
        cfg = hb.resolve_config()
        cfg["engine"]["simulation"]["n_paths"] = n_paths
        cfg["derived"]["seeds"] = [seed]
        res = gh.run_gate(cfg, mode="field", out_dir=os.path.join(SCRATCH, f"s{seed}"))
        for s in res["summary"]:
            per_seed_key = float(s["sigma_rel"])
            spread.setdefault((per_seed_key, float(s["tc"])), []).append(
                float(s["spread_rel_mean"]))
        for label, frac in res["clipped_frac"].items():
            sr = float(label[1:])
            per_seed[sr].append(float(frac))
        print(f"[{i + 1}/{len(seeds)}] seed {seed} done "
              f"({time.time() - t0:.0f}s elapsed): "
              + ", ".join(f"{sr:g}:{per_seed[sr][-1]:.4f}" for sr, _ in ladder),
              flush=True)

    lines = [
        "g2_production_scale.txt — clipped_frac at PRODUCTION scale, fix batch 3 ITEM 2",
        "",
        f"gate mode=field, confirmatory cell, n_paths={n_paths} per seed, "
        f"{len(seeds)} seeds {seeds} run ONE PER GATE so the seed spread is visible.",
        f"Contract ladder (oracle_headroom_gate.sigma_rel_ladder): decision "
        + ", ".join(f"{sr:g}" for sr, e in ladder if e)
        + " | diagnostic " + ", ".join(f"{sr:g}" for sr, e in ladder if not e),
        f"region_of_validity.clipped_frac_max = {frac_max}",
        f"Comparison column: audit/fixlog/g2_measured.txt (n_paths=256, 2 seeds, "
        f"OLD ladder 0.1/0.2/0.4/0.8).",
        f"Measured BEFORE any pilot fit or gate go/no-go run existed.",
        "",
        "| sigma_rel | role | clipped_frac mean | seed std | min | max | "
        "256-path/2-seed | <= 0.25 ? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for sr, elig in ladder:
        v = np.asarray(per_seed[sr], float)
        smoke = SMOKE.get(sr)
        inside = bool(v.mean() <= frac_max)
        verdicts.append((sr, elig, float(v.mean()), inside))
        lines.append(
            f"| {sr:g} | {'decision' if elig else 'DIAGNOSTIC'} | {v.mean():.4f} | "
            f"{v.std(ddof=1):.4f} | {v.min():.4f} | {v.max():.4f} | "
            + (f"{smoke:.4f}" if smoke is not None else "not measured")
            + f" | {'yes' if inside else 'NO'} |")

    lines += ["", "Per-seed clipped_frac (field mode):", ""]
    lines.append("| seed | " + " | ".join(f"{sr:g}" for sr, _ in ladder) + " |")
    lines.append("|---" * (len(ladder) + 1) + "|")
    for j, seed in enumerate(seeds):
        lines.append(f"| {seed} | "
                     + " | ".join(f"{per_seed[sr][j]:.4f}" for sr, _ in ladder) + " |")

    lines += ["", "spread_rel_mean (mean over the per-seed gate runs):", "",
              "| sigma_rel | tc | spread_rel_mean |", "|---|---|---|"]
    for (sr, tc), vals in sorted(spread.items()):
        lines.append(f"| {sr:g} | {tc} | {float(np.mean(vals)):+.4f} |")

    crossings = [(sr, elig, m) for sr, elig, m, inside in verdicts if not inside and elig]
    diag_inside = [(sr, m) for sr, elig, m, inside in verdicts if inside and not elig]
    # per-SEED crossings too: a rung whose seed MEAN is inside the bound while
    # individual seeds sit above it is exactly the borderline the smoke-size
    # measurement could not see, and it is a decision, not a rounding detail.
    straddling = []
    for sr, elig in ladder:
        if not elig:
            continue
        v = np.asarray(per_seed[sr], float)
        n_out = int((v > frac_max).sum())
        if n_out and v.mean() <= frac_max:
            straddling.append((sr, n_out, len(v), float(v.max())))
    lines += ["", "REGION-OF-VALIDITY CHECK (clipped_frac_max = "
              f"{frac_max}; the bound is NOT adjusted by this script):", ""]
    if crossings:
        lines.append("  *** A DECISION RUNG LANDS OUTSIDE THE REGION AT PRODUCTION SCALE:")
        for sr, _e, m in crossings:
            lines.append(f"      sigma_rel={sr:g}: mean clipped_frac={m:.4f} > {frac_max}")
    else:
        lines.append("  every DECISION rung's seed MEAN stays inside the region.")
    if straddling:
        lines.append("")
        lines.append("  *** BUT A DECISION RUNG STRADDLES THE BOUND PER SEED:")
        for sr, n_out, n_all, mx in straddling:
            lines.append(f"      sigma_rel={sr:g}: {n_out} of {n_all} seeds exceed "
                         f"{frac_max} (max {mx:.4f})")
    if crossings or straddling:
        lines.append("      This is a CONTRACT question (revise clipped_frac_max or the")
        lines.append("      ladder), not a code fix, and it is the human's decision.")
    if diag_inside:
        lines.append("  NOTE: a DIAGNOSTIC rung is inside the region at production scale: "
                     + ", ".join(f"{sr:g} ({m:.4f})" for sr, m in diag_inside)
                     + " — it stays diagnostic regardless (the ladder is pre-registered).")
    lines.append("")
    lines.append(f"wall clock: {time.time() - t0:.0f}s")

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
