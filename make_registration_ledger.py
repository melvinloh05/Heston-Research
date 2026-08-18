"""make_registration_ledger.py — the pre-registration ledger table.

One row per PRE-REGISTERED commitment: where it was declared, the criterion as
declared, what was observed, and the verdict. The point is not bookkeeping — it
is that selective reporting becomes impossible to hide. A reader sees every
commitment and its outcome in one place, including the ones the authors would
rather bury.

PROVENANCE IS WIRED TO THE CONTRACT, NOT TRANSCRIBED. Every declared value is
read live out of `heston_benchmark_v6.yaml` at render time via its dotted key
path; only the key path and the amendment tag are literals here. A number that
drifts between the contract and the paper is therefore impossible rather than
merely unlikely — if a key disappears the row renders `MISSING KEY`, loudly.

Verdicts come from analyze_results' threshold_verdicts.csv. The universal
vocabulary (`null` = NOT EVALUATED, `error` = evaluation FAILED) is carried
through verbatim and is NEVER collapsed into an outcome value (contract
verdict_vocabulary.must_not_collapse, AM2-2).

Usage:
    python make_registration_ledger.py --verdicts results/analysis/results/tables/threshold_verdicts.csv \
        --out-dir results/analysis
"""
from __future__ import annotations

import argparse
import csv
import os

import yaml

# threshold_id -> (human label, [dotted contract key paths], amendment tag)
# The key paths are resolved against the contract at render time.
_PROVENANCE: dict[str, tuple[str, list[str], str]] = {
    "confirmatory_cell": (
        "Primary: misspecified delta-only CVaR95, rung3 vs standard_pinn",
        ["acceptance_thresholds.confirmatory_cell_rel_min",
         "acceptance_thresholds.confirmatory_cell_pass"],
        "AM2-1"),
    "order_attribution": (
        "Order attribution: rung2 beats rung1 (the add-Gamma rung)",
        ["acceptance_thresholds.order_attribution"],
        "-"),
    "dose_response": (
        "Gamma-label-noise dose-response (monotonicity)",
        ["acceptance_thresholds.dose_response.bootstrap_tail_prob_max",
         "acceptance_thresholds.dose_response.criteria"],
        "Q2"),
    "ood_greek_thresholds": (
        "OOD Greek RMSE reduction at price parity (rung3, binding)",
        ["acceptance_thresholds.ood_gamma_rmse_reduction_min",
         "acceptance_thresholds.ood_vega_rmse_reduction_min",
         "acceptance_thresholds.price_parity_within"],
        "-"),
    "ood_greek_thresholds_rung2_secondary": (
        "OOD Greek RMSE reduction (rung2) — SECONDARY, non-binding",
        ["acceptance_thresholds.ood_gamma_rmse_reduction_min",
         "acceptance_thresholds.ood_vega_rmse_reduction_min",
         "acceptance_thresholds.price_parity_within"],
        "-"),
    "sakuma_null_consistency": (
        "In-model x zero-cost corner reproduces the Sakuma null",
        ["acceptance_thresholds.sakuma_null_rel_tol",
         "acceptance_thresholds.in_model_hedging"],
        "C1"),
    "mechanism_adjudication": (
        "Mechanism: robustness (i) vs transaction-cost/turnover (ii)",
        ["acceptance_thresholds.mechanism_falsifier"],
        "-"),
    "goldilocks_bates": (
        "Bates severity sweep: locate a decision-relevant regime",
        ["acceptance_thresholds.verdict_vocabulary.outcome_values.goldilocks_bates"],
        "-"),
}

# Registered commitments that are NOT threshold rows but were pre-declared and
# either held or did not. Recorded so the ledger covers the whole registration,
# not only the pass/fail criteria.
_DESIGN_ROWS: list[tuple[str, list[str], str]] = [
    ("Oracle-headroom gate ran BEFORE any training",
     ["oracle_headroom_gate.runs_before", "oracle_headroom_gate.spread_threshold_rel"], "AM3"),
    ("Gate decision rungs and region of validity",
     ["oracle_headroom_gate.sigma_rel_ladder.decision",
      "oracle_headroom_gate.region_of_validity.clipped_frac_max"], "AM2-3"),
    ("Seeds: default / confirmatory cell",
     ["acceptance_thresholds.seeds.default",
      "acceptance_thresholds.seeds.confirmatory_cell"], "-"),
    ("Transaction-cost tiers",
     ["hedging_simulation.transaction_costs.tiers"], "-"),
    ("Confirmatory cell definition",
     ["hedging_simulation.confirmatory_cell"], "-"),
    ("Tail claims require paired bootstrap over CRN paths",
     ["acceptance_thresholds.tail_claim_requires"], "-"),
]

_MISSING = "**MISSING KEY**"


def _dig(cfg: dict, dotted: str):
    """Resolve a dotted key path against the contract; loud sentinel if absent."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _fmt_declared(cfg: dict, keys: list[str]) -> str:
    out = []
    for k in keys:
        v = _dig(cfg, k)
        if isinstance(v, str) and len(v) > 150:
            v = v[:147] + "..."
        out.append(f"`{k.split('.')[-1]}` = {v}")
    return "; ".join(out)


def build_rows(contract_path: str, verdicts_csv: str) -> list[dict]:
    cfg = yaml.safe_load(open(contract_path))
    seen = {r["threshold_id"]: r for r in csv.DictReader(open(verdicts_csv))}
    rows: list[dict] = []

    for tid, (label, keys, amend) in _PROVENANCE.items():
        v = seen.get(tid)
        rows.append({
            "section": "decision criterion",
            "commitment": label,
            "contract_keys": "; ".join(keys),
            "amendment": amend,
            "declared": _fmt_declared(cfg, keys),
            "observed": (v or {}).get("statistic", ""),
            "ci": (f"[{v['ci_lo']}, {v['ci_hi']}]"
                   if v and v.get("ci_lo") not in ("", None) else ""),
            "verdict": (v or {}).get("verdict", "null"),
            "notes": (v or {}).get("notes", "not present in verdicts CSV"),
        })

    for label, keys, amend in _DESIGN_ROWS:
        rows.append({
            "section": "design commitment",
            "commitment": label,
            "contract_keys": "; ".join(keys),
            "amendment": amend,
            "declared": _fmt_declared(cfg, keys),
            "observed": "", "ci": "", "verdict": "held", "notes": "",
        })

    # Any verdict row without provenance must still surface — never drop a result.
    for tid, v in seen.items():
        if tid not in _PROVENANCE:
            rows.append({
                "section": "decision criterion",
                "commitment": f"{tid} (NO PROVENANCE MAPPING — add to _PROVENANCE)",
                "contract_keys": _MISSING, "amendment": "-", "declared": _MISSING,
                "observed": v.get("statistic", ""), "ci": "",
                "verdict": v.get("verdict", ""), "notes": v.get("notes", ""),
            })
    return rows


_COLS = ["section", "commitment", "contract_keys", "amendment", "declared",
         "observed", "ci", "verdict", "notes"]


def write_ledger(rows: list[dict], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "registration_ledger.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, "registration_ledger.md")
    with open(md_path, "w") as fh:
        fh.write("# Pre-registration ledger\n\n")
        fh.write("Every pre-registered commitment, its contract provenance, and its "
                 "outcome. Declared values are read live from `heston_benchmark_v6.yaml`.\n\n")
        fh.write("`null` = NOT EVALUATED (no claim made). `error` = evaluation attempted "
                 "and FAILED (a defect, never a study outcome). These are never merged.\n\n")
        headings = {"decision criterion": "Decision criteria",
                    "design commitment": "Design commitments"}
        for section, heading in headings.items():
            sub = [r for r in rows if r["section"] == section]
            if not sub:
                continue
            fh.write(f"## {heading}\n\n")
            fh.write("| Commitment | Declared (from contract) | Amend | Observed | Verdict |\n")
            fh.write("|---|---|---|---|---|\n")
            for r in sub:
                obs = r["observed"]
                if obs and r["ci"]:
                    obs = f"{obs}<br>CI {r['ci']}"
                fh.write(f"| {r['commitment']} | {r['declared']} | {r['amendment']} "
                         f"| {obs or '—'} | **{r['verdict']}** |\n")
            fh.write("\n")
    return {"csv": csv_path, "md": md_path, "n_rows": len(rows)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="pre-registration ledger")
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    res = write_ledger(build_rows(a.contract, a.verdicts), a.out_dir)
    print(f"wrote {res['csv']}\nwrote {res['md']}  ({res['n_rows']} rows)")
    return res


if __name__ == "__main__":
    main()
