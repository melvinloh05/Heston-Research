"""R06 — exhibits.py plots a MISSING value as a hard 0.0 while its own backing
CSV correctly records the cell as blank.

E2's T_ex panel is the contract falsifier ("T_ex unmoved kills the cost
channel"), and T_ex = 0 is a meaningful reading: "this arm trades exactly like
the oracle". A cell with no t_ex_mean renders identically to that reading.

The same function handles missing values CORRECTLY two blocks later (the 2x2
inset maps None -> NaN and prints "n/a"), so this is an oversight rather than a
convention.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
import exhibits  # noqa: E402

_AGG_COLS = ["sweep", "direction", "magnitude", "lambda_j", "sigma_j", "in_model",
             "method", "tc", "cvar_level", "n_seeds", "cvar_mean", "cvar_seed_std",
             "mean_pnl_mean", "turnover_mean", "turnover_seed_std",
             "tc_component_mean", "directional_component_mean", "t_ex_mean",
             "t_ex_seed_std", "gap_closed_mean", "gap_closed_seed_std",
             "pnl_vs_baseline_cvar_diff_mean", "cvar_boot_se_mean"]

_M = {  # method: (cvar, gap_closed, t_ex, tc_comp, dir_comp, diff_vs_base)
    "oracle":                 (0.40, 1.00, 0.00, 0.05, 0.02, ""),
    "standard_pinn":          (1.00, 0.00, 0.30, 0.08, -0.01, ""),
    "standard_pinn_smoothed": (0.92, 0.13, 0.10, 0.04, -0.02, -0.08),
    "rung1":                  (0.85, 0.25, 0.28, 0.075, 0.00, -0.15),
    "rung3":                  (0.62, 0.63, 0.20, 0.065, 0.015, -0.38),
}


def write_agg(path, blank_tex_for=()):
    rows = []
    for magnitude in (0.0, 1.0):
        for tc in (0.0, 0.01, 0.02):
            for method, (cv, gc, te, tcc, dc, diff) in _M.items():
                scale = 1.0 if magnitude >= 0.5 else 0.4
                blank = method in blank_tex_for
                rows.append({
                    "sweep": "perturbation", "direction": "combined",
                    "magnitude": magnitude, "lambda_j": "", "sigma_j": "",
                    "in_model": magnitude == 0.0, "method": method, "tc": tc,
                    "cvar_level": 0.95, "n_seeds": 5,
                    "cvar_mean": round(cv * scale + tc * 2.0, 6),
                    "cvar_seed_std": 0.03, "mean_pnl_mean": -0.1,
                    "turnover_mean": 0.5 + te, "turnover_seed_std": 0.02,
                    "tc_component_mean": tcc, "directional_component_mean": dc,
                    "t_ex_mean": "" if blank else te,
                    "t_ex_seed_std": "" if blank else 0.01,
                    "gap_closed_mean": gc, "gap_closed_seed_std": 0.05,
                    "pnl_vs_baseline_cvar_diff_mean": diff,
                    "cvar_boot_se_mean": 0.02})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_AGG_COLS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # rung1's T_ex is absent (e.g. its cell never completed / the column was blank)
    agg = write_agg(td / "agg.csv", blank_tex_for=("rung1",))
    out = exhibits.exhibit_e2(agg, str(td / "out"),
                              arms=("rung3", "rung1", "standard_pinn", "oracle"))
    print("exhibit_e2 wrote:", {k: Path(v).name for k, v in out.items()
                                if isinstance(v, str)})
    print()
    print("--- E2 backing CSV, T_ex panel rows ---")
    with open(out["csv"]) as fh:
        for r in csv.DictReader(fh):
            if r["panel"] == "t_ex":
                print(f"   method={r['method']:<24} value={r['value']!r:<10} err={r['err']!r}")
    print()
    print("--- what the FIGURE draws for the same rows ---")
    rows = exhibits.read_csv(agg)
    bym = exhibits._by_method(rows, exhibits.MISSPEC_CELL, 0.01)
    for m in ("rung3", "rung1", "standard_pinn"):
        raw = bym.get(m, {}).get("t_ex_mean")
        plotted = exhibits._num(raw) or 0.0          # verbatim from exhibits.py:431
        print(f"   method={m:<24} csv cell={raw!r:<10} -> bar height {plotted}")
    print()
    print("   rung1's bar is drawn at exactly 0.0 — visually identical to")
    print("   'trades exactly like the oracle', which is the contract's")
    print("   cost-channel falsifier. The CSV correctly says blank.")
    print()
    print("--- the SAME function handles the 2x2 inset correctly (exhibits.py:444-455) ---")
    print("   M = [[quad[(r,b)] if quad[(r,b)] is not None else float('nan') ...]]")
    print("   ax_2x2.text(..., 'n/a' if not math.isfinite(v) else f'{v:.3g}')")
