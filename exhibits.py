"""exhibits.py — deterministic figure/table regeneration from frozen results/.

Every exhibit is a PURE function of CSV artifacts under a results/ tree: P11's
OOD-param Greek aggregates (``ood_param_greeks_agg.csv``), P12's hedging per-seed
and seed-aggregate CSVs (``headline_delta_only_{per_seed,agg}.csv``) and P13's
verdict / dose tables (``dose_response.csv``, ``threshold_verdicts.csv``). NO
simulation, NO training, NO model or engine evaluation happens here: an exhibit
reads CSVs and writes a PNG plus its backing CSV. Regeneration is bit-stable for
fixed inputs — tests hash the emitted CSV, never the PNG.

House style (matches the rest of the repo): matplotlib Agg backend, ONE shared
style block (no seaborn), and method colours drawn from
``Hedging_backtest._OVERLAY_COLORS`` so a given method keeps ONE colour across
E1-E4 (method identity). Each exhibit is a single function returning the written
paths; ``overlay_delta_paths`` is the one thin wrapper over the engine's
``run_delta_overlay`` diagnostic (that one DOES simulate — it is a convenience
delegate, not one of the pure-from-results exhibits).
"""
from __future__ import annotations

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")                                   # headless, deterministic
import matplotlib.pyplot as plt                         # noqa: E402

import Hedging_backtest as hb                            # noqa: E402  (color source)

# ---------------------------------------------------------------------------
# artifact naming (tracks run_hedging / eval_greeks / analyze_results)
# ---------------------------------------------------------------------------
HEDGING_AGG = "headline_delta_only_agg.csv"             # P12 seed-aggregate
HEDGING_PER_SEED = "headline_delta_only_per_seed.csv"   # P12 per-seed
GREEK_AGG = "ood_param_greeks_agg.csv"                  # P11 OOD Greek aggregate
DOSE_CSV = "dose_response.csv"                           # P13 dose-response

# confirmatory cell = combined perturbation at magnitude 1.0 (misspec) / 0.0
# (in-model), tc = 0.01 (CLAUDE.md). Cells are selected by matching agg key cols.
MISSPEC_CELL = {"sweep": "perturbation", "direction": "combined", "magnitude": 1.0}
INMODEL_CELL = {"sweep": "perturbation", "direction": "combined", "magnitude": 0.0}
CONFIRMATORY_TC = 0.01

# supervision ladder (engine method name -> pretty rung label); standard_pinn is
# the price-only structural baseline at the confirmatory cell.
LADDER_DEFAULT = (("standard_pinn", "price"), ("rung1", "+Δ"),
                  ("rung2", "+Δ+Γ"), ("rung3", "+Δ+Γ+ν"))

# residual x supervision 2x2 (supervision, residual, engine method name). The
# price-only x PDE-ablated corner is not a trained hedging arm, so it renders
# n/a by design; pass a custom tuple to slot one in later.
FACTORIAL_DEFAULT = (
    ("price-only", "PDE retained", "standard_pinn"),
    ("Sobolev Δ+Γ+ν", "PDE retained", "rung3"),
    ("price-only", "PDE ablated (ω=0)", ""),
    ("Sobolev Δ+Γ+ν", "PDE ablated (ω=0)", "sans_pde"),
)

# focal arms for the mechanism / decomposition figures (present -> drawn)
MECH_ARMS = ("rung3", "rung1", "standard_pinn", "standard_pinn_smoothed", "oracle")
FOCAL_ARM = "rung3"
BASELINE_ARM = "standard_pinn"
SMOOTHED_ARM = "standard_pinn_smoothed"
ORACLE_ARM = "oracle"

# ---------------------------------------------------------------------------
# method identity: ONE colour + one label per method, shared across E1-E4
# ---------------------------------------------------------------------------
_METHOD_COLOR_ORDER = ("oracle", "standard_pinn", "standard_pinn_smoothed",
                       "rung1", "rung2", "rung3", "sans_pde",
                       "gradient_penalty_only")
_METHOD_COLORS = {m: hb._OVERLAY_COLORS[i % len(hb._OVERLAY_COLORS)]
                  for i, m in enumerate(_METHOD_COLOR_ORDER)}
_NEUTRAL = "#7f7f7f"

_METHOD_LABEL = {
    "oracle": "oracle", "standard_pinn": "standard PINN",
    "standard_pinn_smoothed": "smoothed band", "rung1": "+Δ (rung1)",
    "rung2": "+Δ+Γ (rung2)", "rung3": "+Δ+Γ+ν (rung3)",
    "sans_pde": "Sobolev sans-PDE", "gradient_penalty_only": "grad-penalty",
    "info_matched_baseline": "info-matched",
}

# Okabe-Ito colourblind-safe palette (E3 colourblind variant only)
_CB_SAFE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
            "#D55E00", "#F0E442", "#000000")

_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 120,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8,
}


def _method_color(name: str) -> str:
    return _METHOD_COLORS.get(name, _NEUTRAL)


def _method_label(name: str) -> str:
    return _METHOD_LABEL.get(name, str(name))


# ---------------------------------------------------------------------------
# small IO helpers (typed CSV read; deterministic CSV write; input guards)
# ---------------------------------------------------------------------------

def _coerce(v: str):
    """CSV cell -> int/float/str ('' stays blank); matches run_hedging._coerce."""
    if v == "":
        return ""
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(path) -> list[dict]:
    """Typed rows from a metric CSV (blank cells stay ''); [] when missing."""
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _require(path) -> str:
    """Raise with the EXACT expected path when a required input is absent."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"required exhibit input not found: {path}")
    return str(path)


def _num(v):
    """float(v) or None for blanks / non-numeric (never raises)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _nan_if_missing(v) -> float:
    """`_num(v)`, with a MISSING cell as NaN rather than 0.0.

    Plotting a blank as 0.0 substitutes a real, meaningful value: T_ex = 0 is
    the contract's cost-channel confirming evidence and gap_closed = 0 is an
    affirmative negative result, so an absent cell would render as a finding.
    matplotlib omits NaN bars/points instead of drawing them at zero.
    """
    f = _num(v)
    return float("nan") if f is None else f


def _fmt(v) -> str:
    """Deterministic cell text for the backing CSV (bit-stable across runs)."""
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return "" if not math.isfinite(v) else format(v, ".10g")
    return str(v)


def _write_csv(path, cols, rows) -> str:
    """Write `rows` with fixed `cols`; '\\n' terminator, fixed float formatting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_fmt(r.get(c, "")) for c in cols])
    return str(path)


def _savefig(fig, path) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# row selection (float-tolerant match on the agg key columns)
# ---------------------------------------------------------------------------

def _matches(row: dict, cell: dict, tc=None) -> bool:
    for k, v in cell.items():
        rv = row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            rn = _num(rv)
            if rn is None or abs(rn - float(v)) > 1e-9:
                return False
        elif str(rv) != str(v):
            return False
    if tc is not None:
        rn = _num(row.get("tc"))
        if rn is None or abs(rn - float(tc)) > 1e-9:
            return False
    return True


def _by_method(rows, cell, tc) -> dict:
    """method -> agg row for one (cell, tc) slice."""
    return {r["method"]: r for r in rows if _matches(r, cell, tc)
            and r.get("method") not in (None, "")}


def _tc_tiers(rows, cell) -> list[float]:
    tiers = {_num(r.get("tc")) for r in rows if _matches(r, cell)}
    return sorted(t for t in tiers if t is not None)


def _greek_row(greek_rows, arm, greek, regime, slice_="full") -> dict | None:
    for r in greek_rows:
        if (str(r.get("arm")) == arm and str(r.get("greek")) == greek
                and str(r.get("regime")) == regime and str(r.get("slice")) == slice_):
            return r
    return None


# ===========================================================================
# E1 — supervision ladder + residual x supervision factorial + OOD Greek panel
# ===========================================================================

def exhibit_e1(hedging_agg_csv, greek_agg_csv, out_dir, *,
               cell=MISSPEC_CELL, tc=CONFIRMATORY_TC, ladder=LADDER_DEFAULT,
               factorial=FACTORIAL_DEFAULT, greek_arm=FOCAL_ARM,
               greeks=("gamma", "vega"),
               regimes=("near_feller", "strong_neg_corr"),
               stem="e1_ladder_factorial") -> dict:
    """Ladder (misspec CVaR95 along price->+Δ->+Δ+Γ->+Δ+Γ+ν) +
    a rendered 4-cell residual×supervision factorial table (CVaR mean±seed-std,
    gap_closed_mean) + an OOD-param Greek side panel (Γ/ν reductions vs
    standard_pinn with P11 seed error bars; Vega's payoff lives HERE)."""
    agg = read_csv(_require(hedging_agg_csv))
    greek = read_csv(_require(greek_agg_csv))
    bym = _by_method(agg, cell, tc)

    csv_rows: list[dict] = []
    # ---- ladder rows ----
    ladder_pts = []
    for method, label in ladder:
        row = bym.get(method, {})
        cv, sd = _num(row.get("cvar_mean")), _num(row.get("cvar_seed_std"))
        gc = _num(row.get("gap_closed_mean"))
        ladder_pts.append((method, label, cv, sd, gc))
        csv_rows.append({"section": "ladder", "supervision": label,
                         "residual": "", "method": method, "cvar_mean": cv,
                         "cvar_seed_std": sd, "gap_closed_mean": gc})
    # ---- factorial rows ----
    for sup, res, method in factorial:
        row = bym.get(method, {})
        csv_rows.append({"section": "factorial", "supervision": sup,
                         "residual": res, "method": method,
                         "cvar_mean": _num(row.get("cvar_mean")),
                         "cvar_seed_std": _num(row.get("cvar_seed_std")),
                         "gap_closed_mean": _num(row.get("gap_closed_mean"))})
    # ---- greek panel rows ----
    for regime in regimes:
        for g in greeks:
            gr = _greek_row(greek, greek_arm, g, regime) or {}
            csv_rows.append({"section": "greek", "supervision": greek_arm,
                             "residual": regime, "method": greek_arm, "greek": g,
                             "reduction_mean": _num(gr.get("reduction_vs_standard_pinn_mean")),
                             "reduction_std": _num(gr.get("reduction_vs_standard_pinn_std"))})

    cols = ["section", "supervision", "residual", "method", "greek", "cvar_mean",
            "cvar_seed_std", "gap_closed_mean", "reduction_mean", "reduction_std"]
    csv_path = _write_csv(os.path.join(out_dir, f"{stem}.csv"), cols, csv_rows)

    # ---- figure ----
    with matplotlib.rc_context(_STYLE):
        fig = plt.figure(figsize=(12.5, 4.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.15], wspace=0.32)
        ax_l, ax_t, ax_g = (fig.add_subplot(gs[0, i]) for i in range(3))

        # ladder
        xs = list(range(len(ladder_pts)))
        for x, (method, label, cv, sd, gc) in zip(xs, ladder_pts):
            if cv is None:
                continue
            ax_l.errorbar([x], [cv], yerr=[sd or 0.0], marker="o", ms=6,
                          color=_method_color(method), capsize=3, lw=0)
        line_x = [x for x, p in zip(xs, ladder_pts) if p[2] is not None]
        line_y = [p[2] for p in ladder_pts if p[2] is not None]
        ax_l.plot(line_x, line_y, color=_NEUTRAL, lw=1.2, zorder=0)
        ax_l.set_xticks(xs)
        ax_l.set_xticklabels([p[1] for p in ladder_pts], fontsize=8)
        ax_l.set_ylabel("misspec CVaR95 (loss)")
        ax_l.set_title("supervision ladder", fontsize=10)
        ax_l.margins(x=0.12)

        # factorial table (rendered)
        ax_t.axis("off")
        ax_t.set_title("residual × supervision (confirmatory cell)", fontsize=10)
        sups = list(dict.fromkeys(t[0] for t in factorial))
        ress = list(dict.fromkeys(t[1] for t in factorial))
        lut = {(s, r): m for s, r, m in factorial}
        header = [""] + ress
        table_rows = [header]
        for s in sups:
            trow = [s]
            for r in ress:
                m = lut.get((s, r), "")
                row = bym.get(m, {})
                cv, sd = _num(row.get("cvar_mean")), _num(row.get("cvar_seed_std"))
                gc = _num(row.get("gap_closed_mean"))
                if cv is None:
                    trow.append("n/a")
                else:
                    txt = f"{cv:.3g} ± {sd or 0.0:.2g}"
                    if gc is not None:
                        txt += f"\ngap {gc:.2f}"
                    trow.append(txt)
            table_rows.append(trow)
        tbl = ax_t.table(cellText=table_rows, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 2.0)
        for (ri, ci), cell_obj in tbl.get_celld().items():
            if ri == 0 or ci == 0:
                cell_obj.set_text_props(fontweight="bold")

        # OOD Greek reduction panel (grouped by greek, one bar per regime)
        width = 0.8 / max(1, len(regimes))
        for gi, g in enumerate(greeks):
            for ri, regime in enumerate(regimes):
                gr = _greek_row(greek, greek_arm, g, regime) or {}
                red, err = (_num(gr.get("reduction_vs_standard_pinn_mean")),
                            _num(gr.get("reduction_vs_standard_pinn_std")))
                x = gi + (ri - (len(regimes) - 1) / 2) * width
                ax_g.bar([x], [red if red is not None else 0.0], width=width,
                         yerr=[err or 0.0], capsize=2,
                         color=hb._OVERLAY_COLORS[ri + 3],
                         label=regime if gi == 0 else None)
        ax_g.axhline(0.15, ls="--", lw=1.0, color=_NEUTRAL)
        ax_g.text(len(greeks) - 0.5, 0.155, "0.15 threshold", fontsize=7,
                  ha="right", va="bottom", color=_NEUTRAL)
        ax_g.set_xticks(range(len(greeks)))
        ax_g.set_xticklabels([g.capitalize() for g in greeks])
        ax_g.set_ylabel("reduction vs standard_pinn")
        ax_g.set_title(f"OOD Greek accuracy ({_method_label(greek_arm)})", fontsize=10)
        ax_g.legend(loc="upper right")

        fig.suptitle("E1 · supervision ladder & residual×supervision factorial",
                     fontsize=11, y=1.02)
        png_path = _savefig(fig, os.path.join(out_dir, f"{stem}.png"))
    return {"png": png_path, "csv": csv_path}


# ===========================================================================
# E2 — mechanism: CVaR vs tc per arm, {in-model,misspec}x{tc0,tc>0} inset, T_ex
# ===========================================================================

def exhibit_e2(hedging_agg_csv, out_dir, *, arms=MECH_ARMS, focal=FOCAL_ARM,
               baseline=BASELINE_ARM, misspec=MISSPEC_CELL, inmodel=INMODEL_CELL,
               tex_tc=CONFIRMATORY_TC, stem="e2_mechanism") -> dict:
    """Mechanism figure. Main panel: CVaR95 vs tc per arm on the misspec cell.
    Inset 2x2: focal-vs-baseline CVaR difference over {in-model,misspec} x
    {tc=0, tc>0} (the contract's in-model×cost κ corner). T_ex panel: excess
    turnover per arm with seed error bars (the pre-registered cost falsifier)."""
    agg = read_csv(_require(hedging_agg_csv))
    tiers = _tc_tiers(agg, misspec)
    tc_hi = tiers[-1] if tiers else tex_tc

    csv_rows: list[dict] = []
    # CVaR vs tc
    for method in arms:
        for tc in tiers:
            row = _by_method(agg, misspec, tc).get(method, {})
            cv = _num(row.get("cvar_mean"))
            if cv is None:
                continue
            csv_rows.append({"panel": "cvar_vs_tc", "method": method, "tc": tc,
                             "value": cv, "err": _num(row.get("cvar_seed_std"))})
    # T_ex per arm (positions are tc-independent; read at tex_tc)
    for method in arms:
        row = _by_method(agg, misspec, tex_tc).get(method, {})
        te = _num(row.get("t_ex_mean"))
        csv_rows.append({"panel": "t_ex", "method": method, "tc": tex_tc,
                         "value": te, "err": _num(row.get("t_ex_seed_std"))})

    # 2x2 focal-vs-baseline CVaR diff
    def _diff(cell, tc):
        bym = _by_method(agg, cell, tc)
        frow, brow = bym.get(focal, {}), bym.get(baseline, {})
        d = _num(frow.get("pnl_vs_baseline_cvar_diff_mean"))
        if d is None:                                   # fall back to mean diff
            fc, bc = _num(frow.get("cvar_mean")), _num(brow.get("cvar_mean"))
            d = (fc - bc) if (fc is not None and bc is not None) else None
        return d

    tc_lo = 0.0 if 0.0 in tiers else (tiers[0] if tiers else 0.0)
    quad = {("in-model", "tc=0"): _diff(inmodel, tc_lo),
            ("in-model", "tc>0"): _diff(inmodel, tc_hi),
            ("misspec", "tc=0"): _diff(misspec, tc_lo),
            ("misspec", "tc>0"): _diff(misspec, tc_hi)}
    for (regime, bucket), d in quad.items():
        csv_rows.append({"panel": "inset_2x2", "method": f"{focal}-vs-{baseline}",
                         "tc": bucket, "value": d, "err": "", "regime": regime})

    cols = ["panel", "method", "tc", "regime", "value", "err"]
    csv_path = _write_csv(os.path.join(out_dir, f"{stem}.csv"), cols, csv_rows)

    with matplotlib.rc_context(_STYLE):
        fig = plt.figure(figsize=(11.0, 6.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                              hspace=0.42, wspace=0.28)
        ax_main = fig.add_subplot(gs[0, :])
        ax_tex = fig.add_subplot(gs[1, 0])
        ax_2x2 = fig.add_subplot(gs[1, 1])

        for method in arms:
            xs, ys, es = [], [], []
            for tc in tiers:
                row = _by_method(agg, misspec, tc).get(method, {})
                cv = _num(row.get("cvar_mean"))
                if cv is None:
                    continue
                xs.append(tc); ys.append(cv); es.append(_num(row.get("cvar_seed_std")) or 0.0)
            if xs:
                ax_main.errorbar(xs, ys, yerr=es, marker="o", ms=5, capsize=3,
                                 color=_method_color(method), label=_method_label(method))
        ax_main.set_xlabel("transaction cost (fraction)")
        ax_main.set_ylabel("misspec CVaR95 (loss)")
        ax_main.set_title("CVaR95 vs transaction cost, per arm (combined m=1.0)", fontsize=10)
        ax_main.legend(ncol=2, loc="best")

        # T_ex bars
        tex_methods = [m for m in arms if m != ORACLE_ARM]
        tvals, terr, tcol = [], [], []
        for m in tex_methods:
            row = _by_method(agg, misspec, tex_tc).get(m, {})
            # MISSING -> NaN, never 0.0: matplotlib omits a NaN bar, whereas
            # T_ex = 0 is the cost channel's confirming reading ("trades exactly
            # like the oracle"). Same handling as the 2x2 inset below.
            tvals.append(_nan_if_missing(row.get("t_ex_mean")))
            terr.append(_nan_if_missing(row.get("t_ex_seed_std")))
            tcol.append(_method_color(m))
        ax_tex.bar(range(len(tex_methods)), tvals, yerr=terr, capsize=3, color=tcol)
        ax_tex.axhline(0.0, lw=0.8, color="k")
        ax_tex.set_xticks(range(len(tex_methods)))
        ax_tex.set_xticklabels([_method_label(m) for m in tex_methods],
                               rotation=25, ha="right", fontsize=7)
        ax_tex.set_ylabel("T_ex (excess turnover)")
        ax_tex.set_title(f"T_ex vs oracle (tc={tex_tc})", fontsize=10)

        # 2x2 inset heatmap of focal-vs-baseline diff
        regimes2, buckets2 = ["in-model", "misspec"], ["tc=0", "tc>0"]
        M = [[quad[(r, b)] if quad[(r, b)] is not None else float("nan")
              for b in buckets2] for r in regimes2]
        finite = [v for r in M for v in r if math.isfinite(v)]
        vmax = max((abs(v) for v in finite), default=1.0) or 1.0
        im = ax_2x2.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax_2x2.set_xticks(range(2)); ax_2x2.set_xticklabels(buckets2)
        ax_2x2.set_yticks(range(2)); ax_2x2.set_yticklabels(regimes2)
        ax_2x2.grid(False)
        for i in range(2):
            for j in range(2):
                v = M[i][j]
                ax_2x2.text(j, i, "n/a" if not math.isfinite(v) else f"{v:.3g}",
                            ha="center", va="center", fontsize=8)
        ax_2x2.set_title(f"ΔCVaR {_method_label(focal)} − {_method_label(baseline)}",
                         fontsize=9)
        fig.colorbar(im, ax=ax_2x2, fraction=0.046, pad=0.04)

        fig.suptitle("E2 · mechanism (cost vs robustness channel)", fontsize=11, y=0.98)
        png_path = _savefig(fig, os.path.join(out_dir, f"{stem}.png"))
    return {"png": png_path, "csv": csv_path}


# ===========================================================================
# E3 — dose-response (THE money figure) + colourblind-safe variant
# ===========================================================================

def _dose_pts(dose_rows):
    """(on-axis points sorted by label_error, reference value)."""
    pts, ref = [], None
    for r in dose_rows:
        is_ref = str(r.get("is_reference")) == "True"
        x = _num(r.get("label_error"))
        y = _num(r.get("cvar_mean"))
        if is_ref:
            if y is not None:
                ref = (str(r.get("arm")), y)
            continue
        if x is None or y is None:
            continue
        pts.append({"arm": str(r.get("arm")), "x": x, "y": y,
                    "err": _num(r.get("cvar_seed_std")) or 0.0,
                    "iso": _num(r.get("isotonic_fit")),
                    "src": str(r.get("label_source"))})
    pts.sort(key=lambda p: p["x"])
    return pts, ref


def _draw_e3(ax, pts, ref, palette):
    base_c, bs_c, shuf_c, ref_c = palette[0], palette[1], palette[5], palette[7]
    # generic sigma points; bs_gamma / shuffled highlighted with distinct markers
    for p in pts:
        if p["arm"] == "bs_gamma":
            c, mk, lbl = bs_c, "D", "bs_gamma"
        elif p["arm"] == "shuffled":
            c, mk, lbl = shuf_c, "s", "shuffled"
        else:
            c, mk, lbl = base_c, "o", None
        ax.errorbar([p["x"]], [p["y"]], yerr=[p["err"]], marker=mk, ms=7,
                    color=c, capsize=3, lw=0, label=lbl, zorder=3)
    # isotonic fit line (non-decreasing, ordered by label error)
    iso_pts = [(p["x"], p["iso"]) for p in pts if p["iso"] is not None]
    if len(iso_pts) >= 2:
        ax.plot([a for a, _ in iso_pts], [b for _, b in iso_pts], color=base_c,
                lw=1.8, ls="-", label="isotonic fit", zorder=2)
    # bs_gamma label callout
    for p in pts:
        if p["arm"] == "bs_gamma":
            ax.annotate("cheap BS-Γ labels", (p["x"], p["y"]), fontsize=7,
                        xytext=(6, 8), textcoords="offset points", color=bs_c)
        if p["arm"] == "shuffled":
            ax.annotate("shuffled (right edge)", (p["x"], p["y"]), fontsize=7,
                        xytext=(-6, 8), textcoords="offset points", ha="right",
                        color=shuf_c)
    # gradient-penalty-only reference line
    if ref is not None:
        ax.axhline(ref[1], ls="--", lw=1.2, color=ref_c,
                   label="grad-penalty-only ref")
    ax.set_xlabel("measured γ-label error  ||γ−γ_ref|| / ||γ_ref||")
    ax.set_ylabel("misspec CVaR95 (loss)")
    ax.legend(loc="best")


def exhibit_e3(dose_csv, out_dir, *, stem="e3_dose_response") -> dict:
    """Gamma-label-noise dose-response: x = measured γ-label error (P13
    dose_response.csv), y = misspec CVaR95, with the P13 isotonic fit line, the
    BS-Γ point labelled, the shuffled arm at the right edge and the
    gradient-penalty-only horizontal reference. Emits the house-style PNG, a
    colourblind-safe PNG, and the backing CSV."""
    dose = read_csv(_require(dose_csv))
    pts, ref = _dose_pts(dose)

    csv_rows = []
    for p in pts:
        role = ("bs_gamma" if p["arm"] == "bs_gamma"
                else "shuffled" if p["arm"] == "shuffled" else "point")
        csv_rows.append({"arm": p["arm"], "label_source": p["src"], "role": role,
                         "label_error": p["x"], "cvar_mean": p["y"],
                         "cvar_seed_std": p["err"], "isotonic_fit": p["iso"]})
    if ref is not None:
        csv_rows.append({"arm": ref[0], "label_source": "", "role": "reference",
                         "label_error": "", "cvar_mean": ref[1],
                         "cvar_seed_std": "", "isotonic_fit": ""})
    cols = ["arm", "label_source", "role", "label_error", "cvar_mean",
            "cvar_seed_std", "isotonic_fit"]
    csv_path = _write_csv(os.path.join(out_dir, f"{stem}.csv"), cols, csv_rows)

    out = {"csv": csv_path}
    for suffix, palette in (("", hb._OVERLAY_COLORS), ("_cb", _CB_SAFE)):
        with matplotlib.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7.4, 5.0))
            _draw_e3(ax, pts, ref, palette)
            title = "E3 · γ-label-noise dose-response"
            ax.set_title(title + (" (colourblind-safe)" if suffix else ""), fontsize=10)
            key = "png_cb" if suffix else "png"
            out[key] = _savefig(fig, os.path.join(out_dir, f"{stem}{suffix}.png"))
    return out


# ===========================================================================
# E4 — decomposition: gap-closed, cost/directional split, vanna inset, scatter
# ===========================================================================

def exhibit_e4(hedging_agg_csv, greek_agg_csv, out_dir, *,
               arms=("standard_pinn", "rung1", "rung2", "rung3", "sans_pde"),
               cell=MISSPEC_CELL, tc=CONFIRMATORY_TC, focal=FOCAL_ARM,
               band=SMOOTHED_ARM, vanna_regimes=("near_feller", "strong_neg_corr"),
               stem="e4_decomposition") -> dict:
    """Decomposition panel. (a) fraction-of-gap-closed bar per arm; (b) cost vs
    directional PnL split (stacked); (c) vanna-accuracy inset (P11 reductions);
    (d) band-vs-Sobolev tracking-error/cost scatter (smoothed band vs the focal
    Sobolev arm and the sans-PDE arm)."""
    agg = read_csv(_require(hedging_agg_csv))
    greek = read_csv(_require(greek_agg_csv))
    bym = _by_method(agg, cell, tc)

    csv_rows = []
    for method in arms:
        row = bym.get(method, {})
        csv_rows.append({"panel": "decomp", "method": method,
                         "gap_closed_mean": _num(row.get("gap_closed_mean")),
                         "tc_component_mean": _num(row.get("tc_component_mean")),
                         "directional_component_mean": _num(row.get("directional_component_mean")),
                         "cvar_mean": _num(row.get("cvar_mean")),
                         "turnover_mean": _num(row.get("turnover_mean"))})
    for regime in vanna_regimes:
        gr = _greek_row(greek, focal, "vanna", regime) or {}
        csv_rows.append({"panel": "vanna", "method": focal, "regime": regime,
                         "vanna_reduction_mean": _num(gr.get("reduction_vs_standard_pinn_mean")),
                         "vanna_reduction_std": _num(gr.get("reduction_vs_standard_pinn_std"))})
    # band-vs-Sobolev scatter members
    scatter_methods = [m for m in (band, focal, "sans_pde") if m in bym]
    for m in scatter_methods:
        row = bym.get(m, {})
        csv_rows.append({"panel": "scatter", "method": m,
                         "tc_component_mean": _num(row.get("tc_component_mean")),
                         "cvar_mean": _num(row.get("cvar_mean"))})

    cols = ["panel", "method", "regime", "gap_closed_mean", "tc_component_mean",
            "directional_component_mean", "cvar_mean", "turnover_mean",
            "vanna_reduction_mean", "vanna_reduction_std"]
    csv_path = _write_csv(os.path.join(out_dir, f"{stem}.csv"), cols, csv_rows)

    with matplotlib.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
        ax_gap, ax_split, ax_vanna, ax_scatter = axes.ravel()

        # (a) gap closed
        gvals = [_nan_if_missing(bym.get(m, {}).get("gap_closed_mean")) for m in arms]
        ax_gap.bar(range(len(arms)), gvals,
                   color=[_method_color(m) for m in arms])
        ax_gap.axhline(1.0, ls="--", lw=1.0, color=_NEUTRAL)
        ax_gap.set_xticks(range(len(arms)))
        ax_gap.set_xticklabels([_method_label(m) for m in arms], rotation=25,
                               ha="right", fontsize=7)
        ax_gap.set_ylabel("gap_closed_mean")
        ax_gap.set_title("(a) fraction of oracle gap closed", fontsize=10)

        # (b) cost vs directional split (stacked)
        costs = [_nan_if_missing(bym.get(m, {}).get("tc_component_mean")) for m in arms]
        dirs = [_nan_if_missing(bym.get(m, {}).get("directional_component_mean"))
                for m in arms]
        x = range(len(arms))
        ax_split.bar(x, dirs, color="#1baf7a", label="directional PnL")
        ax_split.bar(x, [-c for c in costs], bottom=dirs, color="#e34948",
                     label="−transaction cost")
        ax_split.axhline(0.0, lw=0.8, color="k")
        ax_split.set_xticks(list(x))
        ax_split.set_xticklabels([_method_label(m) for m in arms], rotation=25,
                                 ha="right", fontsize=7)
        ax_split.set_ylabel("mean PnL component")
        ax_split.set_title("(b) cost vs directional split", fontsize=10)
        ax_split.legend(loc="best")

        # (c) vanna accuracy inset
        vvals, verr = [], []
        for regime in vanna_regimes:
            gr = _greek_row(greek, focal, "vanna", regime) or {}
            vvals.append(_nan_if_missing(gr.get("reduction_vs_standard_pinn_mean")))
            verr.append(_nan_if_missing(gr.get("reduction_vs_standard_pinn_std")))
        ax_vanna.bar(range(len(vanna_regimes)), vvals, yerr=verr, capsize=3,
                     color=[hb._OVERLAY_COLORS[i + 3] for i in range(len(vanna_regimes))])
        ax_vanna.axhline(0.15, ls="--", lw=1.0, color=_NEUTRAL)
        ax_vanna.set_xticks(range(len(vanna_regimes)))
        ax_vanna.set_xticklabels(vanna_regimes, fontsize=7)
        ax_vanna.set_ylabel("vanna reduction")
        ax_vanna.set_title(f"(c) vanna accuracy ({_method_label(focal)})", fontsize=10)

        # (d) band vs Sobolev tracking-error / cost scatter
        for m in scatter_methods:
            row = bym.get(m, {})
            cost = _num(row.get("tc_component_mean"))
            track = _num(row.get("cvar_mean"))
            if cost is None or track is None:
                continue
            ax_scatter.scatter([cost], [track], s=70, color=_method_color(m),
                               label=_method_label(m), zorder=3)
            ax_scatter.annotate(_method_label(m), (cost, track), fontsize=7,
                                xytext=(5, 4), textcoords="offset points")
        ax_scatter.set_xlabel("transaction-cost component (mean)")
        ax_scatter.set_ylabel("CVaR95 (tracking risk)")
        ax_scatter.set_title("(d) band vs Sobolev: cost vs tracking risk", fontsize=10)
        ax_scatter.legend(loc="best")

        fig.suptitle("E4 · decomposition (cost / directional / Greek channels)",
                     fontsize=11, y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png_path = _savefig(fig, os.path.join(out_dir, f"{stem}.png"))
    return {"png": png_path, "csv": csv_path}


# ===========================================================================
# Overlay — thin wrapper over the engine's delta-path overlay diagnostic
# ===========================================================================

def overlay_delta_paths(cfg: dict, providers: dict, out_path: str, *,
                        methods=(FOCAL_ARM, BASELINE_ARM, ORACLE_ARM),
                        direction: str | None = "combined", magnitude: float = 1.0,
                        seed: int | None = None) -> str:
    """Delegate to Hedging_backtest.run_delta_overlay for (rung3, standard_pinn,
    oracle) on the combined misspec cell. Subsets `providers` to `methods` and
    emits them in the canonical method order — run_delta_overlay assigns
    _OVERLAY_COLORS positionally, so a stable order gives a stable, principled
    colour assignment (oracle / standard_pinn keep their E1-E4 colour). This is
    the ONE exhibit that simulates paths: a convenience wrapper over the engine
    diagnostic, not a pure-from-results exhibit."""
    order = {m: i for i, m in enumerate(_METHOD_COLOR_ORDER)}
    present = sorted((m for m in methods if m in providers),
                     key=lambda m: order.get(m, len(order)))
    if not present:
        raise KeyError(f"none of {list(methods)} present in providers "
                       f"{sorted(providers)}")
    sub = {m: providers[m] for m in present}
    return hb.run_delta_overlay(cfg, sub, out_path, direction=direction,
                                magnitude=magnitude, seed=seed)


# ---------------------------------------------------------------------------
# CLI — regenerate every pure-from-results exhibit from one results root
# ---------------------------------------------------------------------------

def regenerate_all(results_root: str, out_dir: str, *,
                   hedging_agg=None, greek_agg=None, dose_csv=None) -> dict:
    """Regenerate E1-E4 from artifacts under `results_root`. Path overrides let a
    caller point at non-default filenames; defaults use the module constants."""
    root = results_root
    ha = hedging_agg or os.path.join(root, HEDGING_AGG)
    ga = greek_agg or os.path.join(root, GREEK_AGG)
    dc = dose_csv or os.path.join(root, "tables", DOSE_CSV)
    return {"e1": exhibit_e1(ha, ga, out_dir),
            "e2": exhibit_e2(ha, out_dir),
            "e3": exhibit_e3(dc, out_dir),
            "e4": exhibit_e4(ha, ga, out_dir)}


def main(argv: list[str] | None = None) -> dict:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", required=True,
                    help="dir holding the P11/P12 agg CSVs (tables/ for P13 dose)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hedging-agg", default=None)
    ap.add_argument("--greek-agg", default=None)
    ap.add_argument("--dose-csv", default=None)
    args = ap.parse_args(argv)
    res = regenerate_all(args.results_root, args.out_dir,
                         hedging_agg=args.hedging_agg, greek_agg=args.greek_agg,
                         dose_csv=args.dose_csv)
    for name, paths in res.items():
        for kind, p in paths.items():
            print(f"{name} {kind}: {p}")
    return res


if __name__ == "__main__":
    main()
