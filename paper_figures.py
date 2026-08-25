"""Paper figures for the STODY submission (F1 lambda dissection, F2 region reversal).

Pure function of the frozen result artifacts: reads only ``results/`` CSVs and writes
PNGs into ``figures/``. No re-scoring, no re-fitting, no registered artifact touched.

F1  per-arm misspecified CVaR95 against lambda_pde beside the headline contrast, the
    exhibit for "only the baseline moves".
F2  held-out delta RMSE on the full contract grid against the same metric restricted to
    the hedging box, one slope line per arm, the exhibit for the ranking reversal.
F3  gamma-label dose-response on both axes: accuracy degrades monotonically in the dose
    while CVaR95 is flat at both registered cost tiers.
F4  mechanism: CVaR95 measured against the oracle hedger across the cost sweep (the
    inversion) beside excess turnover at the registered tier (the under-trading).
F5  information-matching saturation of the price-only network, with rung 3 marked.

F3-F5 replace the E2-E4 exhibits of ``exhibits.py`` for manuscript use: those were built
for the earlier framing and E4's gap-closed panel is empty at this cell, since the
statistic is undefined where the oracle does not beat the baseline.

Style follows ``exhibits.py``: Okabe-Ito colourblind-safe hues assigned in fixed order,
recessive axes, direct labels, one linear scale per panel.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "figures")

# Okabe-Ito, assigned by entity and never cycled.
_OI = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
_NEUTRAL = "#6E6E6E"

_RC = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# lambda_pde -> directory holding the confirmatory hedging aggregate for that weight.
_LAMBDA_HEDGE_DIRS: List[Tuple[float, str]] = [
    (0.0, "results/hedging_robustness/confirmatory"),
    (1e-4, "results/lambda_sweep/hedging/lam0p0001/confirmatory"),
    (3e-4, "results/lambda_sweep/hedging/lam0p0003/confirmatory"),
    (1e-3, "results/lambda_sweep/hedging/lam0p001/confirmatory"),
    (3e-3, "results/lambda_sweep/hedging/lam0p003/confirmatory"),
    (1e-2, "results/hedging/confirmatory"),
]

# Decade ticks only: the 3e-4 / 3e-3 rungs are plotted but not labelled, which keeps the
# axis legible at column width.
_XTICKS = [0.0, 1e-4, 1e-3, 1e-2]
_XLABELS = ["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$"]

_ARM_STYLE = {
    "standard_pinn": ("standard PINN", _OI["vermillion"], "o", "-"),
    "rung3": ("rung 3 (+$\\Delta$+$\\Gamma$+$\\nu$)", _OI["blue"], "s", "-"),
    "oracle": ("oracle $\\Delta$", _NEUTRAL, None, "--"),
}


def _rows(path: str) -> List[Dict[str, str]]:
    with open(os.path.join(ROOT, path), newline="") as fh:
        return list(csv.DictReader(fh))


def _cvar_at(agg_path: str, method: str, tc: float = 0.0) -> float:
    """Misspecified (combined, magnitude 1.0) CVaR95 of one arm at one cost tier."""
    for r in _rows(agg_path):
        if (
            r["method"] == method
            and r["in_model"] == "False"
            and r["sweep"] == "perturbation"
            and abs(float(r["magnitude"]) - 1.0) < 1e-12
            and abs(float(r["tc"]) - tc) < 1e-12
        ):
            return float(r["cvar_mean"])
    raise KeyError(f"{method} tc={tc} not found in {agg_path}")


def figure_lambda(out_path: str) -> str:
    lambdas = [lam for lam, _ in _LAMBDA_HEDGE_DIRS]
    levels = {
        arm: [_cvar_at(os.path.join(d, "headline_delta_only_agg.csv"), arm) for _, d in _LAMBDA_HEDGE_DIRS]
        for arm in _ARM_STYLE
    }

    merged = _rows("results/lambda_sweep/headline_vs_lambda_pde_merged.csv")
    eff = {tc: {} for tc in (0.0, 0.01)}
    for r in merged:
        tc = float(r["tc"])
        if tc in eff:
            eff[tc][float(r["lambda_pde"])] = 100.0 * float(r["rel"])

    with plt.rc_context(_RC):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 2.55))

        for arm, (label, color, marker, ls) in _ARM_STYLE.items():
            ax_l.plot(
                lambdas, levels[arm], color=color, ls=ls,
                marker=marker, markersize=4.0, label=label,
                zorder=3 if arm != "oracle" else 2,
            )
        ax_l.set_xscale("symlog", linthresh=1e-4)
        ax_l.set_xlim(left=-2e-5)
        ax_l.set_xticks(_XTICKS)
        ax_l.set_xticklabels(_XLABELS)
        ax_l.set_xlabel("$\\lambda_{\\mathrm{pde}}$")
        ax_l.set_ylabel("misspecified $\\mathrm{CVaR}_{95}$")
        ax_l.set_title("(a) arm levels: only the baseline moves")
        ax_l.annotate(
            "registered", xy=(1e-2, levels["standard_pinn"][-1]), xytext=(-4, 4),
            textcoords="offset points", ha="right", va="bottom", fontsize=7, color=_NEUTRAL,
        )
        ax_l.legend(loc="upper left")

        for tc, color, marker, label in (
            (0.0, _OI["green"], "o", "0% cost"),
            (0.01, _OI["orange"], "^", "1% cost (registered tier)"),
        ):
            xs = sorted(eff[tc])
            ax_r.plot(xs, [eff[tc][x] for x in xs], color=color, marker=marker,
                      markersize=4.0, label=label, zorder=3)
        ax_r.axhline(0.0, lw=0.8, color="k", zorder=1)
        ax_r.axhline(10.0, ls=":", lw=1.0, color=_NEUTRAL, zorder=1)
        ax_r.text(0.0, 10.8, "registered 10% bar", fontsize=6.8, color=_NEUTRAL,
                  va="bottom", ha="left")
        ax_r.annotate(
            "criterion optimum\n$+7.4\\%$", xy=(1e-4, eff[0.0][1e-4]), xytext=(5, -5),
            textcoords="offset points", fontsize=6.8, color=_NEUTRAL, va="top",
        )
        ax_r.annotate(
            "registered $\\lambda_{\\mathrm{pde}}$\n$+31.5\\%$", xy=(1e-2, eff[0.0][1e-2]),
            xytext=(-4, -2), textcoords="offset points", fontsize=6.8, color=_NEUTRAL,
            va="top", ha="right",
        )
        ax_r.set_xscale("symlog", linthresh=1e-4)
        ax_l.set_xlim(left=-2e-5)
        ax_r.set_xticks(_XTICKS)
        ax_r.set_xticklabels(_XLABELS)
        ax_r.set_xlabel("$\\lambda_{\\mathrm{pde}}$")
        ax_r.set_ylabel("rung 3 vs. standard PINN (%)")
        ax_r.set_title("(b) the headline contrast")
        ax_r.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_region(out_path: str) -> str:
    rows = _rows("results/eval_greeks_hedgeslice/full_vs_hedgebox.csv")
    arms = [
        ("standard_pinn", "standard PINN", _OI["vermillion"]),
        ("feedforward", "price-only net", _OI["skyblue"]),
        ("rung1", "rung 1", _OI["purple"]),
        ("rung3", "rung 3", _OI["blue"]),
    ]
    regimes = [("near_feller", "near_feller anchor"), ("strong_neg_corr", "strong_neg_corr anchor")]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
        for ax, (regime, regime_label) in zip(axes, regimes):
            for arm, label, color in arms:
                rec = next(
                    r for r in rows
                    if r["arm"] == arm and r["regime"] == regime and r["greek"] == "delta"
                )
                full, box = float(rec["rmse_full"]), float(rec["rmse_hedge"])
                ax.plot([0, 1], [full, box], color=color, marker="o", markersize=4.5,
                        label=label, zorder=3)
                ax.annotate(label, xy=(1, box), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=7, color=color)
            ax.set_yscale("log")
            ax.set_xlim(-0.18, 1.62)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["full grid\n(accuracy metric)", "hedging box\n(where hedging happens)"])
            ax.set_title(regime_label)
        axes[0].set_ylabel("held-out $\\Delta$ RMSE")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def _sweep_cvar(direction: str = "combined", tc: float = 0.0) -> Dict[str, Tuple[float, float]]:
    """Misspecified CVaR95 mean and seed sd per arm from the full sweep."""
    out: Dict[str, Tuple[float, float]] = {}
    for r in _rows("results/hedging_sweep/full_sweep/headline_delta_only_agg.csv"):
        if (
            r["sweep"] == "perturbation"
            and r["direction"] == direction
            and r["in_model"] == "False"
            and abs(float(r["magnitude"]) - 1.0) < 1e-12
            and abs(float(r["tc"]) - tc) < 1e-12
        ):
            out[r["method"]] = (float(r["cvar_mean"]), float(r["cvar_seed_std"]))
    return out


def figure_dose(out_path: str) -> str:
    """Dose-response on both axes: label correctness buys accuracy, not CVaR95."""
    # measured label error (e3 exhibit) keyed by arm; sigma_000 is rung 3's own label set.
    err = {r["arm"]: float(r["label_error"]) for r in _rows("results/exhibits/e3_dose_response.csv")
           if r["label_error"] != ""}
    err["rung3"] = err.pop("sigma_000")

    acc = {}
    for r in _rows("results/eval_greeks_full/ood_param_greeks_agg.csv"):
        if r["regime"] == "near_feller" and r["greek"] == "gamma":
            acc[r["arm"]] = (float(r["rel_rmse_mean"]), float(r["rel_rmse_std"]))

    dose = ["rung3", "sigma_010", "sigma_025", "sigma_050"]
    cv0, cv1 = _sweep_cvar(tc=0.0), _sweep_cvar(tc=0.01)

    def series(arms, table):
        xs = [err[a] for a in arms]
        ys = [table[a][0] for a in arms]
        es = [table[a][1] for a in arms]
        return xs, ys, es

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
        panels = (
            (axes[0], acc, "(a) delivered accuracy", "OOD $\\Gamma$ rel. RMSE"),
            (axes[1], cv0, "(b) hedging, $0\\%$ cost", "misspec. $\\mathrm{CVaR}_{95}$"),
            (axes[2], cv1, "(c) hedging, $1\\%$ cost", "misspec. $\\mathrm{CVaR}_{95}$"),
        )
        for ax, table, title, ylabel in panels:
            xs, ys, es = series(dose, table)
            ax.errorbar(xs, ys, yerr=es, color=_OI["blue"], marker="o", markersize=4.0,
                        capsize=2.5, lw=1.6, label="true $\\Gamma$ labels $+$ noise", zorder=3)
            bx, by = err["bs_gamma"], table["bs_gamma"][0]
            ax.errorbar([bx], [by], yerr=[table["bs_gamma"][1]], color=_OI["green"],
                        marker="D", markersize=5.0, capsize=2.5, lw=0, elinewidth=1.4,
                        label="Black-Scholes $\\Gamma$ label", zorder=4)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("measured $\\Gamma$-label error")
            ax.set_xlim(-0.05, 0.58)
        # The biased label is worst-but-one on accuracy and best on both hedging panels,
        # so no panel has a free corner; the legend is shared and sits below the row.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_mechanism(out_path: str) -> str:
    """Cost inversion against the oracle, and the turnover the falsifier asked for."""
    rows = _rows("results/hedging_atc/confirmatory/headline_delta_only_agg.csv")
    sel = [r for r in rows if r["in_model"] == "False" and abs(float(r["magnitude"]) - 1.0) < 1e-12]
    tcs = sorted({float(r["tc"]) for r in sel})
    oracle = {float(r["tc"]): float(r["cvar_mean"]) for r in sel if r["method"] == "oracle"}

    arms = [
        ("standard_pinn", "standard PINN", _OI["vermillion"], "-", "o"),
        ("standard_pinn_smoothed", "standard PINN $+$ band", _OI["vermillion"], "--", "o"),
        ("rung1", "rung 1 ($+\\Delta$)", _OI["purple"], "-", "^"),
        ("rung3", "rung 3 ($+\\Delta{+}\\Gamma{+}\\nu$)", _OI["blue"], "-", "s"),
    ]

    def at(method, tc, col="cvar_mean"):
        return float(next(r for r in sel if r["method"] == method
                          and abs(float(r["tc"]) - tc) < 1e-12)[col])

    with plt.rc_context(_RC):
        fig, (ax_c, ax_t) = plt.subplots(1, 2, figsize=(7.0, 2.55),
                                         gridspec_kw={"width_ratios": [1.5, 1.0]})
        for m, label, color, ls, mk in arms:
            ys = [at(m, tc) - oracle[tc] for tc in tcs]
            ax_c.plot([100 * t for t in tcs], ys, color=color, ls=ls, marker=mk,
                      markersize=4.0, markerfacecolor="white" if ls == "--" else color,
                      label=label, zorder=3)
        ax_c.axhline(0.0, color=_NEUTRAL, lw=1.2, zorder=2)
        ax_c.text(2.0, 0.06, "oracle $\\Delta$", fontsize=6.8, color=_NEUTRAL,
                  ha="right", va="bottom")
        ax_c.set_xlabel("proportional transaction cost (%)")
        ax_c.set_ylabel("$\\mathrm{CVaR}_{95}$ minus oracle")
        ax_c.set_title("(a) the cost regime inverts the ranking")
        ax_c.legend(loc="lower left")

        tex_arms = [(m, lab, c) for m, lab, c, _, _ in arms]
        vals = [at(m, 0.01, "t_ex_mean") for m, _, _ in tex_arms]
        errs = [at(m, 0.01, "t_ex_seed_std") for m, _, _ in tex_arms]
        bars = ax_t.bar(range(len(vals)), vals, yerr=errs, capsize=3,
                        color=[c for _, _, c in tex_arms], width=0.62, zorder=3)
        # Same hatching convention as the dashed line in (a): the band is a policy on the
        # baseline's delta, not a different learned arm.
        bars[1].set_hatch("//")
        bars[1].set_edgecolor("white")
        ax_t.axhline(0.0, color=_NEUTRAL, lw=1.2, zorder=2)
        ax_t.text(len(vals) - 0.5, 0.03, "oracle turnover", fontsize=6.8, color=_NEUTRAL,
                  ha="right", va="bottom")
        ax_t.set_xticks(range(len(vals)))
        ax_t.set_xticklabels(["std\nPINN", "std $+$\nband", "rung 1", "rung 3"], fontsize=7)
        ax_t.set_ylabel("$T_{\\mathrm{ex}}$ (excess turnover)")
        ax_t.set_title("(b) the baseline under-trades")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_info_matching(out_path: str) -> str:
    rows = _rows("results/info_matching/saturation_curve.csv")
    base = [r for r in rows if float(r["width_mult"]) == 1.0]
    wide = [r for r in rows if float(r["width_mult"]) == 2.0]
    xs = [float(r["multiplier"]) for r in base]
    ys = [float(r["gamma_rel_rmse_mean"]) for r in base]
    es = [float(r["gamma_rel_rmse_std"]) for r in base]

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        ax.errorbar(xs, ys, yerr=es, color=_OI["skyblue"], marker="o", markersize=4.0,
                    capsize=2.5, label="price-only network", zorder=3)
        for r in wide:
            ax.errorbar([float(r["multiplier"])], [float(r["gamma_rel_rmse_mean"])],
                        yerr=[float(r["gamma_rel_rmse_std"])], color=_OI["orange"],
                        marker="s", markersize=4.5, capsize=2.5, lw=0, elinewidth=1.4,
                        label="same budget, twice the width", zorder=4)
        plateau = float(base[0]["plateau_multiplier"])
        ax.axvline(plateau, ls="--", lw=0.9, color=_NEUTRAL, zorder=1)
        ax.text(plateau - 0.08, 0.79, "plateau rule fires", fontsize=6.8, color=_NEUTRAL,
                ha="right", va="top")
        ax.set_xticks([int(x) for x in xs])
        ax.axhline(0.0446, ls=":", lw=1.0, color=_OI["blue"], zorder=1)
        ax.text(5.0, 0.055, "rung 3", fontsize=6.8, color=_OI["blue"], ha="right", va="bottom")
        ax.set_ylim(0.0, 0.82)
        ax.set_xlabel("price-point budget multiplier $m$")
        ax.set_ylabel("validation $\\Gamma$ rel. RMSE")
        ax.legend(loc="center left")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(figure_lambda(os.path.join(OUT_DIR, "f1_lambda_pde.png")))
    print(figure_region(os.path.join(OUT_DIR, "f2_region_reversal.png")))
    print(figure_dose(os.path.join(OUT_DIR, "f3_dose_response.png")))
    print(figure_mechanism(os.path.join(OUT_DIR, "f4_mechanism.png")))
    print(figure_info_matching(os.path.join(OUT_DIR, "f5_info_matching.png")))
    print(figure_coverage_region(os.path.join(OUT_DIR, "f6_coverage_region.png")))
    print(figure_label_controls(os.path.join(OUT_DIR, "f7_label_controls.png")))
    print(figure_pipeline(os.path.join(OUT_DIR, "f8_pipeline.png")))
    print(figure_problem(os.path.join(OUT_DIR, "f0_problem.png")))

    # Console echo of the two tables the figures encode, for cross-checking the manuscript.
    print("\nlambda_pde   standard_pinn   rung3    oracle")
    for lam, d in _LAMBDA_HEDGE_DIRS:
        agg = os.path.join(d, "headline_delta_only_agg.csv")
        vals = [_cvar_at(agg, a) for a in ("standard_pinn", "rung3", "oracle")]
        print(f"{lam:<12g} {vals[0]:.4f}        {vals[1]:.4f}   {vals[2]:.4f}")



def _sim_confirmatory_paths(n_paths: int = 150):
    """Regenerate the confirmatory spot paths with the engine's own QE simulator.

    Same contract, same perturbation, same global seed as the confirmatory cell, so
    the bundle drawn in the coverage panel is the trajectory the reported hedges
    actually walk rather than an illustration of one.
    """
    import Hedging_backtest as hb

    cfg = hb.resolve_config("heston_benchmark_v6.yaml", "hedging_config.yaml")
    bm, eng = cfg["benchmark"], cfg["engine"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    base = hb.SimParams.from_regime(bm["regimes"]["baseline"], r, q)
    p = hb.perturb_params(base, "combined", 1.0,
                          eng["misspecification"]["directions"])
    reb, inst = bm["hedging_simulation"]["rebalancing"], bm["hedging_simulation"]["instrument"]
    times, S, _v = hb.simulate_heston_qe(
        p, float(inst["S0"]), float(bm["hedging_simulation"]["horizon"]["T_prime"]),
        int(reb["n_steps"]), n_paths, int(bm["meta"]["global_seed"]),
        float(eng["simulation"]["psi_c"]))
    return float(inst["tau0"]) - times, S, float(inst["K"])


def figure_coverage_region(out_path: str) -> str:
    """Where the accuracy metric looks against where the decision consumes it.

    (a) the labelled contract points, the region the hedge occupies, and the
        trajectory bundle that walks out of the labelled region as tau decays;
    (b) the consequence: the two label-free arms swap rank between the region the
        registered metric scores and the region the decision lives in.
    """
    import numpy as np

    lab = np.load("data/frozen/v6-labels-20260812/train_val/train_val_labels.npz")
    S_l, K_l, tau_l = lab["S"], lab["K"], lab["tau"]
    tau_p, S_p, K_h = _sim_confirmatory_paths()
    box_tau, box_m = (0.08, 0.25), (0.65, 1.25)
    at_strike = (K_l >= 95) & (K_l <= 105)          # carries the hedged contract's strike
    in_window = ((tau_l >= box_tau[0]) & (tau_l <= box_tau[1])
                 & (S_l / K_l >= box_m[0]) & (S_l / K_l <= box_m[1]))
    n_window = int(in_window.sum())
    n_in_box = int((at_strike & in_window & (S_l >= 65) & (S_l <= 125)).sum())

    rows = _rows("results/eval_greeks_hedgeslice/full_vs_hedgebox.csv")
    pairs = [("standard_pinn", "residual-only", _OI["vermillion"]),
             ("feedforward", "value-only", _OI["skyblue"])]
    regimes = [("near_feller", "-", "o"), ("strong_neg_corr", "--", "s")]

    with plt.rc_context(_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                         gridspec_kw={"width_ratios": [1.35, 1.0]})

        ax_a.plot(tau_p, (S_p / K_h).T, color=_OI["blue"], lw=0.35, alpha=0.16, zorder=1)
        ax_a.add_patch(plt.Rectangle((box_tau[0], box_m[0]), box_tau[1] - box_tau[0],
                                     box_m[1] - box_m[0], fill=False, ls="--", lw=1.1,
                                     ec=_OI["vermillion"], zorder=4))
        ax_a.scatter(tau_l[~at_strike], (S_l / K_l)[~at_strike], s=13, color=_NEUTRAL,
                     zorder=3, label=f"labelled points ($n={len(S_l)}$)")
        ax_a.scatter(tau_l[at_strike], (S_l / K_l)[at_strike], s=26, facecolors="none",
                     edgecolors=_OI["orange"], linewidths=1.2, zorder=4,
                     label=f"at the hedged strike ($n={int(at_strike.sum())}$)")
        ax_a.plot([], [], color=_OI["blue"], lw=1.2, alpha=0.6, label="decision trajectories")
        ax_a.plot([], [], ls="--", color=_OI["vermillion"], lw=1.1,
                  label=f"region the decision visits\n({n_window} points project in, "
                        f"{n_in_box} at the hedged strike)")
        ax_a.set_xlim(0, 1.0)
        ax_a.set_ylim(0.45, 2.95)
        ax_a.set_xlabel("$\\tau$ (contract coordinate the decision moves along)")
        ax_a.set_ylabel("$S/K$ (state, normalised)")
        ax_a.set_title("(a) where the labels are, where the decision goes")
        ax_a.legend(loc="upper center", fontsize=6.3, ncol=2,
                    borderpad=0.3, columnspacing=1.1, handletextpad=0.5)

        for arm, label, color in pairs:
            anchor = None
            for regime, ls, mk in regimes:
                rec = next(r for r in rows if r["arm"] == arm and r["regime"] == regime
                           and r["greek"] == "delta")
                ax_b.plot([0, 1], [float(rec["rmse_full"]), float(rec["rmse_hedge"])],
                          color=color, ls=ls, marker=mk, markersize=4.0, lw=1.5, zorder=3)
                if regime == "near_feller":
                    anchor = float(rec["rmse_hedge"])
            ax_b.annotate(label, xy=(1, anchor), xytext=(6, 0),
                          textcoords="offset points", va="center", fontsize=7, color=color)
        ax_b.set_yscale("log")
        ax_b.set_ylim(0.019, 0.115)
        ax_b.set_xlim(-0.15, 1.75)
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels(["full grid\n(metric as reported)", "decision region\n(metric restricted)"])
        ax_b.set_ylabel("held-out $\\partial u/\\partial S$ RMSE")
        ax_b.set_title("(b) the two label-free arms swap rank")
        ax_b.plot([], [], color=_NEUTRAL, ls="-", marker="o", markersize=4,
                  label="held-out (interpolated)")
        ax_b.plot([], [], color=_NEUTRAL, ls="--", marker="s", markersize=4,
                  label="held-out (extrapolated)")
        ax_b.legend(loc="lower left", fontsize=6.6)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_label_controls(out_path: str) -> str:
    """Three controls that resemble derivative supervision without its information.

    Bars are held-out second-derivative relative RMSE at the interpolation regime,
    with the extrapolation regime overlaid as a marker; the vertical rule is the
    unsupervised baseline, so "worse than adding nothing" is read off directly.
    """
    acc = {}
    for path in ("results/eval_greeks_full/ood_param_greeks_agg.csv",
                 "results/eval_greeks_infomatch/ood_param_greeks_agg.csv"):
        for r in _rows(path):
            if r["greek"] == "gamma":
                acc.setdefault(r["arm"], {})[r["regime"]] = float(r["rel_rmse_mean"])

    base = acc["standard_pinn"]["near_feller"]
    entries = [
        ("gradient_penalty_only", "curvature penalty, no labels", _OI["vermillion"]),
        ("info_matched_baseline", "more value data, no labels", _OI["orange"]),
        ("shuffled", "shuffled derivative labels", _OI["green"]),
        ("rung1", "supervised $\\partial u/\\partial S$ only", _OI["purple"]),
        ("rung3", "supervised derivatives", _OI["blue"]),
    ]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.4, 2.5))
        ys = range(len(entries))
        ax.barh(list(ys), [acc[a]["near_feller"] for a, _, _ in entries],
                color=[c for _, _, c in entries], height=0.62, zorder=3)
        ax.scatter([acc[a]["strong_neg_corr"] for a, _, _ in entries], list(ys),
                   marker="|", s=90, color="k", zorder=4, label="second held-out regime")
        ax.axvline(base, color=_NEUTRAL, ls="--", lw=1.1, zorder=2)
        ax.set_ylim(-0.85, len(entries) + 0.90)
        ax.text(base + 0.02, len(entries) + 0.78, f"no derivative\nsupervision ({base:.2f})",
                fontsize=6.8, color=_NEUTRAL, va="top")
        ax.set_yticks(list(ys))
        ax.set_yticklabels([lab for _, lab, _ in entries], fontsize=7.2)
        ax.set_xlabel("held-out $\\partial^2 u/\\partial S^2$ relative RMSE (lower is better)")
        ax.set_xlim(0, 1.12)
        ax.legend(loc="center right", fontsize=6.8, bbox_to_anchor=(1.0, 0.52))
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_problem(out_path: str) -> str:
    """The motivating exhibit: matched values, mismatched curvature, held out.

    One slice through the ``near_feller`` regime, which is excised from training and
    from validation. Panel (a) is the value the surrogates are trained to match;
    panel (b) is the second derivative automatic differentiation returns for free at
    the same points. Unlike every other figure in this module this one evaluates the
    frozen checkpoints rather than reading a scored CSV, because no artifact stores a
    field slice; it re-scores nothing and enters no verdict.
    """
    import numpy as np

    import Hedging_backtest as hb
    from pinn_provider import build_providers

    cfg = hb.resolve_config("heston_benchmark_v6.yaml", "hedging_config.yaml")
    bm = cfg["benchmark"]
    seed = int(bm["meta"]["global_seed"])
    arms = ["standard_pinn", "feedforward", "rung3"]
    provs = build_providers(bm, "results/grid", arms, seed,
                            bm["grid"]["r"], bm["grid"]["q"], include_oracle=True)

    reg, tau, K = bm["regimes"]["near_feller"], 0.25, 100.0
    S = np.linspace(70.0, 130.0, 241)
    out = {k: p.evaluate(S, np.full_like(S, float(reg["v0"])), tau, K)
           for k, p in provs.items()}

    series = [
        # the reference is drawn wide and underneath so the supervised arm, which
        # lies on top of it, reads as tracking rather than as hiding it.
        ("oracle", "reference", _OI["black"], "-", 2.6),
        ("rung3", "derivative-supervised", _OI["blue"], "--", 1.3),
        ("standard_pinn", "residual-only", _OI["vermillion"], "-.", 1.5),
        ("feedforward", "value-only", _OI["orange"], ":", 1.8),
    ]
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
        for key, lab, col, ls, lw in series:
            axes[0].plot(S, out[key]["price"], color=col, ls=ls, lw=lw, label=lab)
            axes[1].plot(S, out[key]["gamma"], color=col, ls=ls, lw=lw)
        axes[0].set_ylabel("$u$")
        axes[0].set_title("(a) the trained quantity", loc="left")
        axes[0].legend(loc="upper left", fontsize=6.6, handlelength=1.9,
                       borderpad=0.2, labelspacing=0.32)
        axes[1].axhline(0.0, color=_NEUTRAL, lw=0.7, zorder=1)
        axes[1].set_ylabel("$\\partial^{2}u/\\partial S^{2}$")
        axes[1].set_title("(b) its curvature, by autodiff", loc="left")
        for ax in axes:
            ax.set_xlabel("$S$")
            ax.set_xlim(70, 130)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def figure_pipeline(out_path: str) -> str:
    """Schematic of the study: where the labels come from and what consumes them.

    Drawn rather than measured, so it reads no artifact. Rendered here instead of in
    TikZ so that the shipped figure is identical under any TeX installation and so
    that it regenerates from the same entry point as every other figure.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    chain = [
        ("coefficient sampler\nevaluation anchors\nexcised", _OI["black"]),
        ("reference solver\nfour legs, cross-\nvalidated to $10^{-3}$", _OI["black"]),
        ("labels\n$u,\\partial_S u,\\partial_{SS}u,\\partial_v u$\ndisagreements masked", _OI["black"]),
        ("surrogate $u_\\phi$\narm $=$ which loss\nterms are active", _OI["blue"]),
    ]
    fork = [
        ("held-out derivative\naccuracy, regimes\nexcised from training", _OI["green"]),
        ("sequential decision\nunder misspecification\n$\\mathrm{CVaR}_{95}$", _OI["vermillion"]),
    ]
    w, gap = 0.176, 0.027

    def _box(ax, cx, cy, h, text, col, bw=None):
        bw = w if bw is None else bw
        ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - h / 2), bw, h,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=1.0, edgecolor=col, facecolor="none",
                                    zorder=3))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=6.2, zorder=4)

    def _arrow(ax, x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, linewidth=0.9,
                                     color=_NEUTRAL, shrinkA=0, shrinkB=0, zorder=2))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 1.55))
        xs = [w / 2 + i * (w + gap) for i in range(4)]
        for x, (t, c) in zip(xs, chain):
            _box(ax, x, 0.5, 0.46, t, c)
        for i in range(3):
            _arrow(ax, xs[i] + w / 2, 0.5, xs[i + 1] - w / 2, 0.5)

        xf = xs[3] + w / 2 + gap + 0.196 / 2
        for cy, (t, c) in zip((0.775, 0.225), fork):
            _box(ax, xf, cy, 0.40, t, c, bw=0.196)
        xj = xs[3] + w / 2 + gap * 0.45
        _arrow(ax, xs[3] + w / 2, 0.5, xj, 0.5)
        ax.plot([xj, xj], [0.225, 0.775], color=_NEUTRAL, lw=0.9, zorder=2)
        for cy in (0.775, 0.225):
            _arrow(ax, xj, cy, xf - 0.196 / 2, cy)

        # the rounded box outlines overhang their nominal extent, so the limits
        # carry a margin: without it the outer boxes are clipped at the frame.
        ax.set_xlim(-0.018, xf + 0.196 / 2 + 0.018)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
