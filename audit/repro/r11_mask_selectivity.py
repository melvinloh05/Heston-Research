"""R11 — is the oracle mask selective on difficulty, on the anchor grids?

Reads the {regime}_grid.npz artifacts written by R10 (production
generate_anchor_grids) and compares MASKED vs SURVIVING points on:
  - moneyness S/K, tau                      (grid geometry)
  - |consensus_g| for each Greek            (state magnitude)
  - rms(consensus_g) kept vs all            (the rel_rmse denominator)
plus mask rate by moneyness bin and by tau bin.

Caveat stated rather than hidden: on a MASKED point the consensus is the median
of legs that disagree, so it is a noisier magnitude proxy than on a surviving
point. It is still the best available estimate of the state's Greek magnitude,
and the comparisons below are ratios of medians/rms, not point claims.

Feller ratio is CONSTANT within an anchor grid (one parameter vector per
regime), so the Feller-vs-mask question is cross-regime here; it is answered
within-sample on the hypercube by R13.

Usage: python audit/repro/r11_mask_selectivity.py <anchors_dir>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from make_labels import LABEL_QUANTITIES

CONTRACT = "heston_benchmark_v6.yaml"
GREEKS = list(LABEL_QUANTITIES)


def _q(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return (float("nan"),) * 3
    return tuple(float(x) for x in np.percentile(a, [25, 50, 75]))


def main(anchors_dir: str) -> None:
    contract = yaml.safe_load(open(CONTRACT))
    primary = list(contract["splits"]["heldout_greek_and_hedging"])
    for name in contract["regimes"]:
        d = np.load(str(Path(anchors_dir) / f"{name}_grid.npz"))
        S_ax, K_ax, T_ax = (np.asarray(d[f"{a}_axis"], float) for a in ("S", "K", "tau"))
        Sg, Kg, Tg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")
        Sf, Kf, Tf = Sg.ravel(), Kg.ravel(), Tg.ravel()
        mn = Sf / Kf
        m = np.asarray(d["mask_any"], bool).ravel()
        keep = ~m
        tag = "PRIMARY" if name in primary else "sanity "
        print(f"\n{'='*78}\n{tag}  {name}   feller={float(d['feller_ratio']):.2f}  "
              f"adi_leg={bool(d['adi_leg'])}  mask_any={m.mean():.4f}  "
              f"({int(m.sum())} masked / {m.size} points)")
        if m.sum() == 0:
            print("  nothing masked — no selectivity to measure")
            continue

        print(f"  {'quantity':>22s} {'masked q25/q50/q75':>30s} {'surviving q25/q50/q75':>30s}")
        for lbl, v in (("moneyness S/K", mn), ("tau", Tf)):
            a, b = _q(v[m]), _q(v[keep])
            print(f"  {lbl:>22s} {a[0]:9.4f}{a[1]:10.4f}{a[2]:10.4f}   "
                  f"{b[0]:9.4f}{b[1]:10.4f}{b[2]:10.4f}")
        for g in GREEKS:
            c = np.abs(np.asarray(d[f"consensus_{g}"], float).ravel())
            a, b = _q(c[m]), _q(c[keep])
            print(f"  {'|consensus_' + g + '|':>22s} {a[0]:9.4g}{a[1]:10.4g}{a[2]:10.4g}   "
                  f"{b[0]:9.4g}{b[1]:10.4g}{b[2]:10.4g}")

        print(f"  {'greek':>10s} {'med|c| masked/surv':>20s} {'rms(c) kept':>13s} "
              f"{'rms(c) all':>12s} {'kept/all':>9s} {'mask rate':>10s}")
        for g in GREEKS:
            c = np.abs(np.asarray(d[f"consensus_{g}"], float).ravel())
            r_med = (np.median(c[m]) / np.median(c[keep])
                     if np.median(c[keep]) > 0 else float("nan"))
            rms_keep = float(np.sqrt(np.mean(c[keep] ** 2))) if keep.any() else float("nan")
            rms_all = float(np.sqrt(np.mean(c ** 2)))
            mg = np.asarray(d[f"mask_{g}"], bool).ravel()
            print(f"  {g:>10s} {r_med:20.3f} {rms_keep:13.4g} {rms_all:12.4g} "
                  f"{rms_keep / rms_all:9.4f} {mg.mean():10.4f}")

        # mask rate by moneyness / tau bin
        edges = np.array([0.0, 0.75, 0.9, 1.0, 1.1, 1.30, 9.9])
        print("  mask rate by moneyness bin:", "  ".join(
            f"[{edges[i]:.2f},{edges[i+1]:.2f}): "
            f"{m[(mn >= edges[i]) & (mn < edges[i+1])].mean():.3f}"
            f"(n={int(((mn >= edges[i]) & (mn < edges[i+1])).sum())})"
            for i in range(len(edges) - 1)
            if ((mn >= edges[i]) & (mn < edges[i + 1])).any()))
        print("  mask rate by tau slice:", "  ".join(
            f"{t:.2f}:{m[Tf == t].mean():.3f}" for t in T_ax))
        wing = (mn < 0.75) | (mn > 1.30)
        print(f"  wing (S/K outside [0.75,1.30]) mask {m[wing].mean():.4f} "
              f"(n={int(wing.sum())}) vs body {m[~wing].mean():.4f} (n={int((~wing).sum())})")

        # mirror of make_labels.mask_neutrality_report check (c), which is run on the
        # TRAIN/VAL artifact only and never on these grids
        for g in ("gamma", "vega"):
            ag = np.abs(np.asarray(d[f"consensus_{g}"], float).ravel())
            order = np.argsort(ag)
            print(f"  mask rate by |consensus_{g}| decile (1=smallest): " + " ".join(
                f"{k+1}:{m[ch].mean():.3f}" for k, ch in enumerate(np.array_split(order, 10))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
