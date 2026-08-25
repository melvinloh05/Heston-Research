"""R3 -- is the retained label set EASIER than the masked set?

If oracle disagreement masking removes the hardest points, every accuracy claim is
scored on a set selected for being tractable, and the claim must be qualified. If it
does not, an obvious attack closes for free.

The frozen mask_neutrality_report already covers Feller bands, parameter marginals,
and |gamma| deciles. This adds the two axes it does not -- maturity and moneyness --
and states a verdict per axis rather than leaving the reader to infer one.

"Harder" is defined per axis in the direction where the surrogate is known to struggle:
curvature HIGH, maturity SHORT (payoff kink least smoothed), moneyness NEAR ATM,
Feller ratio LOW, |rho| HIGH. A mask that concentrates on the hard end biases the
retained set easy; one that concentrates on the easy end does not.
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np
from scipy.stats import ks_2samp

PARTS = "data/frozen/v6-labels-20260812/**/labels.npz"
# axis -> (harder direction, label)
AXES = {"abs_gamma": ("high", "curvature |Gamma|"),
        "tau": ("low", "maturity tau"),
        "moneyness": ("mid", "moneyness S/K (hard near 1)"),
        "feller_ratio": ("low", "Feller ratio"),
        "abs_rho": ("high", "|rho|")}


def load() -> dict[str, np.ndarray]:
    cols: dict[str, list] = {k: [] for k in
                             ("masked", "abs_gamma", "tau", "moneyness",
                              "feller_ratio", "abs_rho")}
    for f in sorted(glob.glob(PARTS, recursive=True)):
        z = np.load(f, allow_pickle=True)
        if "mask_any" not in z.files:
            continue
        m = np.asarray(z["mask_any"], bool)                 # (n_param, n_grid)
        npar, ngrid = m.shape
        S, K, tau = (np.asarray(z[a], float) for a in ("S", "K", "tau"))
        fr = np.asarray(z["feller_ratio"], float)
        rho = np.asarray(z["params"], float)[:, list(z["param_names"]).index("rho")]
        g = np.abs(np.asarray(z["consensus_gamma"], float))
        cols["masked"].append(m.ravel())
        cols["abs_gamma"].append(g.ravel())
        cols["tau"].append(np.tile(tau, (npar, 1)).ravel())
        cols["moneyness"].append(np.tile(S / K, (npar, 1)).ravel())
        cols["feller_ratio"].append(np.repeat(fr, ngrid))
        cols["abs_rho"].append(np.repeat(np.abs(rho), ngrid))
    return {k: np.concatenate(v) for k, v in cols.items()}


def main() -> None:
    d = load()
    m = d["masked"]
    print(f"{m.size} label cells, {m.sum()} masked ({m.mean():.4f})\n")
    rows = []
    print(f"{'axis':<30}{'masked mean':>13}{'retained':>11}{'KS':>8}{'p':>10}  verdict")
    for ax, (hard, label) in AXES.items():
        x = d[ax]
        a, b = x[m], x[~m]
        ks = ks_2samp(a, b)
        # Which end does the mask sit on, relative to the retained set?
        if hard == "high":
            biased = a.mean() > b.mean()
        elif hard == "low":
            biased = a.mean() < b.mean()
        else:                                      # "mid": hard near 1.0
            biased = abs(a.mean() - 1.0) < abs(b.mean() - 1.0)
        verdict = ("MASK ON HARD END -> retained set is easier; qualify accuracy claims"
                   if biased else "mask on easy end -> retained set not selected easy")
        print(f"{label:<30}{a.mean():>13.5f}{b.mean():>11.5f}"
              f"{ks.statistic:>8.3f}{ks.pvalue:>10.2e}  {verdict}")
        rows.append({"axis": ax, "label": label, "harder_direction": hard,
                     "masked_mean": a.mean(), "retained_mean": b.mean(),
                     "masked_median": float(np.median(a)),
                     "retained_median": float(np.median(b)),
                     "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                     "retained_set_easier": bool(biased)})

    # Mask rate by quintile of each axis -- shows shape, not just means.
    qrows = []
    print()
    for ax, (_, label) in AXES.items():
        x = d[ax]
        edges = np.quantile(x, np.linspace(0, 1, 6))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        idx = np.clip(np.searchsorted(edges, x, "right") - 1, 0, 4)
        rates = [float(m[idx == q].mean()) if (idx == q).any() else float("nan")
                 for q in range(5)]
        print(f"{label:<30}mask rate by quintile: "
              + "  ".join(f"{r:.3f}" for r in rates))
        qrows.append({"axis": ax, **{f"q{q+1}": rates[q] for q in range(5)},
                      **{f"q{q+1}_hi": float(edges[q + 1]) for q in range(5)}})

    os.makedirs("results/reanalysis", exist_ok=True)
    with open("results/reanalysis/mask_difficulty.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open("results/reanalysis/mask_difficulty_quintiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0])); w.writeheader(); w.writerows(qrows)
    print("\nwrote results/reanalysis/mask_difficulty{,_quintiles}.csv")


if __name__ == "__main__":
    main()
