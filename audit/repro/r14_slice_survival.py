"""R14 — how many points actually survive the mask on each scored slice.

eval_greeks scores the full grid for the PRIMARY regimes and full + wing + tau
holdout for the SANITY regimes. This enumerates, per regime x slice, the number
of points eval_arm_on_regime would actually score, using the module's OWN
_slice_masks and the same `keep & restrict` rule. A slice with 0 survivors makes
every metric on it NaN.

Usage: python audit/repro/r14_slice_survival.py <anchors_dir>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import eval_greeks as EG

CONTRACT = "heston_benchmark_v6.yaml"


def main(anchors_dir: str) -> None:
    contract = yaml.safe_load(open(CONTRACT))
    primary = list(contract["splits"]["heldout_greek_and_hedging"])
    order = list(contract["splits"]["eval_anchors_heston"])
    print(f"{'regime':>24s} {'role':>8s} {'slice':>5s} {'n_slice':>8s} "
          f"{'survive':>8s} {'frac':>7s}")
    print("-" * 66)
    for name in order:
        npz = str(Path(anchors_dir) / f"{name}_grid.npz")
        d = np.load(npz)
        keep = ~np.asarray(d["mask_any"], bool).ravel()
        role = "PRIMARY" if name in primary else "sanity"
        # eval_greeks scores: primary -> full only; sanity -> full + wing + tau
        slices = {"full": None} if name in primary else {"full": None, **EG._slice_masks(npz)}
        for slname, restrict in slices.items():
            r = (np.asarray(restrict, bool).ravel() if restrict is not None
                 else np.ones(keep.size, bool))
            n_sl, surv = int(r.sum()), int((keep & r).sum())
            print(f"{name:>24s} {role:>8s} {slname:>5s} {n_sl:8d} {surv:8d} "
                  f"{surv / n_sl:7.4f}" + ("   <-- EMPTY: every metric NaN" if surv == 0 else ""))

    print("\nPer-tau-slice survival on feller_violating_volvol "
          "(the regime the contract calls the 'Vega worst-case long-tau slice'):")
    d = np.load(str(Path(anchors_dir) / "feller_violating_volvol_grid.npz"))
    T_ax = np.asarray(d["tau_axis"], float)
    keep = ~np.asarray(d["mask_any"], bool)
    for j, t in enumerate(T_ax):
        k = int(keep[:, :, j].sum())
        print(f"  tau={t:.2f}: {k:5d}/{keep[:, :, j].size} survive"
              + ("   <-- none" if k == 0 else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
