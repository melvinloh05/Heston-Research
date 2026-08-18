"""R15 — network-free difficulty proxies for masked vs kept anchor-grid points.

No arm is trained yet, so "the mask removes the points the network finds hard"
cannot be measured directly. Two proxies that need no network:

  (a) oracle uncertainty (max pairwise leg disagreement, folded with MC SE) — the
      pipeline's OWN difficulty measure, and the denominator of
      improvement_to_oracle_noise;
  (b) local roughness of the true surface: |d2(consensus_gamma)/dS2| by second
      difference along the S axis, normalised by the grid's rms gamma. A rougher
      neighbourhood is harder for any smooth approximator.

Also reports the share of total sum(consensus_g^2) carried by masked points, i.e.
how much of the regime's Greek "mass" the eval set never sees.

Usage: python audit/repro/r15_masked_vs_kept_difficulty.py <anchors_dir>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONTRACT = "heston_benchmark_v6.yaml"
GREEKS = ("price", "delta", "gamma", "vega", "vanna")


def main(anchors_dir: str) -> None:
    contract = yaml.safe_load(open(CONTRACT))
    primary = list(contract["splits"]["heldout_greek_and_hedging"])
    for name in contract["splits"]["eval_anchors_heston"]:
        d = np.load(str(Path(anchors_dir) / f"{name}_grid.npz"))
        m = np.asarray(d["mask_any"], bool)
        mf, keep = m.ravel(), ~m.ravel()
        tag = "PRIMARY" if name in primary else "sanity "
        print(f"\n{tag} {name}  mask_any={mf.mean():.4f}")
        if mf.sum() == 0:
            print("  nothing masked")
            continue

        print(f"  {'greek':>7s} {'unc rms masked':>15s} {'unc rms kept':>13s} "
              f"{'ratio':>7s} {'masked share of sum(c^2)':>25s}")
        for g in GREEKS:
            u = np.abs(np.asarray(d[f"uncertainty_{g}"], float).ravel())
            c2 = np.asarray(d[f"consensus_{g}"], float).ravel() ** 2
            um = float(np.sqrt(np.nanmean(u[mf] ** 2)))
            uk = float(np.sqrt(np.nanmean(u[keep] ** 2)))
            share = float(c2[mf].sum() / c2.sum()) if c2.sum() > 0 else float("nan")
            print(f"  {g:>7s} {um:15.4g} {uk:13.4g} {um / uk if uk > 0 else np.nan:7.1f} "
                  f"{share:25.4f}")

        # (b) local roughness of gamma along S, second difference, C-order (nS,nK,nT)
        gam = np.asarray(d["consensus_gamma"], float)
        rough = np.zeros_like(gam)
        rough[1:-1] = np.abs(gam[2:] - 2 * gam[1:-1] + gam[:-2])
        interior = np.zeros(gam.shape, bool)
        interior[1:-1] = True
        rms_g = float(np.sqrt(np.mean(gam ** 2)))
        rm = rough.ravel()[mf & interior.ravel()]
        rk = rough.ravel()[keep & interior.ravel()]
        print(f"  gamma roughness |d2gamma/dS2| (interior points, /rms gamma "
              f"{rms_g:.4g}): masked median {np.median(rm) / rms_g:.4f} "
              f"(n={rm.size}), kept median {np.median(rk) / rms_g:.4f} (n={rk.size}), "
              f"ratio {np.median(rm) / max(np.median(rk), 1e-300):.2f}x")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
