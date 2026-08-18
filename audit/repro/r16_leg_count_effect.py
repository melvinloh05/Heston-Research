"""R16 — within one regime, does the mask rate track LEG COUNT rather than difficulty?

generate_anchor_grids runs CF + FD on every grid point and folds the MC leg in on
a stratified 10% row subset only (module docstring, MC_STRATIFICATION). So inside
a single regime, at identical parameters, two populations exist:

  mc_mask == True   -> cf + fd (+ adi) + mc      (one more leg)
  mc_mask == False  -> cf + fd (+ adi)

Same regime, same parameter vector, same grid geometry — the only systematic
difference is how many legs got a vote. Comparing their mask rates isolates the
leg-count contribution from any difficulty story. (The MC subset is stratified per
tau slice and drawn without replacement, so it spans the grid rather than
clustering.)

Usage: python audit/repro/r16_leg_count_effect.py <anchors_dir>
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
    print(f"{'regime':>24s} {'adi':>4s} {'n_mc':>6s} {'mask|mc':>8s} "
          f"{'n_nomc':>7s} {'mask|no mc':>10s} {'ratio':>7s}")
    print("-" * 72)
    for name in contract["splits"]["eval_anchors_heston"]:
        d = np.load(str(Path(anchors_dir) / f"{name}_grid.npz"))
        m = np.asarray(d["mask_any"], bool).ravel()
        mc = np.asarray(d["mc_mask"], bool).ravel()
        a, b = m[mc].mean(), m[~mc].mean()
        print(f"{name:>24s} {'yes' if bool(d['adi_leg']) else 'no':>4s} "
              f"{int(mc.sum()):6d} {a:8.4f} {int((~mc).sum()):7d} {b:10.4f} "
              f"{a / b if b > 0 else float('inf'):7.1f}"
              + ("   PRIMARY" if name in primary else ""))

    print("\nPer-greek, on the two PRIMARY regimes:")
    for name in primary:
        d = np.load(str(Path(anchors_dir) / f"{name}_grid.npz"))
        mc = np.asarray(d["mc_mask"], bool).ravel()
        print(f"  {name} (adi_leg={bool(d['adi_leg'])}):")
        for g in GREEKS:
            mg = np.asarray(d[f"mask_{g}"], bool).ravel()
            print(f"    {g:>7s}: mask|mc {mg[mc].mean():.4f}   "
                  f"mask|no-mc {mg[~mc].mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
