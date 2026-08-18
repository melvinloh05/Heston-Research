"""R10 — census the oracle mask on the PRODUCTION evaluation-anchor grids.

Runs make_datasets.generate_anchor_grids UNCHANGED at production settings
(contract grid 41 x 33 x 16, seed 42, mc_subset_frac 0.10, mc_paths 200_000,
near_feller_mc_multiplier 4) into a scratch directory, then reports, per named
regime and per Greek, the mask rate over the FULL grid — a census, not a sample.

Usage: python audit/repro/r10_anchor_grid_masks.py <scratch_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from make_datasets import generate_anchor_grids
from make_labels import LABEL_QUANTITIES

CONTRACT = "heston_benchmark_v6.yaml"


def main(out_dir: str) -> None:
    contract = yaml.safe_load(open(CONTRACT))
    t0 = time.perf_counter()
    res = generate_anchor_grids(CONTRACT, out_dir, seed=42, mc_subset_frac=0.10,
                                mc_paths=200_000, near_feller_mc_multiplier=4)
    print(f"\ngenerate_anchor_grids wall clock: {time.perf_counter() - t0:.1f}s")

    primary = list(contract["splits"]["heldout_greek_and_hedging"])
    print(f"\nPRIMARY regimes (splits.heldout_greek_and_hedging): {primary}")
    print("legs on the FULL grid = cf + fd (+ adi where the 4th leg applies);")
    print("the MC leg enters only on the stratified 10% subset rows.\n")

    hdr = (f"{'regime':>24s} {'feller':>7s} {'adi':>4s} {'n_grid':>7s} "
           f"{'mask_any':>9s} " + " ".join(f"{g:>8s}" for g in LABEL_QUANTITIES))
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name in contract["regimes"]:
        d = np.load(res["regime_npz"][name])
        ma = np.asarray(d["mask_any"], bool)
        per = {g: float(np.asarray(d[f"mask_{g}"], bool).mean()) for g in LABEL_QUANTITIES}
        rows[name] = {"feller": float(d["feller_ratio"]), "adi": bool(d["adi_leg"]),
                      "n": int(ma.size), "mask_any": float(ma.mean()), **per}
        print(f"{name:>24s} {float(d['feller_ratio']):7.2f} "
              f"{'yes' if bool(d['adi_leg']) else 'no':>4s} {ma.size:7d} "
              f"{ma.mean():9.4f} " + " ".join(f"{per[g]:8.4f}" for g in LABEL_QUANTITIES))

    print("\nPRIMARY-regime mask-rate contrast (the cross-regime comparability question):")
    a, b = primary
    print(f"  {a}: mask_any {rows[a]['mask_any']:.4f} over {rows[a]['n']} points "
          f"({rows[a]['n'] - round(rows[a]['mask_any'] * rows[a]['n'])} survive)")
    print(f"  {b}: mask_any {rows[b]['mask_any']:.4f} over {rows[b]['n']} points "
          f"({rows[b]['n'] - round(rows[b]['mask_any'] * rows[b]['n'])} survive)")
    print(f"  ratio {rows[a]['mask_any'] / max(rows[b]['mask_any'], 1e-12):.1f}x")
    print(f"\nartifacts: {res['out_dir']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch_anchors")
