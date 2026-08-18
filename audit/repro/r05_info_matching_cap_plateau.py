"""R05 — run_info_matching: a plateau produced by the ROW CAP, not by information.

subsample_train caps the budget at the frozen train-split row count. When
m*N exceeds that count, consecutive rungs train on BIT-IDENTICAL data, the
Gamma-RMSE curve is exactly flat by construction, and plateau_multiplier
reports plateau_reached=True — the same verdict an information plateau gives.
Nothing in the sweep asserts or warns that the cap bound.
"""
import copy
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
from run_info_matching import PLATEAU_TOL, plateau_multiplier, subsample_train  # noqa: E402


class FakeDS:
    """Minimal ArmDataset stand-in: subsample_train only needs .data/.n_rows."""

    def __init__(self, n_rows):
        self.n_rows = n_rows
        g = torch.arange(n_rows, dtype=torch.float32)
        self.data = {"x": g.unsqueeze(1).repeat(1, 8), "gamma_ref": g}


N = 4096
MULTS = [1, 2, 3, 4, 5]

print("=" * 74)
print("(1) budget ladder against a train split SMALLER than 5N")
print("=" * 74)
for n_rows in (5 * N, 3 * N + 100, 2 * N + 7):
    ds = FakeDS(n_rows)
    subs = {m: subsample_train(ds, m * N, seed=42) for m in MULTS}
    counts = {m: s.n_rows for m, s in subs.items()}
    print(f"\n  frozen train rows = {n_rows}   (N = {N}, cap 5N = {5*N})")
    print(f"  rung -> rows actually trained on: {counts}")
    ident = [m for m in MULTS[1:]
             if torch.equal(subs[m].data["x"], subs[m - 1].data["x"])]
    print(f"  rungs whose data is BIT-IDENTICAL to the rung below: {ident or 'none'}")

print()
print("=" * 74)
print("(2) what plateau_multiplier makes of a cap-flattened curve")
print("=" * 74)
# A curve still improving at every genuine rung, then pinned flat by the cap.
# rows: m=1,2,3 grow; m=4,5 are capped copies of m=3 -> identical RMSE.
genuine = [0.400, 0.300, 0.240]           # ~25%, ~20% real gains, still improving
capped = genuine + [genuine[-1], genuine[-1]]
print(f"  mean Gamma rel-RMSE by rung : {capped}")
print(f"  PLATEAU_TOL                 : {PLATEAU_TOL}")
res = plateau_multiplier(MULTS, capped, tol=PLATEAU_TOL)
print(f"  plateau_multiplier ->  {res}")
print(f"\n  verdict: plateau_reached={res['plateau_reached']} at m={res['plateau_multiplier']}")
print("  ...but the m=3 -> m=4 'gain' of 0.0 is an identity, not a measurement:")
print("  the two rungs saw the same rows.")

print()
print("  For contrast, a genuine information plateau at the same rung:")
genuine_plateau = [0.400, 0.300, 0.240, 0.238, 0.237]
print(f"  curve {genuine_plateau} -> {plateau_multiplier(MULTS, genuine_plateau, tol=PLATEAU_TOL)}")
print("\n  The two verdicts are indistinguishable in the returned dict and in the")
print("  CSV's plateau columns; only the n_train_rows column separates them.")
