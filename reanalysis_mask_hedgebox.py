"""R3b -- is the HEDGE BOX disproportionately masked?

R3 found masking concentrates on short maturity. The hedge box is tau in
[tau0 - T', tau0] = [0.08, 0.25], i.e. the short-maturity end. So the axis-level
finding has a direct consequence the axis table cannot show: how much of the region
the accuracy claim is scored on survives masking, and whether any maturity node
inside it is effectively absent.

This matters because the paper's surviving accuracy claim is scored IN THE BOX.
If label support there is thin, the claim needs the caveat stated with it.
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_greeks import hedge_slice_spec   # noqa: E402

PARTS = "data/frozen/v6-labels-20260812/**/labels.npz"


def main() -> None:
    spec = hedge_slice_spec(yaml.safe_load(open("heston_benchmark_v6.yaml")))
    tot_in = msk_in = tot_out = msk_out = 0
    per_tau: dict[float, list[int]] = {}
    for f in sorted(glob.glob(PARTS, recursive=True)):
        z = np.load(f, allow_pickle=True)
        if "mask_any" not in z.files:
            continue
        m = np.asarray(z["mask_any"], bool)
        S, K, tau = (np.asarray(z[a], float) for a in ("S", "K", "tau"))
        npar = m.shape[0]
        mny = S / K
        inbox = ((tau >= spec["tau_lo"]) & (tau <= spec["tau_hi"])
                 & (mny >= spec["moneyness_lo"]) & (mny <= spec["moneyness_hi"]))
        IB = np.tile(inbox, (npar, 1))
        tot_in += int(IB.sum()); msk_in += int(m[IB].sum())
        tot_out += int((~IB).sum()); msk_out += int(m[~IB].sum())
        for t in np.unique(tau):
            sel = np.tile(tau == t, (npar, 1))
            a, b = per_tau.get(round(float(t), 6), (0, 0))
            per_tau[round(float(t), 6)] = (a + int(m[sel].sum()), b + int(sel.sum()))

    rin, rout = msk_in / tot_in, msk_out / tot_out
    print(f"IN  hedge box: {msk_in}/{tot_in} = {rin:.4f}")
    print(f"OUT of box   : {msk_out}/{tot_out} = {rout:.4f}")
    print(f"ratio in/out = {rin / rout:.3f}")

    os.makedirs("results/reanalysis", exist_ok=True)
    with open("results/reanalysis/mask_hedgebox.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["region", "masked", "total", "mask_rate"])
        w.writerow(["hedge_box", msk_in, tot_in, rin])
        w.writerow(["outside_box", msk_out, tot_out, rout])
        w.writerow(["ratio_in_over_out", "", "", rin / rout])

    rows = []
    for t in sorted(per_tau):
        a, b = per_tau[t]
        rows.append({"tau": t, "in_hedge_box": bool(spec["tau_lo"] <= t <= spec["tau_hi"]),
                     "masked": a, "total": b, "mask_rate": a / b})
    with open("results/reanalysis/mask_by_tau.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    worst = sorted((r for r in rows if r["in_hedge_box"]),
                   key=lambda r: -r["mask_rate"])[:5]
    print("\nWorst-masked maturity nodes INSIDE the hedge box:")
    for r in worst:
        print(f"  tau={r['tau']:.4f}  mask rate {r['mask_rate']:.4f}  ({r['masked']}/{r['total']})")
    print("\nwrote results/reanalysis/mask_hedgebox.csv, mask_by_tau.csv")


if __name__ == "__main__":
    main()
