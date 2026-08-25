"""R1 -- supervised-arm loss decomposition across the residual-weight sweep.

Question this settles: rung3's hedging CVaR is flat across a 100x range of
lambda_pde. Is that because the residual term is a NEGLIGIBLE share of rung3's
objective (a CONDITIONAL robustness claim -- "supervision swamps the residual, so of
course moving its weight does nothing"), or because the residual term carries real
weight and the outcome is insensitive anyway (a STRONG claim)?

The distinction is not cosmetic. Under the conditional reading the protective effect
is arithmetic; under the strong reading it is a property of the trained field. The
paper must not assert the strong version if the data support only the conditional one.

Reads results/{grid,grid_lampde_*,grid_robustness}/<arm>/s<seed>/loss_curves.csv,
which logs each term BEFORE its lambda multiplier. Shares are lambda-weighted.
"""
from __future__ import annotations

import csv
import glob
import json
import os

# lambda_delta/gamma/vega are 1.0 in lambdas_selected.yaml; lambda_pde varies by grid.
_GRIDS = {
    "results/grid_robustness": 0.0,
    "results/grid_lampde_1e-4": 1e-4,
    "results/grid_lampde_1e-3": 1e-3,
    "results/grid_lampde_3e-3": 3e-3,
    "results/grid": 0.01,
}
_LAM = {"price": 1.0, "delta": 1.0, "gamma": 1.0, "vega": 1.0, "bc": 1.0, "vanna": 1.0}
_TERMS = ("price", "pde", "delta", "gamma", "vega", "bc", "vanna", "gamma_penalty")


def final_train_row(path: str) -> dict | None:
    """Last TRAIN row: the composition the optimizer saw at convergence."""
    rows = [r for r in csv.DictReader(open(path)) if r.get("split") == "train"]
    return rows[-1] if rows else None


def decompose(row: dict, lam_pde: float) -> dict:
    w = {}
    for t in _TERMS:
        v = row.get(t)
        if v in (None, ""):
            continue
        lam = lam_pde if t == "pde" else _LAM.get(t, 1.0)
        w[t] = lam * float(v)
    tot = sum(w.values())
    return {"weighted": w, "sum_weighted": tot,
            "logged_total": float(row["total"]),
            "share": {t: (v / tot if tot > 0 else float("nan")) for t, v in w.items()}}


def main() -> None:
    out = []
    for grid, lam_pde in _GRIDS.items():
        for arm_dir in sorted(glob.glob(os.path.join(grid, "*"))):
            arm = os.path.basename(arm_dir)
            if arm.startswith("_") or not os.path.isdir(arm_dir):
                continue
            for seed_dir in sorted(glob.glob(os.path.join(arm_dir, "s*"))):
                lc = os.path.join(seed_dir, "loss_curves.csv")
                if not os.path.exists(lc):
                    continue
                row = final_train_row(lc)
                if row is None:
                    continue
                d = decompose(row, lam_pde)
                rec = {"grid": grid, "arm": arm, "lambda_pde": lam_pde,
                       "seed": int(os.path.basename(seed_dir)[1:]),
                       "step": int(row["step"]),
                       "logged_total": d["logged_total"],
                       "sum_weighted": d["sum_weighted"],
                       "pde_share": d["share"].get("pde", 0.0),
                       "sup_share": sum(d["share"].get(t, 0.0)
                                        for t in ("delta", "gamma", "vega")),
                       "price_share": d["share"].get("price", 0.0)}
                rec.update({f"w_{t}": d["weighted"].get(t, 0.0) for t in _TERMS})
                out.append(rec)

    os.makedirs("results/reanalysis", exist_ok=True)
    cols = list(out[0])
    with open("results/reanalysis/loss_decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(out)

    # Aggregate: mean share by (arm, lambda_pde) over seeds.
    agg: dict[tuple, list] = {}
    for r in out:
        agg.setdefault((r["arm"], r["lambda_pde"]), []).append(r)
    print(f"{'arm':<26}{'lam_pde':>9}{'n':>4}{'PDE share':>11}{'sup share':>11}"
          f"{'price share':>13}{'total':>11}")
    lines = []
    for (arm, lam), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = len(rs)
        m = lambda k: sum(r[k] for r in rs) / n
        print(f"{arm:<26}{lam:>9.0e}{n:>4}{m('pde_share'):>11.4f}"
              f"{m('sup_share'):>11.4f}{m('price_share'):>13.4f}{m('logged_total'):>11.5f}")
        lines.append({"arm": arm, "lambda_pde": lam, "n_seeds": n,
                      "pde_share": m("pde_share"), "sup_share": m("sup_share"),
                      "price_share": m("price_share"),
                      "total_loss": m("logged_total"),
                      "w_pde": m("w_pde"), "w_price": m("w_price")})
    with open("results/reanalysis/loss_decomposition_agg.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lines[0]))
        w.writeheader(); w.writerows(lines)
    print(f"\nwrote results/reanalysis/loss_decomposition{{,_agg}}.csv  ({len(out)} runs)")


if __name__ == "__main__":
    main()
