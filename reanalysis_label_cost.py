"""R5 -- measured compute cost per label, by oracle leg.

The information-matching sweep says how many PRICE points it takes to match what
derivative labels give. It says nothing about what a label COSTS. Those two together
are the only honest way to state the practical case for derivative supervision:
labels are worth their price only if the price is small.

So this times all four legs on one common workload and normalises to the CF leg,
which returns price AND every Greek from a single Gauss-Legendre integration with the
derivatives taken analytically on the integrand -- i.e. Greeks are essentially free
once the price is computed. Everything else pays a multiple.

Wall clock on one machine, single process, after a warm-up call. It is a ratio
statement, not a benchmark: the ratios are what transfer, not the seconds.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import yaml

from oracle import (HestonParams, heston_greeks_adi, heston_greeks_cf,
                    heston_greeks_fd, heston_greeks_mc)

N_POINTS = 64          # one parameter point's (S, K, tau) block, as in the frozen parts
REPEATS = 3


def timeit(fn, *a, repeats: int = REPEATS, warmup: bool = True, **kw):
    """Best-of-`repeats`. The PDE and MC legs get one untimed pass at most --
    they are minutes-scale and a warm-up would not change the ratio."""
    if warmup:
        fn(*a, **kw)
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    cfg = yaml.safe_load(open("heston_benchmark_v6.yaml"))
    g = cfg["grid"]
    r, q = float(g["r"]), float(g["q"])
    base = cfg["regimes"]["baseline"]
    p = HestonParams(kappa=float(base["kappa"]), theta=float(base["theta"]),
                     xi=float(base["xi"]), rho=float(base["rho"]), v0=float(base["v0"]))
    rng = np.random.default_rng(0)
    S = rng.uniform(50.0, 150.0, N_POINTS)
    K = np.full(N_POINTS, 100.0)
    tau = rng.uniform(0.04, 1.0, N_POINTS)

    rows = []
    t_cf, _ = timeit(heston_greeks_cf, S, K, tau, p, r, q)
    rows.append({"leg": "A: CF, analytic AD on the integrand", "seconds": t_cf,
                 "quantities": "price + all Greeks",
                 "note": "one Gauss-Legendre integration serves price and every derivative"})

    t_fd, _ = timeit(heston_greeks_fd, S, K, tau, p, r, q)
    rows.append({"leg": "B: FD stencils on COS prices", "seconds": t_fd,
                 "quantities": "price + all Greeks",
                 "note": "4th-order central stencils, 3-step Richardson sweep"})

    t_mc, _ = timeit(heston_greeks_mc, S, K, tau, p, r, q, n_paths=20_000,
                 repeats=1, warmup=False)
    rows.append({"leg": "C: MC pathwise / LR (20k paths, 1/10 production)",
                 "seconds": t_mc, "quantities": "price + Greeks + std errors",
                 "note": "production uses 200k paths; scale x10"})

    adi = cfg.get("oracle", {}).get("fourth_leg", {})
    t_adi, _ = timeit(heston_greeks_adi, S, K, tau, p, r, q,
                      repeats=1, warmup=False,
                      nx=int(adi.get("nx", 901)), nv=int(adi.get("nv", 481)),
                      xmax=float(adi.get("xmax", 4.0)), vmax=float(adi.get("vmax", 1.6)),
                      steps_per_year=int(adi.get("steps_per_year", 1000)))
    rows.append({"leg": "D: Craig-Sneyd ADI PDE solve", "seconds": t_adi,
                 "quantities": "price + all Greeks",
                 "note": "one solve serves all (S, K) by call homogeneity"})

    for row in rows:
        row["rel_to_cf"] = row["seconds"] / t_cf
        row["seconds_per_point"] = row["seconds"] / N_POINTS

    os.makedirs("results/reanalysis", exist_ok=True)
    cols = ["leg", "quantities", "seconds", "seconds_per_point", "rel_to_cf", "note"]
    with open("results/reanalysis/label_cost.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f"{N_POINTS} (S, K, tau) points; legs A/B best-of-{REPEATS} after warm-up, legs C/D single timed pass\n")
    print(f"{'leg':<48}{'seconds':>10}{'s/point':>11}{'x CF':>9}")
    for row in rows:
        print(f"{row['leg']:<48}{row['seconds']:>10.4f}"
              f"{row['seconds_per_point']:>11.6f}{row['rel_to_cf']:>9.1f}")
    print("\nwrote results/reanalysis/label_cost.csv")


if __name__ == "__main__":
    main()
