"""Compare two numerical reference implementations at three option states.

Run from the repository root with ``python -m examples.reference_demo``.
This CPU example reads no datasets or checkpoints and writes no files.
"""

import numpy as np

from oracle import HestonParams, heston_greeks_cf, heston_greeks_fd


def main():
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.50, v0=0.04)
    spot = np.array([90.0, 100.0, 110.0])
    strike, time_to_expiry = 100.0, 0.25
    interest_rate, dividend_yield = 0.02, 0.0

    analytic = heston_greeks_cf(
        spot, strike, time_to_expiry, params, interest_rate, dividend_yield
    )
    finite_difference = heston_greeks_fd(
        spot, strike, time_to_expiry, params, interest_rate, dividend_yield
    )

    print("Heston call at strike 100 and time to expiry 0.25 years")
    print("   spot      price      delta      gamma    dValue/dVariance")
    for index, value in enumerate(spot):
        print(
            f"{value:7.1f} {analytic.price[index]:10.6f}"
            f" {analytic.delta[index]:10.6f} {analytic.gamma[index]:10.6f}"
            f" {analytic.vega[index]:18.6f}"
        )

    print("\nMaximum absolute difference between CF and FD-on-COS references")
    for name in ("price", "delta", "gamma", "vega"):
        difference = np.abs(getattr(analytic, name) - getattr(finite_difference, name))
        print(f"  {name:6s} {np.max(difference):.3e}")
    print("\nHere vega means variance sensitivity. Solver agreement is not an error bound.")


if __name__ == "__main__":
    main()
