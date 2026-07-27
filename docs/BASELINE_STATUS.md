# BASELINE_STATUS

_Generated 2026-07-13. Baseline establishment run for Sobolev-PINN Greeks for Heston (v6)._

## Environment

| Component | Version |
|-----------|---------|
| Python    | 3.12.7  |
| torch     | 2.5.1   |
| numpy     | 1.26.4  |
| scipy     | 1.11.4  |
| PyYAML    | 6.0.1   |
| Platform  | macOS-26.5.2-arm64 (Darwin 25.5.0) |

## Test suite (`python -m pytest -q`)

**20 passed, 0 failed** (2.9 s).

| File | Passed |
|------|--------|
| test_sobolev_pinn.py    | 11 |
| test_hedging_backtest.py | 9 |

No environment or test fixes were required — the suite was green as checked out.

> Note: the prior memory note ("8 pre-existing hedging failures, Hedging_backtest.py:39
> wants missing v4 yaml") is **stale**. Hedging_backtest.py resolves against
> `heston_benchmark_v6.yaml` + `hedging_config.yaml` and all hedging tests pass.

## Oracle self-test (`python oracle.py`, quick)

All four trust gates **PASS**:

1. Literature prices (Albrecher Table 1, phi_2 column, T=1..15) — PASS
2. Trust gate xi->0 == Black-Scholes (legs cf, fd; all Greeks) — PASS
3. Three legs agree within tol on baseline (0 masked of 15 points) — PASS
4. MC leg bit-identical from seed — PASS

`ALL SELF-TESTS PASSED — trust gate cleared; oracle output usable.`

(Quick selftest only; `python oracle.py --full` is the pre-freeze certification and was not run.)

## Config load check

`pinn_config.yaml`: all **18 arms** load via `SobolevPINN.load_arm` and instantiate as
`SobolevPINN`, each with an **identical 13121 parameters** (single-architecture invariant holds).

```
rung0_price_only, rung1_delta, rung2_delta_gamma, rung3_delta_gamma_vega,
standard_pinn, feedforward, info_matched_baseline,
sigma_000, sigma_010, sigma_025, sigma_050, bs_gamma, shuffled,
gradient_penalty_only, sobolev_sans_pde, gamma_only, lambda_pde_zero,
optional_vanna_arm
```

`hedging_config.yaml`: `Hedging_backtest.resolve_config(contract, engine)` returns
`{benchmark, engine, derived}`; derived seeds `= [42, 43, 44, 45, 46]`
(global_seed 42 + 0..seeds_min-1). No load errors.

## Housekeeping

- `.gitignore` already present and comprehensive — left unchanged.
  (Observation, no action taken: it globs `*.pdf`, which would exclude the project
  reference `Albrecher et al. 2007.pdf` from version control; flag for a human decision.)
- `docs/` created (this file is its first occupant).

## Git status

**Not a git repository** — `git status` reports `fatal: not a git repository`. The repo map in
CLAUDE.md assumes version control; `git init` has not been run. No commits, no branch.
