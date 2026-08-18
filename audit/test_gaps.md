# test_gaps.md — Tier-1 behaviours with no falsifying test

Baseline: 189 passed. The suite is genuinely non-vacuous by the mechanical checks:

- **No `skip` / `xfail` markers anywhere.** (The `grep` hits for "skip" are function *names*
  about ledger/part resume, not pytest markers.)
- **No assertion-free tests.** Seven test bodies contain no bare `assert`, but all seven use
  `np.testing.assert_allclose` / `assert_array_equal` at tight tolerances
  (`rtol=1e-10`, and `assert_array_equal` — atol 0 — in
  `test_pinn_provider.py::test_parity_with_direct_model_call`).
- **Monkeypatching is confined to I/O boundaries**, never to the unit under test:
  `test_exhibits` patches the engine's overlay call (avoids simulation in tests),
  `test_run_hedging`/`test_train` patch training to keep runtime down,
  `test_modal_app` patches the cloud SDK. No test replaces the function it is asserting on.
- **Golden values are cross-checked, not self-generated**: `test_providers` compares
  `heston_greeks_cf_v0` against the *independent scalar* `heston_greeks_cf`, and
  `test_hedging_backtest::test_merton_price_vs_mc` compares closed form against Monte Carlo.

What follows is the real gap list: behaviours that would keep passing if broken.

---

## `analyze_results.py`

| # | Behaviour with no falsifying test | Nearest existing test | Why it still passes |
|---|---|---|---|
| 1 | `_mechanism_reading` must not label a *wrong-direction* gap as a channel | `test_mechanism_reading_four_patterns` | Every gap in all four patterns has a **negative** `diff` (`-1.0`, `-0.05`, `-2.0`). The sign-blind branch is never entered. → finding **A2** |
| 2 | The confirmatory verdict must be tied to the pre-registered seed count | `test_confirmatory_pass_and_fail` (10 seeds), `test_order_attribution_pass_and_null` (6 seeds) | Neither asserts on `n_seeds`; the 6-seed sibling establishes a short seed set as an accepted input. → **A3** |
| 3 | Thresholds must equal the contract's | — none — | No test opens `heston_benchmark_v6.yaml`. → **A1** |
| 4 | A *corrupt* (not absent) artifact must not degrade to `null` | `test_run_analysis_missing_artifacts_degrade_to_null` | The test asserts `null` **is** the correct outcome for absence; the broad `except Exception` makes corruption indistinguishable. → **A4** |
| 5 | `cvar_mean` and `isotonic_fit` must be computed on the same seed set | `test_dose_response_monotone_end_to_end` | The fixture gives every dose arm the same seeds, so `common == each arm's own seeds` and the two aggregations coincide. → **A5** |
| 6 | `rel_improvement` must behave when `cvar_b <= 0` | `test_pooled_stratified_constant_shift_excludes_zero` | All fixtures build losses with a positive tail. → Q4 |

**Top-5 tests that would catch these** (one line each):

1. `assert ar._mechanism_reading([g(0.0, +0.4, +0.2, +0.65)], tex_flat)["reading"] == "null"`
   — a significant *positive* tc=0 gap must not read as `channel_i`.
2. Build a confirmatory dir with 3 seeds, assert `verdict == "null"` and that a structured
   `n_seeds` field (not just `notes`) reports 3.
3. `for k, v in contract_thresholds.items(): assert getattr(ar, k) == v` — parametrized over
   `heston_benchmark_v6.yaml`'s `acceptance_thresholds` + `cvar_convention` + tc tiers.
4. Write a syntactically valid npz whose PnL arrays have mismatched shapes; assert
   `run_analysis` surfaces it as something *other* than the `null` used for absence.
5. Give one dose arm a missing seed; assert `cvar_mean` and `isotonic_fit` in the emitted row
   are computed over the same seed set.

---

## `Hedging_backtest.py`

The best-tested module in the repo. Genuine gaps:

| # | Behaviour with no falsifying test | Nearest existing test | Why it still passes |
|---|---|---|---|
| 1 | Realized `dt` must match `hedging_simulation.rebalancing.dt` | `test_resolve_config_exposes_p5_staging_and_tc_tiers` | Asserts `T_prime == 0.17` and `frequency_per_year == 252` **separately**; never divides one by the other. → **H1** |
| 2 | The QE exponential branch's atom at `v == 0` must reach the providers | `test_seed_determinism`, `test_no_look_ahead` | The baseline regime (Feller 1.78) produced **0 exact zeros in 22 000 states** in my run — the branch is simply not exercised at the tested regime. `test_pinn_provider::test_v_zero_finite_and_no_clamp` covers the provider side directly, so the risk is bounded, but no *engine* test drives a Feller-stressed cell. |
| 3 | Dividend income accrual (`q > 0`) | all hedging tests | The contract pins `q = 0.00`, so the `dy * pos * px * dt` term is identically zero in every test and in production. Untested but contract-excluded. |
| 4 | `charge_final_unwind: false` | `test_total_traded_t_ex_hand_example` | Config pins it `true`; the `else 0.0` branch is dead in practice. |

**Test that would catch #1:**
`assert abs(T_prime / n_steps - bm["hedging_simulation"]["rebalancing"]["dt"]) < 1e-6`
— it fails today at 0.37%, which is the point.

---

## `gate_headroom.py`

| # | Behaviour with no falsifying test | Nearest existing test | Why it still passes |
|---|---|---|---|
| 1 | The `0.10` decision threshold must equal the contract's | `test_report_and_csv_written` | Asserts the report contains a DECISION section, not which number produced it. → **G1** |
| 2 | The delta clip must not silently shrink the delivered corruption | `test_calibration_matches_target` | Calibration is asserted on `eta_dS` **before** the clip; the test never measures the delta error that survives `np.clip`. → **G2** |
| 3 | `field` vs `iid` t_ex ordering at *production* σ and path count | `test_field_tex_below_iid` | Runs at smoke sizes (`--n-paths 128 --n-seeds 1`-scale). The declared deviation #1 in CLAUDE.md rests on this ordering; it is asserted only in miniature. |
| 4 | iid mode's stateful `self._rng` across the engine's seed loop | `test_field_determinism` | Asserts determinism of **field** mode only (a pure function of state). iid mode is explicitly redraw-per-call, so its per-cell noise depends on evaluation order — untested, and contrast-only by design. |

**Top-5 tests overall** (the five I would write first, across all three Tier-1 modules):

1. **Sign guard on the mechanism reading** — `_mechanism_reading` with a positive tc=0 gap
   must not return `channel_i` (catches **A2**, the most dangerous finding).
2. **Contract-constant parity test**, parametrized over every threshold in
   `heston_benchmark_v6.yaml`, asserting the literals in `analyze_results`, `gate_headroom`
   and `eval_greeks` equal them (catches **A1**, **G1**, and every future drift).
3. **Seed-count gate on the confirmatory verdict** — 3-seed directory ⇒ not `pass`
   (catches **A3**).
4. **Realized-dt assertion** — `T_prime / n_steps == contract dt` (catches **H1**; fails
   today, which is the useful outcome).
5. **Row-cap assertion in the info-matching sweep** — construct a train split smaller than
   `5N` and assert the sweep raises or marks `plateau_reached=False`
   (catches **I1**).

---

## Cross-cutting gaps

- **No test renders a missing value in an exhibit.** `test_exhibits.py`'s `_HEDGE_METHODS`
  fixture populates `t_ex`, `gap_closed`, `tc_component` and `directional_component` for
  every method, and the suite hashes the **backing CSV**, never the PNG — which is exactly
  the artifact that stays correct when the figure does not. → **X1**
- **No test interrupts a run.** The resume path is tested
  (`test_ledger_resume_skips_completed_cells`) but only from a *cleanly finished* prior run,
  so the mid-run state of `headline_delta_only_per_seed.csv` and `resolved_config.yaml` is
  never observed. → **R1**
- **No test asserts the gate is invoked with the runlog's `sigma_gamma_pilot`.**
  `test_pilot_prints_finite_sigma_gamma` checks the value is finite and recorded; the
  human copy-paste step between `train.py --pilot` and `gate_headroom --sigma-gamma` is
  untestable as currently designed. → **T1**
- **No test exercises `train.py` in its default (early-stop) mode and checks the
  `matched_epochs` label.** `test_matched_epochs_checkpoint_exists_alongside_best` asserts
  both checkpoint files exist but not that `reached_max_steps` is `True`. → **T2**
