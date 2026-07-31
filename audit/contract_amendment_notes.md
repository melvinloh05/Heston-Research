# Contract amendment notes (pre-freeze)

Branch `contract/v6-amendment`, one commit. This document is the required output of that
task: the annotated diff, the per-value behaviour-change statement, and any placement
awkwardness — for the human reviewer to check by hand before anything merges.

`python -m pytest -q` after the edit: **189 passed** — identical to the pre-edit baseline
(`audit/FINDINGS.md` line 3). No code path reads any of the new keys yet, which is the
expected result: this task touched only `heston_benchmark_v6.yaml` and `CLAUDE.md`, no
`.py` file.

---

## Annotated diff

```diff
diff --git a/heston_benchmark_v6.yaml b/heston_benchmark_v6.yaml
index b4e1e2e..e86c3c5 100644
--- a/heston_benchmark_v6.yaml
+++ b/heston_benchmark_v6.yaml
@@ -44,6 +44,7 @@ oracle_headroom_gate:
   decision_rule: >
     If spread < pre-registered 10% CVaR95 threshold at chosen frequency and TC tiers,
     RETUNE (rebalance frequency, misspecification severity) before M4. Non-optional.
+  spread_threshold_rel: 0.10   # C1 — was gate_headroom.py:361 literal; the "10%" named above
   motivation: "Sakuma Table 5 (oracle ~= network hedge) is the power warning."
   runs_before: all_training
```
**Resolves:** finding **C1** (5-undeclared-parameters inventory) / audit inventory row
`gate_headroom.py:361 gate spread thresh`. The `10%` was already stated in prose in
`decision_rule` two lines up; this adds the same number as a machine-readable key under the
section that already owns the concept, per task rule 3 ("place under the section that
already owns its concept").

```diff
@@ -108,7 +109,7 @@ splits:
   heldout_greek_and_hedging: [near_feller, strong_neg_corr]        # OOD type 1: BOTH metrics
   ood_models_hedging_only: [bates, merton]                          # OOD type 2: hedging PnL only
   tau_maturity_holdout: {enabled: true, role: sanity_check}
-  moneyness_wing_holdout: {enabled: true, role: sanity_check, gamma_metric: absolute_only}
+  moneyness_wing_holdout: {enabled: true, role: sanity_check, gamma_metric: absolute_only, moneyness_bounds: [0.75, 1.30]}  # bounds: C1 — was eval_greeks.py:45
```
**Resolves:** **C1** / `eval_greeks.py:45 moneyness wing bounds`. Added as a fourth field
inside the existing flow mapping to match the file's existing style for this key
(`{enabled, role, ...}`), rather than breaking it into block form.

```diff
@@ -171,6 +172,36 @@ loss:
   toggles: {use_pde: lambda_pde, supervise_delta: lambda_D, supervise_gamma: lambda_G, supervise_vega: lambda_v}
   lambda_discipline: {tune_on: validation_only, assert_test_untouched: true, log_selected_values: true}
 
+# -----------------------------------------------------------------------------
+# LAMBDA SELECTION  (Q3 — sourcing decision, pre-registered before any run)
+# -----------------------------------------------------------------------------
+# Shared lambda_pde across arms is the correct control (one model class, identical
+# architecture/ansatz; see methods.ansatz_control) — the open question Q3 resolves is which
+# arm the SHARED value is sourced from, not whether to share it.
+lambda_selection:
+  lambda_delta: {value: 1.0, fixed: true, swept: false}       # train.py:_apply_lambdas — "fixed 1.0; not a swept axis"
+  lambda_pde:
+    source_arm: standard_pinn                                  # Q3 decision — was rung3_delta_gamma_vega (train.py:135)
+    rationale: >
+      Tuning a shared hyperparameter on the treatment arm (rung3) and applying it to the
+      baseline (standard_pinn) handicaps the baseline, for which the PDE residual is the
+      ONLY structural signal. Sourcing lambda_pde from standard_pinn's own validation score
+      is the conservative direction wrt the study's hypothesis.
+    candidates: [0.0, 0.01, 0.1, 1.0]                          # train.py:139-140 sweeps.lambda_pde default
+  lambda_gamma:
+    source_arm: rung3_delta_gamma_vega                          # at the lambda_pde fixed immediately above
+    candidates: [0.3, 1.0, 3.0]                                # train.py:141 — playbook pre-reg {0.3,1,3}
+  lambda_vega:
+    source_arm: rung3_delta_gamma_vega                          # at the lambda_pde fixed immediately above
+    candidates: [0.3, 1.0, 3.0]                                # train.py:142 — playbook pre-reg {0.3,1,3}
+  tune_on: validation_only
+  robustness_row:
+    description: >
+      The confirmatory contrast (rung3 vs standard_pinn) is ADDITIONALLY reported at the
+      rung3-sourced lambda_pde, so the sourcing choice can be shown to be immaterial to the
+      headline result (or shown not to be).
+    status: robustness_result_not_a_second_confirmatory_test    # Q3 — does not replace acceptance_thresholds.confirmatory_cell_pass
+
 # -----------------------------------------------------------------------------
 # LABEL-NOISE DOSE-RESPONSE  (correctness axis; margin_X dissolved)
 # -----------------------------------------------------------------------------
```
**Resolves:** **Q3** (λ_pde sourced from the baseline arm, not the treatment arm). New
top-level section, placed immediately after `loss.lambda_discipline` since that is the key
that currently states "tune_on: validation_only, assert_test_untouched: true" for the loss
weights in general — `lambda_selection` is the specific, per-λ elaboration of that same
discipline. Declares: the source arm for each λ, that selection is validation-only, the
candidate grids (read off `train.py:139-142`), that λ_delta stays fixed at 1.0 (unchanged —
task rule says only the λ_pde *sourcing* changes), and the robustness row the task asked to
be pre-registered as a robustness result rather than a second confirmatory test.

```diff
@@ -193,6 +224,7 @@ label_noise_dose_response:
 # -----------------------------------------------------------------------------
 information_matching:
   baseline_price_points: grow_until_greek_accuracy_plateaus
+  plateau_tol: 0.02   # C1 — was run_info_matching.py:62 PLATEAU_TOL; rel-improvement floor below which the curve reads as plateaued
   cap: 5N
```
**Resolves:** **C1** / `run_info_matching.py:62 PLATEAU_TOL`. Placed under
`information_matching`, the section that already owns "grow_until_greek_accuracy_plateaus".

```diff
@@ -206,7 +238,16 @@ hedging_simulation:
   strategy: delta_only_underlying
   instrument: {type: european_call, S0: 100.0, K: 100.0, tau0: 0.25}
   horizon: {T_prime: 0.17, construction: "..."}
-  rebalancing: {frequency: daily, dt: 0.003968, fixed_across_arms: true}
+  # Q1: T_prime*frequency_per_year = 42.84 is not an integer, so a literal dt=0.003968 (1/252)
+  # was never achievable given T_prime=0.17. T_prime stays fixed (see horizon above); dt is
+  # now DERIVED, not declared. n_steps = round(T_prime * frequency_per_year); the realized
+  # grid is linspace(0, T_prime, n_steps+1) (Hedging_backtest.py:941; audit finding H1).
+  rebalancing:
+    frequency: daily
+    frequency_per_year: 252         # target frequency that produces n_steps below
+    n_steps: 43                     # Q1 — round(T_prime * frequency_per_year); Hedging_backtest.py:941
+    dt_realized: 0.003953488        # Q1 — T_prime / n_steps (252.94 rebalances/year, not 252)
+    fixed_across_arms: true         # unchanged: identical grid for every arm, no comparison affected
```
**Resolves:** **Q1** (finding H1). `T_prime` is untouched at 0.17 per the task's decision.
`dt` is retired as an independent literal; `n_steps: 43` and `dt_realized: 0.003953488` are
the derived quantities `Hedging_backtest.py:941` already computes at runtime, now written
down as what actually ran, plus `frequency_per_year: 252` kept as the *target* frequency
that produces `n_steps` via `round(T_prime * frequency_per_year)`. `fixed_across_arms: true`
carried over unchanged — task rule 2 says it "remains the property that matters."

```diff
@@ -278,12 +319,21 @@ acceptance_thresholds:
   confirmatory_cell_pass: "..."
   headline_scale_free: fraction_of_baseline_to_oracle_gap_closed
   order_attribution: "rung2 beats rung1 at the cell, CI excludes 0; failure = honest null"
-  dose_response: "monotone (isotonic + rank correlation); flat = regularization null"
+  dose_response:                              # Q2 — was one-line prose "monotone (isotonic + rank correlation); flat = regularization null"
+    criteria: "Spearman rho > 0 (shape confirmed via isotonic fit) AND one-sided seed-bootstrap tail probability P(rho<=0) < bootstrap_tail_prob_max"
+    bootstrap_tail_prob_max: 0.05             # C1/Q2 — was analyze_results.py:450 spearman_p_max
+    tail_prob_definition: "a one-sided BOOTSTRAP TAIL PROBABILITY over seeds that rho<=0 -- NOT a classical p-value"
+    interpretation: >
+      Low-powered at 5-7 dose levels: a 'flat' verdict establishes ONLY "monotonicity not
+      demonstrated", NOT that the dose-response IS flat. These are different claims; the
+      memo must not conflate them.
+    conservatism: "this criterion is conservative wrt the study's hypothesis -- it can only move a verdict toward the pre-registered null"
   ood_gamma_rmse_reduction_min: 0.15
   ood_vega_rmse_reduction_min: 0.15           # Greek-accuracy leg
   price_parity_within: 0.10
   mechanism_falsifier: "..."
   in_model_hedging: NOT_PASS_FAIL             # = (in-model x cost) 2x2 corner; must reproduce Sakuma null
+  sakuma_null_rel_tol: 0.02                   # C1 — was analyze_results.py:622 rel_tol; band for the in_model_hedging consistency check above
   seeds: {default: 5, confirmatory_cell: 10}
```
**Resolves:** **Q2** (the `dose_response` key becomes a block with an explicit
`bootstrap_tail_prob_max`, a `tail_prob_definition` stating it is a bootstrap tail
probability over seeds and not a classical p-value, an `interpretation` clause distinguishing
"monotonicity not demonstrated" from "is flat," and a `conservatism` note) and **C1**
(`sakuma_null_rel_tol: 0.02`, placed directly under `in_model_hedging` since that is the key
it operationalizes — the consistency check the contract already names as `NOT_PASS_FAIL`).

---

## Behaviour-change statement, per changed value

| key | value | code path today | behaviour changes today? |
|---|---|---|---|
| `oracle_headroom_gate.spread_threshold_rel` | 0.10 | none reads it (`gate_headroom.py:361` still hardcodes `0.10`) | **No.** |
| `splits.moneyness_wing_holdout.moneyness_bounds` | [0.75, 1.30] | none reads it (`eval_greeks.py:45` still hardcodes it) | **No.** |
| `information_matching.plateau_tol` | 0.02 | none reads it (`run_info_matching.py:62` still hardcodes `PLATEAU_TOL`) | **No.** |
| `hedging_simulation.rebalancing.n_steps` / `dt_realized` / `frequency_per_year` | 43 / 0.003953488 / 252 | none reads these keys (`Hedging_backtest.py:941` recomputes `n_steps` itself from `T_prime` and `frequency_per_year`, which was already 252 in the old flat mapping) | **No.** The old `dt: 0.003968` key was also never read by any code path (grep confirms — `audit/FINDINGS.md` H1 quotes the engine computing its own `n_steps`/`dt`, not reading a `dt` key at all), so removing it changes nothing either. |
| `acceptance_thresholds.dose_response.*` | see above | none reads it (`analyze_results.py:450` still hardcodes `spearman_p_max=0.05`) | **No.** |
| `acceptance_thresholds.sakuma_null_rel_tol` | 0.02 | none reads it (`analyze_results.py:622` still hardcodes `rel_tol=0.02`) | **No.** |
| `lambda_selection.lambda_pde.source_arm` | `standard_pinn` (was `rung3_delta_gamma_vega`) | `train.py:135` still hardcodes `load_arm(args.pinn_cfg, "rung3_delta_gamma_vega")` as the base arm for `--select-lambdas` | **Protocol only, no numerical change until λ selection is rerun** — this is the one item the task flagged in advance as a protocol change rather than a value transcription (task rule 2). The contract now states the intended source; `train.py:135` has not been changed (task rule 3: no `.py` file touched) and must be updated in a separate batch before any λ-selection run is treated as contract-compliant. |
| `lambda_selection.lambda_delta/gamma/vega.candidates`, `tune_on`, `robustness_row` | see diff | none reads these keys yet | **No** (declaration only; these are the grids `train.py:139-142` already uses as its own defaults). |

Net: everything in §1-3 of the task is a **pure transcription** — every value equals what
the code already does, and no code reads the new keys yet, so nothing about a run today
would differ. §4 (Q3) is the one **protocol** change, and it is explicitly not yet wired to
any code path.

---

## Placement notes / downstream-read awkwardness

- **`lambda_selection` duplicates `loss.lambda_discipline`'s `tune_on: validation_only` and
  `assert_test_untouched: true` fields conceptually** (same discipline, restated per-λ). A
  future code change threading `lambda_selection` into `train.py` should probably read
  `tune_on` from one place, not both — flagging so nobody adds a second, divergent
  `tune_on` value under `loss.lambda_discipline` later.
- **`hedging_simulation.rebalancing` changed from a flow mapping (`{...}`) to a block
  mapping.** Every other key in `hedging_simulation` (`instrument`, `paths`,
  `pnl_convention`, `transaction_costs`) mixes both styles already, so this isn't a
  structural outlier, but a downstream reader doing a naive `grep "rebalancing:"` for the
  old one-line form will not match. `frequency_per_year: 252` living under `rebalancing`
  rather than under `paths` or a new top-level `simulation_clock` key is a judgment call —
  it's the natural home since it's the number that determines `n_steps`, but a future
  code change wiring this key should read `hedging_simulation.rebalancing.n_steps`
  directly rather than recomputing `round(T_prime * frequency_per_year)` a second time in
  Python, to avoid a second place the 42.84-rounds-to-43 arithmetic could drift.
- **`frequency_per_year: 252` duplicates `hedging_config.yaml:24`**, which is the key the
  engine actually reads (`eng["rebalancing"]["frequency_per_year"]` at
  `Hedging_backtest.py:863,941,1218` — `eng` is the resolved *engine* config, i.e.
  `hedging_config.yaml`, not this pre-registration file). This mirrors the
  "duplicated-but-equal" pattern C1 already documents for the other 19 quantities (a
  pre-registration copy alongside an operational copy in code/engine config); it is why the
  behaviour-change table above says no code reads the new `heston_benchmark_v6.yaml` key —
  the engine was already getting 252 from `hedging_config.yaml` before this edit and
  continues to. Flagging so a future single-source-of-truth pass (C1's fix) treats
  `hedging_config.yaml:24` as the one to thread through, not a second copy to invent.
- **`dose_response` and `moneyness_wing_holdout` both went from a scalar/short value to a
  richer structure.** Any future code reading `bm["acceptance_thresholds"]["dose_response"]`
  expecting a string (there is no such code today — table in C1 confirms
  `analyze_results.py` doesn't open the contract at all) needs to read
  `dose_response.bootstrap_tail_prob_max`, not `dose_response` itself. Noted here so the
  separate code-change batch (C1's one-line fix: thread `resolve_config`'s
  `acceptance_thresholds` into the verdict functions) knows the shape changed, not just
  that a value was added.
- **No key was awkward enough to warrant a flat catch-all** (task rule explicitly forbids
  one); every new key found a section that already owns its concept.

---

## Out of scope, confirmed left alone

- `Hedging_backtest.py:49` QE `gamma1`/`gamma2 = 0.5` — per task instruction, this is the
  Andersen central-scheme choice, not a contract quantity. Not added anywhere.
- `CLAUDE.md` — updated with four one-line pointers (§5 of the task): the derived-`dt`
  invariant, the λ_pde sourcing decision, and the C1/Q2 threshold pointers folded into the
  existing "Dose-response" bullet since they're all newly-declared thresholds. No
  restatement of the full amendment rationale there.
