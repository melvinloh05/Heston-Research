# CLAUDE.md — Sobolev-PINN Greeks for Heston (v6)

## What this project is
One causal study: does explicit supervision of Gamma (with Vega) in a trained, PDE-retaining,
parametric Heston PINN improve DELTA-ONLY hedging PnL under misspecified dynamics — and through
which channel (transaction-cost/turnover vs model-uncertainty robustness)? Attribution runs on a
supervision ladder (price → +Δ → +Δ+Γ → +Δ+Γ+ν; the claim lives on the rung1→rung2 gap) and a
Gamma-label-noise dose-response. The residual×supervision factorial is completed by a
Sobolev-sans-PDE arm (ω_PDE=0). The mechanism is ADJUDICATED, NOT ASSUMED — pre-registered 2×2,
TC sweep {0,1,2%}, excess-turnover statistic T_ex. Do not pre-commit to the cost outcome.

## Single source of truth
`heston_benchmark_v6.yaml`. It is READ-ONLY for you: never edit it; if a task seems to require
changing it, STOP and say so — that is a human decision. When code and YAML disagree, the YAML
wins and the code changes.

## Non-negotiable invariants (violating any of these invalidates results)
- Strategy: delta-only hedging of the underlying. Delta-gamma is secondary, evaluation-only.
- PRIMARY metrics are HELD-OUT ONLY: OOD-param Greek RMSE+tails (near_feller, strong_neg_corr;
  Δ,Γ,ν,vanna) and misspecified delta-only hedging CVaR95 (loss = −PnL) at TC ∈ {0, 0.01, 0.02}.
- PnL convention: self-financing; initial premium = θ_train ORACLE price for ALL arms;
  terminal liability mark = true-DGP price at T′; hedge horizon T′ = 0.17 on the
  (S0=100, K=100, τ0=0.25) call; rebalancing daily, fixed across arms. `dt` is DERIVED, not
  declared: `hedging_simulation.rebalancing.n_steps`/`dt_realized` (dt_realized=0.003953488,
  not 1/252 — T′=0.17 doesn't divide evenly into daily steps; contract amendment Q1).
- CRN: all arms hedge the SAME paths within a cell. Path banks are frozen artifacts.
- Training sampling: Latin hypercube over κ∈[1,4], θ∈[0.02,0.12], ξ∈[0.20,0.60], ρ∈[−0.80,−0.20],
  v0∈[0.01,0.12]; REJECT Feller 2κθ/ξ² < 0.40; EXCISE balls of 10% relative radius around every
  named evaluation anchor. Named regimes are EVAL ANCHORS, never training data.
- Oracle: 3-way cross-validation (CF-analytic / FD-on-COS / MC), agreement tol_rel = 1e-3; 4th
  ADI leg REQUIRED for near_feller, feller_violating_volvol, and any sampled point with Feller
  ratio in [0.40, 0.60]. Quantities: price, delta, gamma, vega, vanna. Mask points where legs
  disagree; mask is declared pre-training and must pass the three neutrality checks.
- One model class; every arm is a PINNConfig. Identical architecture, identical ansatz/terminal
  treatment across arms. λ selection on VALIDATION ONLY (LockedTestSet enforces). Sourcing of
  the shared λ_pde is a pre-registered decision, not a default: see `lambda_selection` in the
  contract (λ_pde ← `standard_pinn`, λ_gamma/λ_vega ← `rung3_delta_gamma_vega`; contract
  amendment Q3). `train.py:_run_select_lambdas` sources λ_pde from
  `lambda_selection.lambda_pde.source_arm` (the contract), not a literal.
- Seeds: 5 default, 10 on the confirmatory cell. Confirmatory cell = (combined perturbation,
  1% TC, baseline regime, rung3 vs standard_pinn).
- Every run logs: config hash, seed, wall clock, param count, derivative-eval count, peak memory.
- PnL convention is LIVE: initial premium = theta_train ORACLE provider price for ALL arms
  (engine premium_override; rows carry premium_convention_ok), horizon T'=0.17 with
  true-DGP terminal mark (Bates CF / Merton series); the engine RAISES rather than silently
  settling at expiry.
- Loss terms are scale-normalized (loss_scale_mode label_second_moment); the GAMMA scale is
  frozen from the TRUE consensus (batch key gamma_ref) so every dose-response arm and the
  gradient-penalty arm share ONE normalization. Never remove gamma_ref from batches; never
  let per-arm labels set the gamma scale.
- Anchor excision (10% relative-radius L2 ball in range-normalized coords, all five named
  regimes) is part of every training/label sampling call. Never sample without anchors.
- The smoothing control is a PURE no-trade band (no EMA). t_ex = sum-incl-endpoints
  definition; turnover column is the legacy per-step mean.
- resolve_config asserts engine shift_at_m1 lands exactly on contract perturbation targets.

## Pre-registered thresholds (state before results; report failures as honest nulls)
- Confirmatory pass: misspec delta CVaR95 improvement ≥10% relative AND paired-bootstrap 95% CI
  excludes 0, at the confirmatory cell — the number is `acceptance_thresholds.
  confirmatory_cell_rel_min` (0.10, contract amendment AM2-1); no Python literal remains.
- Order attribution: rung2 beats rung1 at the cell, CI excludes 0.
- Dose-response: Spearman ρ>0 (isotonic fit for shape) AND one-sided seed-bootstrap tail
  probability P(ρ≤0) < `acceptance_thresholds.dose_response.bootstrap_tail_prob_max` (0.05,
  NOT a classical p-value); flat = "monotonicity not demonstrated", NOT "is flat" — contract
  amendment Q2. Sakuma-null consistency check band: `acceptance_thresholds.sakuma_null_rel_tol`
  (0.02, contract amendment C1). Gate go/no-go threshold: `oracle_headroom_gate.
  spread_threshold_rel` (0.10, C1). Moneyness wing bounds: `splits.moneyness_wing_holdout.
  moneyness_bounds` (C1). Info-matching plateau tolerance: `information_matching.plateau_tol`
  (0.02, C1).
- OOD Greek RMSE: Γ and ν reduction ≥15% at price parity within 10%.
- Mechanism: gap at 0% TC ⇒ robustness channel (i); zero-at-0% widening with TC and T_ex→0 ⇒
  cost channel (ii); T_ex unmoved kills (ii) regardless of PnL. Both readings publishable.
- In-model × κ=0 cell must reproduce the Sakuma null (consistency check, NOT pass/fail).
- Verdict vocabulary is pre-registered in `acceptance_thresholds.verdict_vocabulary` (AM2-2):
  `null` = NOT EVALUATED (artifact legitimately absent), `error` = evaluation attempted and
  FAILED (artifact present but corrupt). Never collapse the two in any table, memo or figure.
- Gate σ ladder and its region of validity: `oracle_headroom_gate.{sigma_rel_ladder,
  effective_sigma_reporting, region_of_validity}` (AM2-3) — decision rungs ≤0.15 only, 0.20 and
  0.40 diagnostic-only, and a gate whose pilot lands above `region_of_validity.clipped_frac_max`
  (0.25) is INCONCLUSIVE, not a pass or a no-go. Do not widen or remove the delta clip.
- The clip AMPLIFIES the delivered gamma error in the decision band (not "understates, so the
  gate is conservative" — that premise is measured false): see the `AM3-1` block above
  `sigma_rel_ladder` and `region_of_validity.interpretation`. Comparing the pilot against
  NOMINAL σ_rel would be anti-conservative; `effective_sigma_reporting.compare_pilot_against`
  is a correctness requirement.
- σ_rel = 0.20 is DIAGNOSTIC, not a decision rung — `sigma_rel_ladder.
  production_scale_measurement` (AM3-2): at production scale its clipped_frac straddles the
  bound (0.2470 mean / 0.2513 max, 3 of 10 seeds outside). `clipped_frac_max` stays 0.25;
  never raise it to re-admit a rung.
- A pilot above the decision band follows `oracle_headroom_gate.ladder_extension_contingency`
  (AM3-3), declared before any pilot ran: extension rungs enter DIAGNOSTIC, need a
  production-scale clipped_frac re-measurement with mean AND per-seed max inside the fixed
  bound, and any result is reported as an extended-ladder result.

## Declared design deviations
Pre-registered-machinery deviations that currently live only in code docstrings, recorded here
so no future session silently "corrects" them back. Rationales are quoted from the code.

1. **Gate corruption field is anisotropic + amplitude-matched iid** (`gate_headroom.py`).
   - Spec said: isotropic field bandwidth 1.0 in (S, v, τ); iid comparator scaled by
     σ·S·√(v·dt).
   - We do: ANISOTROPIC, S-dominated bandwidth (1.0, 0.1, 0.1); iid matched to the field's
     SPATIAL amplitude with no √dt factor.
   - Why (module docstring): "With an ISOTROPIC field, d(eta)/dv is as large as d(eta)/dS, and
     because the range-normalized variance coordinate moves ~4x more per daily step than the
     spot coordinate (measured on the confirmatory paths: mean |dz_v| ~ 0.038 vs |dz_S| ~
     0.009), the isotropic field injects per-step delta JITTER from variance motion — turnover
     that has nothing to do with a gamma error, making field t_ex EXCEED iid t_ex, the opposite
     of the design's premise." And: "The original sqrt(mean(v)*dt) scale is ~0.02 at daily dt,
     which suppresses iid turnover by construction and makes the 'iid overstates cost' contrast
     vacuous at the contract frequency."
   - Moves: the gate's iid-bracket numbers (iid arm t_ex and CVaR95 spread). The FIELD-mode
     ceiling and the human go/no-go are UNCHANGED — docstring: "Neither change touches the
     gate's CVaR95 spread reading or the go/no-go."

2. **Gate decision requires CI-excludes-0 in EVERY seed** (`gate_headroom.py`).
   - Spec said: threshold met "with CI excluding 0" (wording ambiguous, arguably a pooled CI).
   - We do: pick the smallest σ_rel with `ci_excludes_zero_frac == 1.0` — every seed's paired CI
     excludes 0 (the stricter reading).
   - Why (code comment): "smallest sigma_rel whose mean spread clears the pre-registered 10%
     relative threshold WITH every seed's paired CI excluding 0."
   - Moves: which σ_rel row the gate flags as the decision point per TC tier (a stricter, later
     trigger than a pooled CI would give).

3. **P13 pooled-diff CVaR synthetic test must NOT assert ~0** (`test_analyze_results.py`).
   - Spec said: synthetic check "±c shift in two seeds → pooled diff ~ 0".
   - We do: assert only that the opposing-shift pooled diff is a small residual (< 0.3·c), NOT
     ~0 — CVaR is a tail statistic, not translation-symmetric across opposing per-seed shifts
     (empirically pooled diff ≈ +0.18 for ±0.5, i.e. ~0.36·c).
   - Why (test docstring): "opposing per-seed shifts do NOT yield a genuine c-sized gap: the
     pooled diff is a small residual (~0.2c, a one-sided-tail boundary effect of CVaR), an order
     below the -c gap the SAME c produces under a constant shift."
   - Moves: nothing in the results — this is a guard on the TEST. The "pooled diff ~ 0"
     assertion must NEVER be added; the current < 0.3·c residual assertion is correct.

**DO-NOT-TOUCH — frozen bare-arithmetic MC seed derivations.** `make_datasets.py`
(`mc_seed = seed + 104729*ridx + 7919*j`) and `make_labels.py` (`seed=seed + 7919*(mc_seed_offset
+ i)`) fold indices into a scalar seed by bare arithmetic, violating the documented
`np.random.default_rng([seed, _STREAM_*])` SeedSequence stream-constant convention used
everywhere else (Hedging_backtest _STREAM_DIFFUSION/JUMP/BOOT/PAIRED, providers _STREAM_SPOTCHECK,
gate_headroom _STREAM_FIELD/IID, analyze_results _STREAM_POOLED). They are FROZEN: changing them
changes the frozen labels/datasets. No future session may "fix" them.

## Autonomy dial
- You may freely: write/edit code and tests, run pytest and local smoke runs, make plots/tables,
  triage logs.
- Propose and WAIT for approval before: anything that spends money (GPU dispatch), anything that
  freezes an artifact (dataset generation to data/frozen, path banks, git tags), deleting
  data/results, editing any YAML config, adding dependencies, choosing/changing the no-trade
  band width, promoting any artifact into data/frozen, editing lambdas_selected.yaml once
  written.

## Commands
- Tests: `python -m pytest -q` (all must pass before any commit)
- Oracle certification: `python oracle.py` (quick) / `python oracle.py --full` (pre-freeze)
- Style: match existing code — numpy-style, type hints, no new deps without approval.

## Repo map
oracle.py (4-leg Heston Greek oracle + cross-validation + selftest) · greek_labels.py (gamma-label
sources for dose-response) · SobolevPINN.py (single arm class; loss from config flags) ·
train_pinn.py (training, compute accounting, λ discipline, samplers) · Hedging_backtest.py
(model-agnostic delta-only hedging engine, QE paths, exact PnL decomposition) ·
providers.py (HestonCFProvider = theta_train oracle hedger, pathwise spot check) ·
make_labels.py (hypercube label artifact + 4th-leg band routing + mask-neutrality report +
build_arm_labels) · test_providers.py · test_make_labels.py · configs:
pinn_config.yaml, hedging_config.yaml, heston_benchmark_v6.yaml (read-only) · docs/ (
claim memo,).