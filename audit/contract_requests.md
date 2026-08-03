# contract_requests.md — keys the code needs that `heston_benchmark_v6.yaml` does not declare

Written during fix batch 1 (branch `fix/audit-batch-1`). The contract is READ-ONLY to the
agent and was amended by hand in `a5a08a1`; these are the values fix batch 1 needed and did
NOT find. Each is left as a single literal carrying a `TODO(C1)` marker at exactly one site,
so there is one place to edit when (if) the key is declared.

---

## 1. `confirmatory_rel_threshold` — the 10% relative CVaR95 improvement (P1, blocking-ish)

**Needed by.** `analyze_results.confirmatory_cell` — the headline verdict of the study.

**Where the literal now lives.** `Hedging_backtest.contract_thresholds`, key
`confirmatory_rel_threshold = 0.10`, marked `TODO(C1)`. It is the ONLY re-typed
threshold left in the codebase; every other verdict number is read from the YAML.

**What the contract says today.** Prose in two places, no number as data:

```yaml
acceptance_thresholds:
  confirmatory_cell_pass: "misspec delta CVaR95 improvement >=10% relative AND paired-bootstrap 95% CI excludes 0 (combined,1% TC,baseline,rung3 vs standard_pinn)"
metrics:
  cvar_convention: {loss: "-PnL", level: 0.95, threshold_form: "relative >=10% AND paired-bootstrap 95% CI excludes 0"}
```

**Requested amendment** (mirrors the shape the C1 amendment already used for
`oracle_headroom_gate.spread_threshold_rel`):

```yaml
acceptance_thresholds:
  confirmatory_cell_rel_min: 0.10   # the ">=10% relative" named in confirmatory_cell_pass
```

**Why it matters.** The gate's identical 10% IS declared
(`oracle_headroom_gate.spread_threshold_rel`), so the two numbers that must move together —
the ceiling the gate measures and the floor the confirmatory verdict demands — are currently
declared in different places (one YAML, one Python). Editing the gate's 10% without editing
the Python leaves the confirmatory verdict on the old number, silently.

**Not urgent for compute.** The value is correct today (0.10 == the prose), and no result
moves. This is a pre-registration-hygiene request, not a bug.

---

## 2. Nothing else was missing

Every other threshold fix batch 1 needed was present in the amended contract and is now read
from it:

| value | contract key |
|---|---|
| CVaR level 0.95 | `metrics.cvar_convention.level` |
| bootstrap B 2000 | `hedging_config.yaml risk.bootstrap_B` (engine supplement, by design) |
| global seed 42 | `meta.global_seed` |
| confirmatory seed count 10 | `meta.seeds_confirmatory_cell` |
| confirmatory cell (direction / tc / regime) | `hedging_simulation.confirmatory_cell` |
| TC tiers | `hedging_simulation.transaction_costs.tiers` |
| OOD Γ / ν reduction 0.15 | `acceptance_thresholds.ood_{gamma,vega}_rmse_reduction_min` |
| price parity 0.10 | `acceptance_thresholds.price_parity_within` |
| Sakuma-null band 0.02 | `acceptance_thresholds.sakuma_null_rel_tol` (C1 amendment) |
| dose bootstrap tail prob 0.05 | `acceptance_thresholds.dose_response.bootstrap_tail_prob_max` (Q2 amendment) |
| gate spread threshold 0.10 | `oracle_headroom_gate.spread_threshold_rel` (C1 amendment) |
| wing bounds (0.75, 1.30) | `splits.moneyness_wing_holdout.moneyness_bounds` (C1 amendment) |
| plateau tol 0.02 | `information_matching.plateau_tol` (C1 amendment) |
| λ sourcing arms | `lambda_selection.lambda_{pde,gamma,vega}.source_arm` (Q3 amendment) |
| realized dt / n_steps | `hedging_simulation.rebalancing.{n_steps,dt_realized}` (Q1 amendment) |

The confirmatory MAGNITUDE (m = 1.0) is not requested as a key: it is structural, not a
threshold — the engine's `shift_at_m1` is asserted onto the contract's perturbation targets by
`Hedging_backtest._assert_contract_targets`, so m = 1.0 IS the contract endpoint by
construction.
