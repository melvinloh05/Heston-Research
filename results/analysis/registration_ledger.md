# Pre-registration ledger

Every pre-registered commitment, its contract provenance, and its outcome. Declared values are read live from `heston_benchmark_v6.yaml`.

`null` = NOT EVALUATED (no claim made). `error` = evaluation attempted and FAILED (a defect, never a study outcome). These are never merged.

## Decision criteria

| Commitment | Declared (from contract) | Amend | Observed | Verdict |
|---|---|---|---|---|
| Primary: misspecified delta-only CVaR95, rung3 vs standard_pinn | `confirmatory_cell_rel_min` = 0.1; `confirmatory_cell_pass` = misspec delta CVaR95 improvement >=10% relative AND paired-bootstrap 95% CI excludes 0 (combined,1% TC,baseline,rung3 vs standard_pinn) | AM2-1 | 0.0002234903383255538<br>CI [-0.024212200379693183, 0.02423093869711561] | **fail** |
| Order attribution: rung2 beats rung1 (the add-Gamma rung) | `order_attribution` = rung2 beats rung1 at the cell, CI excludes 0; failure = honest null | - | 0.05735513281078042<br>CI [0.052741557352067224, 0.062096044477346665] | **fail** |
| Gamma-label-noise dose-response (monotonicity) | `bootstrap_tail_prob_max` = 0.05; `criteria` = Spearman rho > 0 (shape confirmed via isotonic fit) AND one-sided seed-bootstrap tail probability P(rho<=0) < bootstrap_tail_prob_max | Q2 | 0.028571428571428574<br>CI [-0.08571428571428573, 0.7714285714285715] | **flat** |
| OOD Greek RMSE reduction at price parity (rung3, binding) | `ood_gamma_rmse_reduction_min` = 0.15; `ood_vega_rmse_reduction_min` = 0.15; `price_parity_within` = 0.1 | - | 0.8635422644085887<br>CI [0.8592891818754326, 0.8677953469417449] | **fail** |
| OOD Greek RMSE reduction (rung2) — SECONDARY, non-binding | `ood_gamma_rmse_reduction_min` = 0.15; `ood_vega_rmse_reduction_min` = 0.15; `price_parity_within` = 0.1 | - | 0.8696520127917692<br>CI [0.8491583179213957, 0.8901457076621426] | **fail** |
| In-model x zero-cost corner reproduces the Sakuma null | `sakuma_null_rel_tol` = 0.02; `in_model_hedging` = NOT_PASS_FAIL | C1 | 0.2722162347775128<br>CI [-0.6647999834254114, -0.6311080423167322] | **flag** |
| Mechanism: robustness (i) vs transaction-cost/turnover (ii) | `mechanism_falsifier` = 0%-TC gap => robustness (i); zero-at-0% widening + T_ex->0 => cost (ii); T_ex unmoved rejects (ii). Both publishable. STATE BEFORE RESULTS. | - | -0.9624408151980117<br>CI [-0.9803943819851954, -0.9443202610140843] | **channel_i** |
| Bates severity sweep: locate a decision-relevant regime | `goldilocks_bates` = ['decision_relevant_regime_located', 'no_decisive_regime'] | - | 0.31054061254121645<br>CI [0.2828776480073575, 0.3386475874633602] | **no_decisive_regime** |

## Design commitments

| Commitment | Declared (from contract) | Amend | Observed | Verdict |
|---|---|---|---|---|
| Oracle-headroom gate ran BEFORE any training | `runs_before` = all_training; `spread_threshold_rel` = 0.1 | AM3 | — | **held** |
| Gate decision rungs and region of validity | `decision` = [0.05, 0.1, 0.15]; `clipped_frac_max` = 0.25 | AM2-3 | — | **held** |
| Seeds: default / confirmatory cell | `default` = 5; `confirmatory_cell` = 10 | - | — | **held** |
| Transaction-cost tiers | `tiers` = [0.0, 0.01, 0.02] | - | — | **held** |
| Confirmatory cell definition | `confirmatory_cell` = {'perturbation': 'combined', 'tc_tier': 0.01, 'regime': 'baseline', 'contrast': 'rung3_vs_standard_pinn', 'seeds': 10} | - | — | **held** |
| Tail claims require paired bootstrap over CRN paths | `tail_claim_requires` = paired_bootstrap_over_CRN_paths_with_seed_variance_separated | - | — | **held** |

