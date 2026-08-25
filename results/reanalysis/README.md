# Re-analyses on frozen artifacts (2026-08-24)

No new training, no new hedging runs, no contract or paper edits. Everything here is
computed from artifacts already frozen, plus one timing measurement of the oracle legs.
Reproduce with the `reanalysis_*.py` scripts at the repo root.

---

## R1 — Supervised-arm loss decomposition
`loss_decomposition.csv`, `loss_decomposition_agg.csv` · `reanalysis_loss_decomp.py`

**Question.** rung3's hedging CVaR is flat across a 100x range of `lambda_pde`. Is that
because the residual is a negligible share of its objective (CONDITIONAL robustness), or
does the residual carry real weight and the outcome not move anyway (STRONG)?

**Answer: STRONG.** lambda-weighted share of rung3's final training loss:

| lambda_pde | 0 | 1e-4 | 1e-3 | 3e-3 | 0.01 (registered) |
|---|---|---|---|---|---|
| PDE share | 0.000 | 0.144 | 0.327 | 0.421 | **0.533** |
| supervision share | 0.972 | 0.843 | 0.665 | 0.573 | 0.462 |

At the registered weight the residual is the single largest term in rung3's objective —
larger than delta, gamma and vega combined — and the hedging outcome still does not move.
For contrast, `standard_pinn` carries a comparable residual share (0.43–0.62) and its CVaR
moves 2.1370 -> 3.0518. Same objective dominance, opposite outcome. The protective effect
is a property of the trained field, not an artefact of the residual being small.

## R2 — Metric validation: does the hedge box predict better than the global grid?
`metric_validation_{per_seed,agg}.csv`, `..._nobsgamma.csv` · `reanalysis_metric_validation.py`

Rank 7 arms by derivative RMSE (global slice vs hedge-box slice) against realised
zero-cost misspecified CVaR95, per seed and regime. Clean comparison: same points, same
parameters, different support.

**The box wins, but only marginally.** Pooled pairwise concordance 0.733 (box) vs 0.710
(global) over all arms; 0.860 vs 0.835 with `bs_gamma` excluded. For DELTA on
`strong_neg_corr` the box is *worse* (rho 0.271 vs 0.350). `bs_gamma` is the dominant
principled departure — it mis-orders in 40/40 cells, which is the known minimum-variance
anomaly, not a metric failure.

**This does not support recommending the box on its own.** See R4.

## R3 — Mask difficulty
`mask_difficulty.csv`, `mask_difficulty_quintiles.csv`, `mask_hedgebox.csv`,
`mask_by_tau.csv` · `reanalysis_mask_difficulty.py`, `reanalysis_mask_hedgebox.py`

Overall mask rate 0.0979 (3209 / 32768 label cells). Masked vs retained, by axis:

| axis | masked mean | retained mean | KS | retained set easier? |
|---|---|---|---|---|
| curvature abs(Gamma) | 0.00415 | 0.00908 | 0.689 | no (mask sits on LOW curvature) |
| maturity tau | 0.202 | 0.599 | 0.634 | **yes** (mask sits on SHORT tau) |
| moneyness S/K | 0.722 | 1.088 | 0.556 | no (mask sits deep OTM) |
| Feller ratio | 2.706 | 3.073 | 0.088 | yes, mildly |
| abs(rho) | 0.552 | 0.497 | 0.149 | yes, moderately |

Masking concentrates where price and Greeks are near zero and a RELATIVE agreement
tolerance is hardest to satisfy — short tau, deep OTM, low curvature. That is a tolerance
artefact rather than difficulty selection. But it has a direct consequence:

**The hedge box is masked at 1.30x the outside rate (0.1233 vs 0.0948)**, and inside the
box individual maturity nodes are nearly absent: tau=0.0932 is 98.8% masked, tau=0.1154
is 84.8%. The surviving in-box accuracy claim is scored on a region whose label support
is thinnest exactly where the hedge terminates. This belongs with the claim, not in a
footnote.

## R4 — Occupancy-weighted derivative error
`occupancy_metric_{per_seed,agg}.csv`, `metric_comparison.csv`
· `reanalysis_occupancy_metric.py`, `reanalysis_metric_comparison.py`

The principled estimand behind the box: E_{mu^pi}[(Dhat C - D C)^2], derivative error
integrated against the measure of states the controller actually visits. Paths regenerated
from the frozen seed-keyed streams; states are exactly those the engine forms positions at;
reference is the same theta_train CF oracle the engine hedges against. 43,000 states per
seed, 5 seeds, 7 arms.

Spearman rho against realised zero-cost misspecified CVaR95:

| derivative | global grid | hedge box | **occupancy** |
|---|---|---|---|
| delta | 0.386 | 0.371 | **0.543** |
| gamma | 0.468 | 0.507 | **0.707** |
| vega | 0.714 | 0.775 | **0.800** |

Pairwise concordance:

| derivative | global grid | hedge box | **occupancy** |
|---|---|---|---|
| delta | 0.619 | 0.648 | **0.724** |
| gamma | 0.667 | 0.700 | **0.771** |
| vega | 0.776 | 0.800 | **0.829** |

**Occupancy dominates on every derivative and both statistics.** The box, by contrast, is
barely better than the global grid and for delta is not better at all. The methodological
recommendation should therefore be the occupancy-weighted estimand, with the box named as
its crude proxy — not the box on its own.

**Confound, stated.** The three metrics differ in TWO ways, not one: weighting (uniform
grid nodes vs mu^pi) AND parameters (OOD anchors vs the confirmatory cell's perturbed
coefficients). "Occupancy beats box" is a claim about the whole estimand, not about
reweighting alone. The global-vs-box comparison IS clean.

## R5 — Measured label cost per oracle leg
`label_cost.csv` · `reanalysis_label_cost.py`

64 (S, K, tau) points, one machine, single process; ratios transfer, seconds do not.

| leg | s/point | x CF |
|---|---|---|
| A: CF, analytic AD on the integrand | 0.000923 | 1.0 |
| B: FD stencils on COS prices | 0.007754 | 8.4 |
| C: MC pathwise / LR (20k paths) | 0.001734 | 1.9 (~19x at the production 200k) |
| D: Craig–Sneyd ADI PDE solve | 0.634687 | **687.6** |

Greeks are essentially free when taken analytically on the CF integrand — one
Gauss-Legendre integration serves the price and every derivative. Pair with the
information-matching plateau (m = 3 in fd-equivalent units): derivative labels are worth
roughly 3x their price-point budget and cost ~1x a price call from leg A, but 688x from
the fourth leg the Feller band requires.

## T1 — Regret decomposition: approximation vs control design
`regret_decomposition.csv` · `reanalysis_regret_decomposition.py`

Zero cost, misspecified, from the frozen MV comparator run:

| term | value |
|---|---|
| policy-design regret (exact delta - MV) | 0.4120 |
| rung3 approximation regret (rung3 - exact delta) | 0.0141 |
| ratio | **29.2x** |
| rung3 approximation share of total regret vs MV | **3.3%** |

**The plan's "roughly 37x" does not reproduce from these artifacts; the correct figure is
29.2x.** The stronger and equally true statement is the share: 96.7% of rung3's remaining
gap to the best available controller is policy design, not approximation. For
`standard_pinn` the split reverses (70.8% approximation), which is what being the damaged
arm looks like.
