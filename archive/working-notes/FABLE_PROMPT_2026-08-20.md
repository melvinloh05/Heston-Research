You are adjudicating a pre-registered empirical study. Your job is to PROSECUTE, not
validate. Assume the author is motivated to find a positive story and has already
fooled himself once. Where you agree with him, say so in one line and move on; spend
your effort on where he is wrong.

Do not ask clarifying questions. Do not recompute anything — all numbers below are
given as measured, and the pipeline that produced them was audited across twelve
components (settlement, CVaR sign, CRN discipline, premium and terminal-liability
conventions, provider state symmetry, PDE residual, comparator parameter sourcing,
label construction, and the Greek code path shared by grid and hedge evaluation) with
no inverting defect found. Do not propose new experiments unless you believe the paper
is unpublishable without one, in which case say so explicitly and bound the compute.

=== THE STUDY ===

A parametric physics-informed neural network (PINN) is trained to price European
calls under the Heston stochastic-volatility model over an 8-D input
(S, K, tau, kappa, theta, xi, rho, v0). "Sobolev supervision" means additionally
supervising the derivatives Delta = dC/dS, Gamma = d2C/dS2, Vega = dC/dv against a
4-leg numerical oracle (characteristic function / FD-on-COS / Monte Carlo /
Craig-Sneyd ADI, cross-validated to 1e-3 relative).

Registered causal question: does supervising GAMMA improve DELTA-ONLY hedging PnL
under misspecified dynamics (hedge with Heston, but the true data-generating
process is Bates or Merton jump-diffusion), and through which channel — reduced
transaction cost/turnover, or improved robustness to model misspecification?

Supervision ladder (arms): standard_pinn (price + PDE residual) -> rung1 (+Delta)
-> rung2 (+Delta+Gamma) -> rung3 (+Delta+Gamma+Vega). The registered claim lives on
the rung1->rung2 gap. Primary metric: CVaR95 of hedging loss, paired bootstrap over
common random numbers, 10 seeds at the confirmatory cell. Registered transaction-cost
tiers {0%, 1%, 2%}.

STATISTICAL CONVENTION. All CIs below are the registered pooled-stratified paired
bootstrap over CRN paths, reported in ABSOLUTE CVaR95 units, not percentage points.
Every contrast quoted is additionally seed-robust across 10 seeds unless stated.
A pre-registered economic negligibility floor exists: acceptance_thresholds.
sakuma_null_rel_tol = 0.02, i.e. contrasts below 2% relative are declared
economically inert regardless of significance.

=== REGISTERED OUTCOMES (adjudicated in a prior memo; do not relitigate 1, 2, 4, 5) ===

1. Confirmatory cell (1% TC): FAILS. No improvement.
2. Order attribution (rung1->rung2, the registered claim): the CI excludes zero on the
   HARMFUL side (+0.0574 [+0.0527, +0.0621]). Adding Gamma supervision worsens
   delta-only hedging relative to Delta supervision alone — but the effect is 1.2%
   relative, BELOW the pre-registered 2% negligibility floor. The honest reading is
   "statistically real, economically negligible, harmful in sign", not "significantly
   worsens".
4. The registered 1% TC tier is ~200x the transaction rate calibrated in the
   reference paper for hedging in the UNDERLYING (0.005%), and ~2x their own stress
   tier. This is a design error, discovered after freezing. It is NOT being removed
   from the paper.
5. Greek accuracy (out-of-distribution RMSE reduction for Gamma and Vega at price
   parity) PASSES.

OPEN FOR RE-ADJUDICATION (was outcome 3; the new results below bear directly on it):
   At 0% TC the full-ladder effect is +31.50% (rung3 vs standard_pinn), decomposing
   as Delta 91.9% / Gamma 6.5% / Vega 1.7%. NOTE: that decomposition is itself
   measured at lambda_pde = 0.01 only. If the gap is mostly baseline damage (see A),
   those are shares OF THE DAMAGE, and the Gamma rung's share at a defensible
   lambda_pde is unmeasured.

PRIOR ADJUDICATION CONCLUDED: the gamma-hedging claim is dead; a Greek-accuracy claim
and a "tracking channel" finding survive.

   THE "TRACKING CHANNEL" FINDING, stated so you can rule on it: the operative
   mechanism is neither registered channel. It is delta-approximation / tracking
   quality. Evidence: ~86% of the 0%-TC effect is present in-model (in-model +27.2%
   vs misspecified +31.5%); the baseline under-trades the oracle by 38% (turnover
   0.0248 vs 0.0398, excess turnover -0.696 against rung3's -0.007); and its delta at
   hedge states is a flattened sigmoid, regression slope 0.637 on the oracle delta,
   biased +0.101 OTM / -0.029 ATM / -0.107 ITM. The registered 2x2 (cost vs
   robustness) has no bin for this.

=== FIVE NEW RESULTS THAT POSTDATE THAT ADJUDICATION ===

(A) LAMBDA_PDE SENSITIVITY CURVE. The weight on the PDE residual term, lambda_pde,
was selected on validation and registered at 0.01, sourced from the standard_pinn
arm. The headline as a function of lambda_pde (rung3 vs standard_pinn, misspecified,
10 seeds, every CI excluding zero, every rung seed-robust):

    lambda_pde   effect @ 0% TC   paired CI (abs CVaR)   effect @ 1% TC
    0.0             +2.86%        [-0.0672, -0.0588]        -3.14%
    1e-4            +7.37%        [-0.1726, -0.1598]        -5.61%
    3e-4            +9.96%        [-0.2392, -0.2243]        -5.35%
    1e-3           +15.36%        [-0.3874, -0.3685]        -5.28%
    3e-3           +21.98%        [-0.6052, -0.5800]        -3.78%
    0.01 (REG)     +31.50%        [-0.9804, -0.9443]        +0.01%

Smooth and monotone, but CONVEX in log(lambda) — the slope accelerates: 5.4, 10.3,
13.9, 18.2 percentage points per decade across successive intervals. The registered
value sits where the curve is steepest. Not a knife-edge; no interior optimum; no
plateau. At lambda_pde = 0 the arms are bit-identical to two other registered arms
(a plain feedforward net and a "Sobolev-sans-PDE" arm). Replicated by two
independently written drivers, agreeing to 0.00e+00 at three overlapping rungs.

    DECOMPOSITION BY ARM (this is the measurement that identifies what moves).
    rung3 CVaR95 across lambda_pde = 0 / 1e-4 / 3e-4 / 1e-3 / 3e-3 / 0.01:
        2.0758  2.0779  2.0800  2.0810  2.0905  2.0904
    standard_pinn over the same range:
        2.1370  2.2434  2.3101  2.4587  2.6794  3.0518
    Oracle at this cell: 2.0794. So rung3 drifts 0.7% across a 100x change in
    lambda_pde and is pinned at the oracle throughout; the baseline degrades 43%.
    ONLY THE BASELINE MOVES.

    AT THE REGISTERED 1% TIER the contrast is negative at EVERY lambda_pde except the
    registered one, where it is +0.01%. (0.01, 1%) is therefore the single most
    favourable cell in the entire explored (lambda_pde x cost) grid. This is
    coincidence, not tuning: lambda_pde was fixed by a validation-blind rule before
    any hedging ran, and the cost tiers were registered before that.

(B) THE REGISTERED LAMBDA_PDE IS WRONG BY THE CONTRACT'S OWN SELECTION RULE — and
separately, the pre-registered robustness row is discharged and is material.

    (i) Re-scoring the contract's own criterion (train._val_greek_score: mean
    normalised validation RMSE over price/Delta/Gamma/Vega, unmodified) on a fine
    grid, mean over 10 seeds:
        lambda_pde:   0        1e-4      3e-4      1e-3      3e-3      0.01
        score:     0.13090   0.08200   0.08614   0.10385   0.12817   0.17733
    The registered 0.01 is the WORST of the six points, more than 2x the optimum.
    The recomputation reproduces the registered scores_table_pde EXACTLY at both
    overlapping points (0.17565 at 0, 0.16957 at 0.01, matching lambdas_selected.yaml
    to five decimals), so it is the same criterion, not a proxy. The optimum is flat
    between 1e-4 and 3e-4 and is not sharply located; what is sharp is that the
    registered point lies far outside that basin. The original grid was
    {0, 0.01, 0.1, 1.0} — four points straddling a minimum two orders of magnitude
    below the smallest non-zero candidate. This is a grid-resolution failure, not a
    rule failure.

    (ii) The contract pre-registered that the confirmatory contrast be re-run with
    lambda_pde sourced from rung3 instead of standard_pinn. That value is 0.0.
    Headline moves +31.50% -> +2.86% and reverses at 1% TC. The contract's own
    wording admitted this could be "shown not to be" immaterial. It is material.

(C) MINIMUM-VARIANCE DELTA COMPARATOR. Delta_MV = dC/dS + (rho*xi/S)*dC/dv
(local risk minimization). Delta_MV BEATS the exact Heston oracle delta by 19.79%
(misspecified) and 10.31% (in-model). The study's headline "gap closed" statistic is
98.59% when measured against the oracle, but only 69.80% when measured against the
hedge that actually wins.

(D) REGION MISMATCH. Greek accuracy was registered on the full parameter grid, but
hedging occurs in a narrow near-the-money box. On the near_feller anchor, delta RMSE
full-grid vs hedge-box: standard_pinn 0.05370 / 0.09373; feedforward 0.08806 /
0.04090. THE RANKING REVERSES between the region where accuracy is measured and the
region where hedging happens. It reverses in all five seeds and at both OOD anchors
(strong_neg_corr: standard_pinn 0.04242 / 0.06891, feedforward 0.08153 / 0.02270).
standard_pinn is the only arm that degrades moving into the box; every other arm
improves there. Proposed mechanism (hypothesis, NOT measured): the PDE residual
contains (1/2) v S^2 d2C/dS2, dominated near the payoff kink at S=K, so the residual
weight degrades the network exactly in the hedged region. What IS measured: the
residual normaliser loss_scale_pde = mean((r*price)^2) has rms 0.527 against the
residual's dominant term rms 4.674, inflating the effective weight ~79x, and at the
registered lambda the PDE term is 49% of standard_pinn's total loss.

    The Gamma-accuracy claim SURVIVES the region check: rung3 Gamma reduction is
    86-89% on the full grid and 91-93% in the hedge box (near_feller 89.4 -> 93.0,
    strong_neg_corr 86.4 -> 91.0). It improves in the box.

(E) CONTEXT. The "in-model" condition is NOT in-distribution: all five named
evaluation regimes, including the baseline, were excised from training by 10%
relative-radius balls, so the in-model cell tests generalisation of Greek fields to
unseen parameters, not coping with wrong dynamics. And under Bates the effect decays
monotonically with jump severity (+27.65% at the mildest cell -> -4.21% at the
severest, (lambda_j, sigma_j) = (0.5, 0.15)), which cuts against a
robustness-to-misspecification reading.

=== WHAT I WANT FROM YOU ===

1. THE CENTRAL QUESTION. Taking (A)-(E) together: is the surviving effect a property
   of Sobolev supervision, or a property of how badly the PDE residual damages the
   baseline near the payoff kink? State which, commit to it, and give the strongest
   counter-argument to your own answer.

2. KILL LIST. For each of the three claims the prior memo left standing — Greek
   accuracy, the "tracking channel" finding, and the 0%-TC ladder effect — rule
   SURVIVES / SURVIVES ONLY IF QUALIFIED / DEAD, with the specific reason. (D)
   bears on the first; (A)+(B) on the third.

3. THE ONE SENTENCE. Write the paper's central claim as a single sentence that no
   hostile reviewer with all of the above in hand could call overstated. Then write
   the title that goes with it.

4. FRAMING. Should this be (i) a positive methods paper with heavy caveats, (ii) a
   negative-results / failed-replication paper, (iii) a paper about the measurement
   pathology itself — that PDE-residual weighting and evaluation-region choice
   jointly determine the apparent benefit of derivative supervision — or (iv) not
   submittable in any of these forms on the current evidence. Pick one. Argue
   against the others.

5. STRONGEST ATTACK. State the single most damaging question a reviewer will ask,
   and the honest answer. If the honest answer sinks the paper, say that.

6. WHAT NOT TO DO. Name any framing in the above that you think is the author
   rationalizing, and any number he should stop quoting.

Be concrete and terse. No preamble, no summary of what I told you. Lead with your
answer to (1).

LOGISTICS, stated last so it does not colour the judgment: the target is the NeurIPS
2026 STODY workshop, deadline 29 Aug 2026. The deadline constrains SCOPE, not
honesty. If the answer is "do not submit this", say so.
