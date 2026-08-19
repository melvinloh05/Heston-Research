# Mechanism adjudication memo

Descriptive only. Channel language is limited to the pre-registered readings (CLAUDE.md 'Mechanism'); nothing below chooses a channel beyond `_mechanism_reading`.

## Pre-registered verdicts

| threshold | cell | statistic | CI | verdict |
|---|---|---|---|---|
| confirmatory_cell | combined m=1.0 tc=0.01 | 0.0002235 | [-0.02421, 0.02423] | **fail** |
| order_attribution | rung2-vs-rung1 combined m=1.0 tc=0.01 | 0.05736 | [0.05274, 0.0621] | **fail** |
| dose_response | combined m=1.0 tc=0.01 | 0.02857 | [-0.08571, 0.7714] | **flat** |
| ood_greek_thresholds | rung3 near_feller/strong_neg_corr full-grid | 0.8635 | [0.8593, 0.8678] | **fail** |
| ood_greek_thresholds_rung2_secondary | rung2 near_feller/strong_neg_corr full-grid | 0.8697 | [0.8492, 0.8901] | **fail** |
| sakuma_null_consistency | combined m=0.0 tc=0.0 | 0.2722 | [-0.6648, -0.6311] | **flag** |
| mechanism_adjudication | combined m=1.0 TC-sweep | -0.9624 | [-0.9804, -0.9443] | **channel_i** |
| goldilocks_bates | bates severity sweep tc=0.01 | 0.3105 | [0.2829, 0.3386] | **no_decisive_regime** |

## rung3 - standard_pinn gap over the TC sweep (misspec m=1.0)

loss units; negative = rung3 better (lower CVaR95).

| tc | gap (CVaR diff) | pooled 95% CI | rel |
|---|---|---|---|
| 0.0 | -0.9624 | [-0.9804, -0.9443] | 0.3152 |
| 0.01 | -0.001086 | [-0.02421, 0.02423] | 0.0002235 |
| 0.02 | 1.319 | [1.295, 1.345] | -0.1912 |

### in-model (m=0.0) gap over the TC sweep

| tc | gap (CVaR diff) | pooled 95% CI | rel |
|---|---|---|---|
| 0.0 | -0.648 | [-0.6648, -0.6311] | 0.2722 |
| 0.01 | 0.3104 | [0.2915, 0.3301] | -0.06898 |
| 0.02 | 1.211 | [1.187, 1.234] | -0.1724 |

## 2x2: {in-model, misspec} x {tc=0, tc>0}

The kappa of the contract's 2x2 (the in-model x cost corner) IS the COST parameter — this table is the cost axis of the adjudication.

| | tc=0 | tc>0 |
|---|---|---|
| misspec (m=1.0) | -0.9624 | -0.001086 |
| in-model (m=0.0) | -0.648 | 0.3104 |

## Excess-turnover statistic T_ex

t_ex(rung3) - t_ex(standard_pinn) = 0.6886 (seed 95% CI [0.6618, 0.7154], n=10 seeds). T_ex CI EXCLUDES 0 on the INCREASING side (turnover moved, but UP: the baseline under-trades and supervision restores oracle-level turnover) -> cost channel cannot be credited, and not because turnover was unmoved.

d(gap)/d(tc) slope = 114.1.

**Pre-registered reading: `channel_i`** (present_i=True, present_ii=False).
  - channel_i  = robustness: gap present at tc=0 (CI excludes 0).
  - channel_ii = cost: gap widens with tc AND T_ex reduced (CI excludes 0).
  - decomposition = both; no_channel = neither (an ADJUDICATED reading, distinct from the universal `null` = not evaluated).

## Goldilocks (Bates severity sweep)

| lambda_j | sigma_j | gap | pooled 95% CI | decisive |
|---|---|---|---|---|
| 0.0 | 0.05 | 0.3105 | [0.2829, 0.3386] | no |
| 0.1 | 0.05 | 0.3674 | [0.3361, 0.3997] | no |
| 0.1 | 0.1 | 0.3714 | [0.3364, 0.4081] | no |
| 0.1 | 0.15 | 0.3691 | [0.3289, 0.4098] | no |
| 0.25 | 0.05 | 0.4599 | [0.4229, 0.4979] | no |
| 0.25 | 0.1 | 0.4641 | [0.4193, 0.5101] | no |
| 0.25 | 0.15 | 0.4344 | [0.3783, 0.4905] | no |
| 0.5 | 0.05 | 0.6186 | [0.5755, 0.6633] | no |
| 0.5 | 0.1 | 0.6608 | [0.6032, 0.7173] | no |
| 0.5 | 0.15 | 0.6691 | [0.5944, 0.7412] | no |

## Three candidate stories

Per the roadmap workflow note — described, not adjudicated.

**What happened.** The pre-registered reading is `channel_i`. The tc=0 gap is -0.9624 (CI [-0.9804, -0.9443]); at the top tc tier it is 1.319. T_ex(rung3)-T_ex(std) = 0.6886. The dose-response verdict is `flat` (Spearman statistic 0.02857).

**What could be an artifact.** A gap that appears only once TC is charged, with T_ex covering 0, could be turnover-accounting noise rather than a robustness effect; a large in-model gap would suggest the misspecification contrast is contaminated; a monotone TC slope with tiny per-seed CIs but few seeds may be under-powered. Check the seed count and the in-model 2x2 corner before reading the cost axis.

**What is surprising.** Note here any sign flip between the per-seed and pooled-stratified readings, any regime where the Bates gap is decisive while the combined-perturbation gap is not, and whether the T_ex direction agrees with the sign of the TC-slope. These are logged for human adjudication; the memo does not resolve them.
