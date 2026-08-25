# Results-pipeline bug audit — 2026-08-20

**Question put:** the results invert essentially every directional prediction of the
pre-results draft. Is there a bug in the collection code that produced that inversion?

**Verdict: no bug found that inverts the results.** The hedging engine, providers,
settlement, CRN discipline, CVaR, label construction, Greek code paths and the MV-delta
comparator are all correct, and I verified each against the artifacts rather than by
reading alone. **But the audit did find a measurement-design defect that explains the
inversion**, is seed-robust across all five seeds, and is arguably more consequential
than a bug would have been, because it is invisible in every table the paper currently
prints.

Everything below was recomputed from the frozen artifacts and checkpoints.

---

## 1. What is clean (audited, no defect)

| # | Component | Check performed | Result |
|---|---|---|---|
| 1 | `_settle_core` / `settle_delta` | read; self-financing recursion, cost = `tc·px·|Δpos|` on traded notional, dividend accrual O(dt), internal `_assert_decomposition` | correct |
| 2 | `cvar` | mean of worst `ceil(0.05n)` values of `-PnL` | correct sign; higher = worse |
| 3 | CRN discipline | `_eval_methods`: one `(S, v, times)` per cell, positions computed once per arm and **independent of tc**, all arms settled on the same paths | correct |
| 4 | Premium convention | `prem_override` = θ_train oracle price, computed once per cell, broadcast to **every** arm incl. smoothed variants; `premium_convention_ok` on every row | correct |
| 5 | Terminal liability | computed once per cell from the **true DGP** (`heston_bates_terminal_mark` / `merton_terminal_mark`), passed identically to every arm; engine raises rather than settling at expiry | correct |
| 6 | Provider state | `HestonCFProvider` and `PINNProvider` both receive `(S_t, max(v_t, 1e-6), τ = T−t)`; PINN pins (κ,θ,ξ,ρ) to θ_train and feeds pathwise v into the `v0` slot — the same function the oracle evaluates | consistent; no state asymmetry |
| 7 | PDE residual | `pde_residual` matches the contract's Heston PDE term for term, signs included (`−V_τ + ½vS²V_SS + ρξvS V_Sv + ½ξ²v V_vv + (r−q)S V_S + κ(θ−v)V_v − rV`) | correct |
| 8 | PDE gating | `if cfg.use_pde and cfg.lambda_pde != 0.0` — `sans_pde` / `feedforward` carry `lambda_pde=1.0` in their checkpoint cfg but `use_pde=False`, so the term is skipped | inert; **cosmetic hazard only** (see §5) |
| 9 | λ_pde = 0 arm identity | `results/grid_robustness/standard_pinn/s42` vs `results/grid/feedforward/s42`: **bit-identical weights**; same for rung3(λ=0) vs `sobolev_sans_pde` | confirmed at weight level |
| 10 | MV-delta comparator | `MVDeltaProvider` wraps a `HestonCFProvider` built from **θ_train** params and uses θ_train ρ, ξ — no true-DGP leakage; price/gamma pass through; only the hedge ratio changes | sound |
| 11 | Dose-response labels | `make_gamma_labels`: multiplicative `γ(1+N(0,σ))` on gamma only, δ/ν always true; `shuffled` = permutation; `bs_gamma` = BS gamma at matched IV | matches contract |
| 12 | Greek code paths | `eval_greeks.py:147` and `PINNProvider.evaluate` both call `model.greeks_eval` | **identical code path** — grid and hedge Greeks are computed the same way |

Consequence: the observed numbers are what the code was asked to compute. The inversion
is not an artifact of the measurement apparatus.

---

## 2. The defect: the two headline metrics are measured in disjoint regions, and the
## arms rank **oppositely** in them

Delta RMSE against the frozen consensus, **baseline anchor**, identical metric, seed 42:

| region | standard_pinn (λ=0.01) | feedforward (λ=0) | rung1 | rung3 | sans_pde |
|---|---|---|---|---|---|
| **FULL grid** (S∈[50,150], K∈[60,140], τ∈[0.04,1.0]) — where the OOD Greek table lives | **0.0414** | 0.1040 | 0.0091 | 0.0028 | 0.0125 |
| **HEDGING slice** (K≈100, τ∈[0.08,0.25], S∈[65,125]) — where the hedging headline lives | **0.0706** | 0.0269 | 0.0084 | 0.0021 | 0.0030 |

On the full grid the residual-only baseline is **2.5× better** than a price-only net.
In the hedging slice it is **2.6× worse**. The ordering reverses.

`standard_pinn` is the **only arm that degrades** moving from the full grid into the
hedging slice (0.0414 → 0.0706). Every other arm improves there
(rung3 0.0028 → 0.0021, sans_pde 0.0125 → 0.0030, feedforward 0.1040 → 0.0269).

**Seed-robust — all five seeds, no exceptions:**

| seed | std_pinn FULL | ffwd FULL | std_pinn SLICE | ffwd SLICE | reverses? |
|---|---|---|---|---|---|
| 42 | 0.04139 | 0.10403 | 0.07062 | 0.02693 | yes |
| 43 | 0.04546 | 0.08391 | 0.07802 | 0.02199 | yes |
| 44 | 0.04856 | 0.05251 | 0.08150 | 0.01996 | yes |
| 45 | 0.04598 | 0.05828 | 0.08035 | 0.02371 | yes |
| 46 | 0.04923 | 0.10975 | 0.08363 | 0.01498 | yes |

The same reversal holds at `near_feller` (0.0822 vs 0.0436) and `strong_neg_corr`
(0.0667 vs 0.0300), so it is a property of the **(S,K,τ) region**, not of the anchor.

### 2.1 Measured at actual hedging path states, the arms order as their PnL does

Evaluated on the confirmatory paths themselves (combined perturbation, m=1.0, seed 42,
2000 paths × 43 steps), against the oracle delta on the same states:

| arm | delta RMSE | bias | regression slope on oracle Δ | total traded | t_ex |
|---|---|---|---|---|---|
| oracle | — | — | 1.000 | 2.841 | 0 |
| rung3 | 0.0038 | −0.0001 | 0.996 | 2.838 | −0.003 |
| rung2 | 0.0057 | −0.0018 | 0.997 | 2.833 | −0.008 |
| sans_pde | 0.0056 | +0.0018 | 0.996 | 2.843 | +0.002 |
| rung1 | 0.0202 | −0.0037 | 0.965 | 2.780 | −0.061 |
| bs_gamma | 0.0280 | **−0.0254** | 0.980 | 2.761 | −0.080 |
| feedforward | 0.0484 | −0.0225 | 0.880 | 2.643 | −0.198 |
| standard_pinn | **0.0989** | −0.0357 | **0.637** | 2.186 | −0.655 |
| shuffled | 0.1458 | −0.0059 | **0.404** | 1.872 | −0.969 |

The baseline's delta is a **flattened sigmoid** — OTM +0.101, ATM −0.029, ITM −0.107 —
i.e. a slope deficit, not a level bias. That is precisely the "38% under-trading"
reported as a mechanism finding; it is a measured property of the delta surface.

**Measured in the same region, accuracy predicts hedging.** Slice/path delta RMSE orders
rung3 < sans_pde < rung2 < rung1 < feedforward < standard_pinn < shuffled, and zero-cost
CVaR₉₅ orders 2.090 ≈ 2.090 < 2.107 < 2.169 ≈ 2.153 < 3.052 < 4.869. The only departures
are `feedforward` and `bs_gamma`, both of which carry a **negative hedge-ratio bias**
(−0.023, −0.025) in the minimum-variance direction — the MV effect, not an RMSE effect.

**This directly contradicts the draft's framing that the hedging and accuracy axes move
independently.** They rank oppositely only because they are measured in disjoint regions.

---

## 3. Contributing cause 1 — λ_pde was tuned where the residual helps and applied where
## it hurts

`train.py::_run_select_lambdas` scores λ_pde on `standard_pinn` via
`_val_greek_score`: mean normalized validation RMSE of price/δ/γ/ν over the **hypercube
validation split crossed with the full (S,K,τ) grid**.

- Validation-region delta RMSE: standard_pinn (λ=0.01) **0.0418** vs feedforward (λ=0)
  **0.0607** → λ=0.01 wins by 31%.
- Hedge-state delta RMSE: **0.0989** vs **0.0484** → λ=0.01 loses by 104%.

The selection was correct *given its objective* and blind to the region that decides the
headline. The named anchors are excised from training **and validation** by construction,
so no validation point can ever report the hedging region's behaviour.

**The λ_pde grid also selected at its own lower boundary.** Candidates
`[0.0, 0.01, 0.1, 1.0]`; validation scores `0.1757, 0.1697, 0.3332, 0.3724`. The winner
(0.01) is the smallest nonzero candidate, beats λ=0 by only **3.4%**, and every larger
value degrades sharply. Nothing between 0 and 0.01 was ever tried — and that interval is
where the 43% hedging swing lives. This is the same pathology already caught once in the
no-trade-band grid (registered grid selected its own boundary at 0.04; extended grid found
an interior optimum at 0.08).

---

## 4. Contributing cause 2 — the PDE normalizer inflates the effective λ_pde ~79×

`_freeze_loss_scales` sets `loss_scale_pde = mean((r·price)²)`, described as "the rV
discount term of the residual … a model-free magnitude reference".

Measured on the frozen labels:

- `rms(r · price)` = **0.527** (the normalizer)
- `rms(0.5 · v · S² · Γ_true)` = **4.674** (the residual's own dominant term)
- → the normalizer is **8.9× too small in rms, 79× in squared units**

So a nominal λ_pde carries roughly **79× the weight** it would under a scale-matched
normalization, and the candidate grid `{0, 0.01, 0.1, 1.0}` explores effective weights
of about `{0, 0.79, 7.9, 79}` in dominant-term units — every nonzero rung is heavy.

Measured consequence at the as-trained λ_pde = 0.01 (checkpoint's own cfg, seed 42):

| arm | price term | PDE term | PDE share of total |
|---|---|---|---|
| standard_pinn | 8.64e−5 | 8.23e−5 | **49%** |
| rung1 | 8.45e−6 | 1.25e−4 | 67% |
| rung3 | 1.77e−6 | 4.20e−4 | 72% |
| feedforward | 3.05e−6 | — | — |

**In-sample normalized price MSE: standard_pinn 8.64e−5 vs feedforward 3.05e−6 — the
residual makes the price fit 28× worse in-sample.** (rung3 reaches 1.77e−6, better than
feedforward, so derivative labels *improve* price fit even while carrying the residual.)

And the residual is stiffest exactly where the headline is measured: the dominant term's
rms is **4.82 in the hedging slice vs 3.33 on the full grid (1.45×)**. The over-smoothing
pressure is concentrated in the ATM/short-τ box.

---

## 5. Contributing cause 3 — label coverage thins along the hedge's life

The entire training set is 512 parameter points × **64 shared (S,K,τ) triples**
(`n_skt: 64`, contract AM4-1). Distance from the hedged contract to the nearest triple,
in range-normalized (S,K,τ):

| hedge state | nearest triple | distance | triples within 0.15 |
|---|---|---|---|
| inception (S=100, K=100, τ=0.25) | S=100.4, K=100.1, τ=0.2598 | **0.011** | 1 |
| mid-life (τ=0.17) | same | 0.094 | 1 |
| terminal (τ=0.08) | S=85.1, K=94.2, τ=0.1111 | **0.169** | **0** |

The hedge **starts on a labelled point and walks out of the labelled region as τ decays**.
Zero triples fall inside the box (K=100±5, τ∈[0.08,0.25], S∈[65,125]).

PDE collocation is *not* subject to this — `sample_pde_points` draws (S,K,τ) uniformly
over the full grid ranges each epoch — but it *is* subject to the same anchor-ball
excision in parameter space. So in the hedging region the residual-only arm has physics
signal at other parameters and no label signal at all, which is consistent with it being
the one arm that degrades there.

---

## 6. What this does to the paper's claims

1. **"The hedging and accuracy axes move independently" — not established.** They rank
   oppositely because they are measured in disjoint (S,K,τ) regions. Restricted to one
   region, accuracy predicts hedging for every arm, with residual departures attributable
   to hedge-ratio bias (the MV effect). This is the single most important correction.
2. **The +31.5% zero-cost headline is region- and hyperparameter-contingent.** It measures
   the residual-only baseline's slice-specific over-smoothing at a λ_pde selected on a
   different region, at the lower boundary of its grid, under a normalizer that inflates
   its effective weight ~79×. It is not a general property of Sobolev supervision.
3. **The mechanism finding (tracking/approximation channel) survives** — it is now
   supported at the level of the delta surface itself (slope 0.637 vs 1.0), not only via
   PnL. Its *magnitude* is confounded by items 1–2.
4. **The OOD Greek result survives and is not affected** by any of this: it is measured on
   the full grid, where it is large, monotone in the ladder, and seed-stable.
5. **The bs_gamma / MV-delta finding survives** — the comparator is clean and the bias
   direction is visible directly in the hedge-state delta bias column.

---

## 7. Recommended actions, ranked

1. **Re-run the OOD Greek metric restricted to the hedging slice and report both.**
   *No GPU, no retraining* — the checkpoints and anchor grids are on disk; this is the
   analysis I ran above. It converts "the axes are independent" into "the axes agree when
   measured in the same place, and here is where they don't."
2. **Re-tune λ_pde on a grid extending below 0.01** (e.g. `{1e-4, 1e-3, 3e-3, 0.01}`),
   and/or with a scale-matched PDE normalizer; report the headline as a function of λ_pde.
   *Needs GPU dispatch — propose and wait for approval.* This decides whether +31.5%
   survives, and it is the number a referee will attack first.
3. **Declare the (S,K,τ) coverage gap** as a scope condition, or add triples covering the
   hedged contract's life. The current draft does not mention that the hedged instrument
   leaves the labelled region as τ decays.
4. **Report the effective-λ inflation and the boundary selection** in the methods section.
   Both are true, both are checkable, and disclosing them is much stronger than having a
   referee derive them.
5. **Cosmetic:** `sans_pde` and `feedforward` checkpoints carry `lambda_pde=1.0` in their
   saved cfg while `use_pde=False`. Functionally inert, but anyone inspecting checkpoints
   will misread it. Normalize to 0.0 on the next write, or note it in the artifact README.

---

## 8. Honest statement of what this audit does *not* establish

- It does **not** show the results are wrong. Every number in the record reproduces.
- It does **not** show the +31.5% would vanish at a better λ_pde — that requires the
  retrain in item 2. It shows only that the number was never optimized against, or even
  measured in, the region that decides it.
- It does **not** rescue the registered confirmatory claim, which fails at 1% TC for
  reasons (cost inversion) independent of everything above.
- The direction of the λ_pde effect on hedging is known only at the two endpoints
  (λ=0 → 2.137, λ=0.01 → 3.052). The interval between them is unmeasured, and the honest
  reading is that the baseline's quality — and therefore the headline — is **unresolved**
  within it.

---

## 9. RESULTS of the five actions (completed 2026-08-20)

All five actions are complete. Test suite: **282 passed, 0 failed, three consecutive
full runs** (was 279/3-failed on arrival). No registered artifact was modified; every
new output lives under a new path.

### 9.1 Action 2 — the λ_pde sensitivity sweep: the headline is a readout of λ_pde

80 fits (4 new λ rungs × 2 arms × 10 seeds), 154 min, 0 failures, protocol identical to
the production grid except λ_pde (verified bit-identical across Python 3.9/3.12 first, so
the local runs are comparable to the dispatched grid). Confirmatory cell, misspecified,
tc = 0, 10 seeds, pooled statistic — every rung seed-robust:

| λ_pde | standard_pinn CVaR₉₅ | rung 3 CVaR₉₅ | gap | **rel, seed-mean** | rel, pooled-paths |
|---|---|---|---|---|---|
| 0 | 2.1370 | 2.0758 | 0.061 | **+2.86%** | +2.95% |
| 1e-4 | 2.2434 | 2.0779 | 0.165 | **+7.37%** | +7.41% |
| 3e-4 | 2.3101 | 2.0800 | 0.230 | **+9.96%** | +10.03% |
| 1e-3 | 2.4587 | 2.0810 | 0.378 | **+15.36%** | +15.36% |
| 3e-3 | 2.6794 | 2.0905 | 0.589 | **+21.98%** | +22.08% |
| **1e-2 (registered)** | 3.0518 | 2.0904 | 0.961 | **+31.50%** | +31.52% |

*(oracle at this cell: 2.0794)*

*Convention note (corrected 2026-08-20): the CVaR₉₅ levels are means of the per-seed
CVaR; `rel, seed-mean` is computed from those levels, `rel, pooled-paths` from a single
CVaR over all 10 seeds' paths concatenated (the `pooled_rel` column of
`headline_vs_lambda_pde.csv`, and the contract's registered statistic). An earlier draft
of this table mixed the two — CVaR levels from one, rel from the other. The two agree to
≤0.1pp at every rung, so nothing downstream moves; `headline_vs_lambda_pde_merged.csv`
uses the seed-mean convention throughout.*

**The headline is monotone in λ_pde over the whole explored interval, with no interior
optimum and no plateau: +2.9% → +31.5%, a 11.0× range.** The registered value sits at
the top of it.

**The decomposition settles what is moving.** rung 3 is flat — 2.0758 → 2.0904, a 0.7%
drift, pinned at the oracle's 2.0794 at every rung. `standard_pinn` degrades
monotonically, 2.1370 → 3.0518 (+43%). The effect size is therefore **not a measurement
of what supervision achieves** (which is constant, and already at the oracle) but of
**how much the PDE residual damages the unsupervised baseline**. There is no λ_pde at
which the effect settles; you read off whatever the baseline's handicap is.

At the registered 1% tier the picture is worse, not better: the contrast is negative at
every rung below 0.01 (−3.7% to −6.5%) and only reaches ≈0 at the registered value.
**Correcting λ_pde does not rescue the confirmatory claim — it turns a null into a
significant reversal.**

### 9.2 The registered λ_pde is wrong by the contract's own selection rule

The audit predicted the candidate grid selected at its boundary. Re-scoring the
**contract's own criterion** (`train._val_greek_score`: mean normalised validation RMSE
over price/Δ/Γ/ν) on the fine grid:

| λ_pde | selection score, seed 42 | mean over 10 seeds |
|---|---|---|
| 0 | 0.17565 | 0.13090 |
| **1e-4** | **0.08476** | **0.08200 ← optimum** |
| 3e-4 | 0.08365 | 0.08614 |
| 1e-3 | 0.10356 | 0.10385 |
| 3e-3 | 0.13005 | 0.12817 |
| 1e-2 (registered) | 0.16957 | 0.17733 |

The recomputation **reproduces the registered `scores_table_pde` exactly** at both
overlapping points (0.17565 at λ = 0, 0.16957 at λ = 0.01 — matching
`lambdas_selected.yaml` to five decimals), so this is the same criterion, not a proxy.

**The registered λ_pde = 0.01 is the WORST point on the fine grid, more than 2× the
score of the optimum at 1e-4.** The coarse grid {0, 0.01, 0.1, 1.0} straddled a minimum
near 1e-4 and returned the least-bad of its four points. This is a selection-grid
resolution failure, not a rule failure: the rule works, it was evaluated at four points
that miss the minimum by two orders of magnitude.

At the criterion's actual optimum the headline is **+7.4%**, not +31.5%.

### 9.3 What the paper may now claim

- The zero-cost hedging effect must be reported **as a function of λ_pde**, with the
  registered point marked. A single figure is not identified: the contract's own
  selection rule, applied on a grid fine enough to find its minimum, gives +7.4%.
- The statement that survives is about **rung 3, not about the gap**: rung 3 attains
  oracle-level delta-only CVaR₉₅ (2.076–2.090 vs oracle 2.079) **at every λ_pde**,
  including λ_pde = 0. That is a real, seed-robust, hyperparameter-independent result,
  and it is the one worth leading with.
- The registered confirmatory verdict is unchanged and still **fails** — and at the
  corrected λ_pde it fails harder.

### 9.4 Actions 1, 3, 4, 5 — delivered

- **A1**: `eval_greeks.py` now emits a contract-derived `hedge` slice beside `full` for
  the primary regimes (`hedge_slice_spec`, 5 new tests). Reversal confirmed on the
  registered regimes with the production checkpoints — Δ 0.61×→2.29× (`near_feller`),
  0.52×→3.04× (`strong_neg_corr`), Γ reversing too. `results/eval_greeks_hedgeslice/`.
  A test enforces that the new slice cannot move a registered verdict.
- **A3/A4**: `docs/PAPER_DISCLOSURES.md` — paper-ready text for the coverage-gap scope
  condition and the effective-λ / boundary-selection disclosures. §A4(c) now needs
  updating to quote §9.1–9.2 above rather than promising the sweep.
- **A5**: `SobolevPINN.effective_lambda_pde()` + 2 guard tests + `results/grid/README.md`
  documenting the `lambda_pde = 1.0 / use_pde = False` trap. Frozen checkpoints untouched.

### 9.5 Defects found and fixed while doing the above

1. **`run_info_matching.py` did not parse under Python 3.9** (PEP 701 f-string), which
   made `test_contract_thresholds` and `test_run_info_matching` uncollectable under the
   default `python3`. Hoisted the expression; both interpreters now compile every module.
2. **`HestonCFProvider` is not bitwise deterministic**, contrary to its docstring:
   repeated identical calls differ by ~1.8e-15 (measured on real path states), and it
   persists with BLAS threads pinned to 1, so it is not purely a thread-count effect. Six
   assertions across three tests demanded exact equality on values flowing through it and
   failed nondeterministically. Moved to tolerances ~1e-12 (absolute where the quantity is
   a delta near zero), keeping exact checks where exactness genuinely holds — the RFF
   field at amp = 0, and the frozen path banks, for which I added a direct bit-for-bit
   path assertion that the old test only checked indirectly. **The cause of the residual
   1-ULP nondeterminism is not fully explained**; it is 13 orders below any reported
   effect and moves nothing scientific, but the docstring's claim was false and the suite
   could not serve as a regression gate while it flaked.
3. **My own defect**: a new test consumed the global torch RNG unseeded, shifting state
   for later modules. Wrapped in `torch.random.fork_rng()`.

---

## 10. Independent replication + the merged cost profile (2026-08-20, final)

**Replication.** Melvin ran a parallel λ_pde curve (`lambda_pde_curve.sh`,
`robustness_hedge.py`, `results/grid_lampde_{1e-4,1e-3,3e-3}`) independently of
`lambda_sweep.py`. Three rungs overlap. They agree **exactly**:

| λ_pde | Melvin's run | this audit's run | \|diff\| |
|---|---|---|---|
| 1e-4 | +7.3743% | +7.3743% | 0.00e+00 |
| 1e-3 | +15.3614% | +15.3614% | 0.00e+00 |
| 3e-3 | +21.9786% | +21.9786% | 0.00e+00 |

Checkpoints are bit-identical at the seeds checked, so the two pipelines are the same
computation reached by different routes. The λ_pde finding is replicated, not asserted.

**Merged curve** (`results/lambda_sweep/headline_vs_lambda_pde_merged.csv`), combining
this audit's rungs with the A&T-anchored intermediate tiers from Melvin's runs. Relative
CVaR₉₅ improvement, rung 3 vs standard_pinn, misspecified, 10 seeds:

| λ_pde | tc=0 | tc=5e-5 | tc=0.25% | tc=0.5% | tc=1% | tc=2% |
|---|---|---|---|---|---|---|
| 0 | +2.86% | +2.83% | +1.19% | −0.45% | −3.14% | −5.86% |
| **1e-4 (criterion optimum)** | **+7.37%** | +7.29% | +3.28% | **−0.30%** | **−5.61%** | −9.43% |
| 3e-4 | +9.96% | — | — | — | −5.35% | −10.18% |
| 1e-3 | +15.36% | +15.23% | +9.17% | +3.63% | −5.28% | −13.44% |
| 3e-3 | +21.98% | +21.82% | +14.48% | +7.69% | −3.78% | −16.26% |
| **1e-2 (registered)** | **+31.50%** | +31.32% | +22.70% | **+14.54%** | +0.01% | −19.12% |

### 10.1 The cost-tier reframe does not survive the corrected λ_pde

`docs/ADJUDICATION_2026-08-18.md` §2 let the A&T cost-tier argument survive as a labelled
post-hoc sensitivity, resting on the effect clearing the 10% bar across A&T's calibrated
range — in particular **+14.54% at their 0.5% stress tier**. That number is λ_pde-specific.
At the selection criterion's own optimum (1e-4) the same tier reads **−0.30%**: the effect
is gone, not merely reduced. At λ = 0 it is −0.45%.

So the reframe's strongest surviving number depends entirely on a hyperparameter that the
contract's own rule sets two orders of magnitude lower. **The cost-tier sensitivity should
now be withdrawn, not merely labelled** — it cannot be stated at any λ_pde the selection
criterion endorses.

The registered 1% tier is negative at **every** rung except the registered one, where it is
+0.01%. The registered null is therefore the single most favourable point in the entire
(λ_pde × tc) grid that was explored.

### 10.2 What is now established, and what it costs the paper

| claim | status after this audit |
|---|---|
| rung 3 reaches oracle-level delta-only CVaR₉₅ | **SURVIVES** — 2.076–2.090 vs oracle 2.079 at every λ_pde incl. 0; seed-robust; hyperparameter-independent |
| OOD Greek accuracy 86–93% | **SURVIVES** on the full grid; must be reported beside the hedging-box slice, where the arms rank oppositely |
| "+31.5% zero-cost hedging benefit" | **NOT IDENTIFIED** — monotone readout of λ_pde over 2.9%–31.5%; at the criterion optimum, +7.4% |
| "the effect clears 10% across A&T's calibrated range" | **WITHDRAWN** — −0.30% at 0.5% at the criterion optimum |
| "hedging and accuracy axes move independently" | **WITHDRAWN** — artifact of disjoint measurement regions |
| registered confirmatory verdict | **FAILS**, and fails harder at the corrected λ_pde |
