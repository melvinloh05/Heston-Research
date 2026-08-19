# Adjudication of the v6 result record — 2026-08-18

> **Second pass, same date (§8):** an external review caught that this document mixed CI
> conventions (seed-level normal CIs where the contract's registered statistic is the
> pooled-stratified paired bootstrap) without flagging the switch. §8 audits every affected
> claim; inline corrections are marked "[corrected — §8]". Melvin has also confirmed the
> 0.575% tier was reverse-engineered from our own 10%-bar interpolation — it is now DROPPED
> from claim text, not merely "document or drop".

Role: adjudicator, not advocate. Every number below was re-derived from the artifacts listed in
the request (contract YAML, `analyze_results.py`, the results CSVs/npz-derived per-seed tables,
the three PDFs). Where I relied on a code path I read it; where I relied on a summary I checked
it against the raw table. The order-attribution note patch and the band-grid extension were
treated as suspect and re-derived independently.

Record limitations acknowledged up front: no citation map, no A&T counter-memo, no
never-claim-firsts list (claim_memo_v8 §1 carries it by reference from v7; v7 is not in the
repo), no roadmap contingency buffer. The phrase "documented demotion, not goalpost-moving"
does not appear verbatim anywhere in the record; the closest in-record standards are AM3-2
("raising the bound so that the same ladder still passes would be circular — refitting a fitted
bound against the data that broke it") and AM3-3's reporting discipline ("reported AS an
extended-ladder result … the headline keeps the original ladder's answer alongside"). Those are
the standards I applied. Claim memo §8's "Sakuma 2026b" (gamma level → turnover) is not in the
record; only the 0DTE DML paper is. Any triangulation sentence touching 2026b is unverified.

---

## 1. Verdict audit

| criterion | coded verdict | YAML-derived verdict | discrepancy |
|---|---|---|---|
| confirmatory_cell | fail | **fail** | none. rel = +0.0002 (bar 0.10), pooled CI [−0.0242, +0.0242] includes 0, 10 seeds present. Clean, real fail. |
| order_attribution | fail | **fail** — but the registered gloss "fail = honest null" (YAML outcome_values comment) is **false in fact** | Re-derived independently of the patched note: diff = CVaR(rung2) − CVaR(rung1), pooled +0.0574 [+0.0527, +0.0621]; my per-seed re-derivation +0.0583 [+0.0347, +0.0820]. Significant **reversal**, not a null. The registered two-value vocabulary cannot express the outcome that occurred. The patch's content is correct; the vocabulary is the defect. |
| dose_response | flat | **flat**, glossed per Q2 as "monotonicity not demonstrated" ONLY | Verdict value correct. The note string ("flat = regularization null", analyze_results.py:687 and the docstring at :579) implements the **superseded** reading. The contract itself still contains that superseded reading at `label_noise_dose_response.readings` — Q2 (acceptance_thresholds.dose_response.interpretation) governs; the YAML carries an unreconciled stale clause (human amendment needed; YAML read-only for me). |
| ood_greek_thresholds | fail | **fail under the literal clause; the clause is mis-specified** | Γ red 0.894/0.864 and ν red 0.927/0.930 clear 0.15 by ~6×. The sole failing conjunct is parity: `price_parity = rel_rmse_arm/rel_rmse_base − 1` = −0.908/−0.821, coded two-sided (`abs(pp) ≤ 0.10`, analyze_results.py:744). The YAML text ("price parity within 0.10") states no direction; two-sided is the literal reading, and the code implements it faithfully. The clause's evident purpose (price is `role: sanity_check`; guard against sacrificing price for Greeks) is one-sided, and as coded the clause fails an arm for **improving** price 82–91% while σ_050 (half-destroyed labels) *passes* parity at near_feller (pp −0.056) — a reductio, but a design defect, not a scoring bug. Verdict stays fail as registered; report direction. |
| sakuma_null_consistency | flag | **flag is the faithful output of a mis-mapped check** | Sakuma Table 5 compares true Δ vs a **delta-supervised** DML net (his nRMSE_Δ = 0.8%). The correct translation is rung3-vs-oracle in-model: pooled −0.0061 [−0.0073, −0.0050] (10 seeds) — pooled-significant, but \|rel\| ≈ 0.35% is inside the registered 2% negligibility tolerance, so the criterion returns **consistent**: the Sakuma null reproduces on the right contrast [corrected — §8]. The coded contrast (rung3 vs standard_pinn) tests something Sakuma never ran; its 27.2% gap is a fact about the baseline's delta, and the flag correctly detected that the registered mapping was wrong. The consistency check did its job. |
| mechanism_adjudication | channel_i | **channel_i is the faithful output of the registered rule; the rule's interpretation is refuted by its own 2×2** | See §3. Additional defects: (i) memo prints "T_ex CI covers 0 (turnover unmoved)" beside CI [0.6618, 0.7154], which **excludes** 0 — the code conflates "not reduced" with "unmoved" (mechanism_memo, analyze_results.py:1101); (ii) the contract falsifier's literal wording "cost channel requires T_ex → 0 for clean arm" is **satisfied** (t_ex(rung3) = −0.0069 ≈ 0) while the code tests diff < 0 — the registered wording assumed an over-trading baseline; the actual baseline under-trades 38%. Channel-ii rejection still stands (no widening toward improvement), but the printed reason is wrong. |
| goldilocks_bates | no_decisive_regime | **tier-dependent; the contract fixes no tier** — the code inherited tc=0.01 | At 0.01, all 10 severity cells are significant **reversals** (+0.31…+0.67, CIs exclude 0 on the harmful side) — "no decisive regime" again hides significant harm. At tc=0 (a registered tier) the sweep shows +27.7% → −4.2% monotone decay; decisive improvement cells exist at low severity. The verdict is a function of an unregistered analysis choice and must be reported as such. Re-running at tc=0 is a free analysis change. |
| headline_scale_free | (metric, no row) | **undefined/degenerate at the registered cell** | At misspec tc=0.01 the oracle CVaR ≥ baseline; per-seed gap_closed is blank for 4/10 seeds and ranges 0.43–1.45 on the rest; in-model at 0.01 it emits ≈0.87 toward an oracle that is *worse* than baseline (meaning inverted); at 0.575% it emits 1.44–1.77. Clean only at low tiers (rung3 closes 98.9% misspec / ~101% in-model at tc=0). The metric presupposes oracle-beats-baseline, which fails for tc ≳ 0.5%. Emission should be sign-guarded. |

**Doubt (d), answered:** yes — one tier choice (0.01) sits at the effect's zero crossing
(interpolated crossing ≈ 1.0005%; measured rel at the tier +0.0002) and simultaneously
degrades confirmatory_cell, order_attribution, the dose-response's y-axis, goldilocks, and the
headline scale-free metric. That is a reportable design finding about scorecard fragility. It is
**not** a warrant to re-score anything: at the registered tier the effect is genuinely zero
because two real forces cancel (tracking gain vs cost of restored turnover), and that
cancellation is a result, not an artifact.

---

## 2. The reframe, prosecuted

### For "post-hoc rationalisation" (prosecution)
1. 1% is justified **nowhere** in the contract. `transaction_costs.tiers: [0, 0.01, 0.02]`
   carries no rationale, no citation; no amendment among Q1–AM4 touched it; the registration
   ledger records it as "held". It is registered three times over.
2. The team **had A&T in-repo since July 2** and cites it in the same YAML (the T′
   construction, the mechanism triangulation). A&T §4.3's calibrated rates were available at
   freeze time and 1% was registered anyway. "Mis-calibrated nuisance parameter" is an
   admission against the registration, not against the world.
3. The ATC run (7 tiers) **post-dates the registered fail**. The recalibration argument was
   constructed after the result was known. The 0.575% tier appears **nowhere in A&T** (their
   tiers: 0.005%/0.25% normal, 0.5% high) and has no documented provenance anywhere in the
   repo; it reads as reverse-engineered ("last tier that clears"). Calling it part of "their
   full calibrated range" is false.
4. The crossing-at-1% is a fact about **this design** (daily grid, T′ = 0.17, ATM, this
   baseline's 38% under-trade, CVaR95). It moves with frequency, horizon, moneyness. Its
   coincidence with the registered tier carries no meaning either way.
5. At every tier ≥ 1% the winner is the **no-trade-band baseline** (registered-grid band 0.04:
   4.591 vs rung3 4.859 at 1%) — so "the tier was unfair to rung3" undersells that at that
   tier the entire accurate-hedging family loses to cost-aware laziness (Whalley–Wilmott). A
   band-on-rung3 arm was never run; the cost-regime frontier is unexplored.
6. The counterfactual test: had the effect died at 0.5% too, the same argument-form would
   retreat to 0.25%, then 0.005%. An argument that can always retreat is not falsifiable as
   stated.

### For "honest reading of a mis-registered parameter" (defense)
1. The **registered design itself contains tc=0** as a first-class tier, the {0, 1%, 2%}
   mechanism sweep, and the 2×2. The +31.5% zero-cost result (pooled CI [−0.980, −0.944],
   10 seeds) is pre-registered machinery, not a rescue. No reframe is needed to report it.
2. The A&T anchoring is external, instrument-class-matched (S&P-like index call), and
   **leg-correct**: their p1 attaches to "Option 1 with K1 = 0 (equivalent to using the
   underlying S)" and "for the delta hedging strategy we use only Option 1" — the 0.005%
   normal rate is the only leg a delta-only hedger trades. 1%/0.005% = 200×; 1%/0.5% (their
   high tier) = 2×. Both of Melvin's ratios check out as stated.
3. Measured at the A&T-anchored post-hoc tiers (10 seeds, paired CIs excluding 0):
   +31.3% at 0.005%, +22.7% at 0.25%, +14.5% at 0.5%; +12.2% at the provenance-free 0.575%;
   10%-bar crossing interpolates to ≈ 0.65%. Crucially, at the one rate that is actually
   calibrated for the traded leg (0.005%), the effect (31.3%) is indistinguishable from
   frictionless (31.5%) — the claim needs **one calibrated point, not a range**, which removes
   the retreating-goalpost structure.
4. The project's own precedent (AM3-2/AM3-3, and the band-grid extension handling) defines the
   honest template: extensions enter diagnostic, are reported as extensions, and the original
   answer stays the headline. The demoted form of the reframe fits that template exactly.

### Ruling
**The reframe as stated dies.** The sentence "the registered verdict is a FAIL produced by a
mis-calibrated nuisance parameter, not by absence of effect" is not licensed: it re-scores a
registered endpoint post-hoc; "mis-calibrated" invokes a calibration standard the registration
never adopted; and at the registered tier there **is** absence of effect — the zero is real.

**What survives, under stated conditions** (all four required):
1. The registered FAIL stays the headline verdict, first and unhedged, in abstract and results.
2. The TC-profile is reported as a **post-hoc sensitivity analysis** with the tier-addition
   timing disclosed (added after the registered verdict was known); every figure marks
   registered vs post-hoc tiers.
3. The 0.575% tier is **dropped from all claim text** (Melvin has confirmed it was derived by
   interpolating where our own effect crossed the 10% bar — reverse-engineered, exactly the
   prosecuted structure); "A&T's calibrated range" never covers it. The claim leans on the
   single calibrated underlying-leg rate (0.005%) and on 0.5% as *their* stress tier, with the
   leg attribution stated. The 10%-bar crossing recomputed without the tainted tier
   (interpolating 0.5% → 1%) is ≈ 0.66%, essentially unchanged.
4. The band-baseline result is reported alongside: at ≥1% the registered-grid band on the
   *worst* delta beats every supervised arm and the oracle — the cost regime rewards
   cost-awareness, not accuracy, and this project did not test cost-aware supervised arms.

Permissible sentence: "The pre-registered confirmatory test fails at its registered 1% tier
(+0.02%, CI includes 0). The registered tier is 200× the underlying-leg cost Armstrong & Tatlow
calibrate for the same instrument class and 2× their high-cost stress tier; in post-hoc tiers at
their calibrated normal (0.005%) and stress (0.5%) rates the effect is +31.3% and +14.5% with
paired 95% CIs excluding zero, and its zero crossing sits at ≈1%."

---

## 3. The mechanism

**Which channel: neither registered one.** The evidence triangulates to a channel outside the
menu — **delta-approximation quality → discrete-hedge tracking error**:

- In-model gap +27.2% at tc=0 (10 seeds, CI excl. 0); misspecified +31.5%. ~86% of the effect
  needs no misspecification. (Caveat, correctly noted by Melvin: the anchor is excised from
  training — nearest point 0.29 in range-normalized L2 per claim memo §5.1's 0.231–0.367
  range — so "in-model" still tests parameter-space generalization. That refines, not rescues,
  channel (i): the "misspecification" the supervision defends against is the net's own
  approximation error, not the DGP's.)
- Bates severity: +27.7% → −4.2% monotone decay as jump intensity rises; Merton +9.8% → −3.7%
  at 1%. A robustness channel predicts persistence or growth with misspecification; the record
  shows decay. Within-family perturbation adds only ~+4.3pp (27.7 → 32.0 at m: 0 → 1) — the
  genuinely robustness-flavored component is that ~4pp residual, riding a ~28pp in-model base.
- Mean PnL is identical across arms at tc=0 (+0.0773); the entire gap is tail/variance —
  replication error, not drift.
- Turnover: oracle 0.0398, rung3 0.0397 (t_ex −0.0069 ≈ 0), std_pinn 0.0248 (t_ex −0.6955).
  The baseline **under-trades 38%** because its delta is damped — the opposite of the noisy
  over-trading baseline the cost channel (ii) presupposed. Supervision *restores* oracle-level
  turnover, which costs money at tc > 0; the cost mechanism operates against the treatment.
- The coded channel_i verdict is the faithful output of the registered rule; the registered
  interpretation of channel_i ("model-uncertainty robustness, A&T-adjacent") is refuted by the
  in-model corner — which the registered 2×2 existed to check, and which flagged. **The design
  caught its own menu's incompleteness.** A&T's own conclusion (gamma hedging's value is
  robustness, not cost reduction) is about strategy choice with exact Greeks; it does not
  transfer to Greek-error-in-delta-only hedging, and our record does not contradict it — it
  measures a different object.
- The missing channel was in the project's intellectual inventory the whole time: claim memo §8
  lists Noguer as "gamma **error** → discretization bound, theory only". The data landed on the
  channel the menu omitted. **Report as a design finding, plainly.**

**Was the menu incomplete? Yes** — and doubly so: it lacked the tracking/approximation channel,
and its cost channel had the sign of the baseline's turnover pathology backwards.

---

## 4. The ladder, and what may be claimed about Gamma

Measured at tc=0, of the 31.5% total: +Δ 28.94pp (91.9%), +Γ 2.04pp (6.5%), +ν 0.53pp (1.7%).
Per-tier paired CIs for the Γ increment (rung2 − rung1, 10 seeds):

| tc | 0 | 0.005% | 0.25% | 0.5% | 0.575% | 1% | 2% |
|---|---|---|---|---|---|---|---|
| diff | −0.062 | −0.062 | −0.049 | −0.022 | −0.012 | **+0.058** | +0.221 |
| CI excl. 0? | yes | yes | yes | yes | no | **yes (harmful)** | yes (harmful) |

[corrected — §8] The table above uses seed-level CIs (the registered *companion* statistic).
Under the registered pooled-stratified statistic the marginal cells sharpen: at 1% the Δ
increment still helps (pooled −0.0655 [−0.0872, −0.0433]), the Γ increment reverses (both
conventions agree), and the ν increment **also reverses** (pooled +0.0071 [+0.0060, +0.0081];
seed-level CI includes 0). At 0.575% the Γ increment is still a pooled-significant
*improvement* (−0.0122 [−0.0163, −0.0080], rel +0.3% — economically negligible). So: under the
registered statistic, Γ and ν reverse at 1% while Δ does not under either convention; all
three reverse at 2% under both.

Meanwhile rung1→rung2 cuts OOD Gamma RMSE ≈ 60% on both held-out regimes (0.130/0.328 and
0.106/0.261 RMSE ratios), vanna improves similarly, price improves, and the info-matched
price-only baseline plateaus at 0.605 validation gamma rel-RMSE (2% rule, capacity control
flat) — gamma labels deliver what prices cannot.

**Ruling on Gamma as the subject:**
- Of a **Greek-accuracy** claim: yes. The ~60% Γ-RMSE cut is ladder-attributed, large,
  CI-backed, and info-matching-defended.
- Of a **hedging** claim: no. The hedging increment is +2pp at zero cost, gone by 0.575%,
  significantly harmful at the registered tier. The hedging story belongs to Δ supervision
  (and to the baseline's pathology — see §6).
- The bridge "better Gamma → better hedging" is severed by three record facts: bs_gamma (wrong
  gamma labels, best hedger, beats the oracle in-model by −0.156 [−0.166, −0.145]);
  feedforward (worse OOD delta RMSE than std_pinn, hedges far better — grid RMSE and
  path-relevant accuracy dissociate); and the cost reversal.
- Title and abstract **cannot keep Gamma as the subject of the hedging claim**. CLAUDE.md's
  "the claim lives on the rung1→rung2 gap": on the registered confirmatory metric, that gap is
  a significant reversal. The claim as scoped is dead; what lives on that gap is accuracy.

---

## 5. The dose-response tension, reconciled

Two axes, two answers, no contradiction:

- **Accuracy axis (OOD Γ RMSE): monotone.** Reductions vs std at near_feller: 0.894 (σ=0) →
  0.858 (0.10) → 0.800 (0.25) → 0.649 (0.50) → 0.553 (bs) → −0.134 (shuffled) → −1.329
  (gradpen); same ordering at strong_neg_corr. Melvin's rung1-relative control numbers
  (−246/−370%, −36/−65%, −611/−903%) re-derive exactly. Correctness matters for delivered
  Greeks. (No registered test on this axis — the formalization is a free analysis addition.)
- **CVaR axis [corrected — §8]: small monotone dose effect at tc=0 under the registered
  pooled statistic; inverted at 1%; biased points order backwards.** Pooled, the σ-arms at
  tc=0 degrade CVaR in dose order — +0.0112 / +0.0229 / +0.0432 for σ = 0.10/0.25/0.50, each
  CI excluding 0 — i.e., σ=0.5 gives back roughly half to 70% of the (small) Γ(+ν) increment.
  None of these is seed-robust (seed-level CIs include 0; for dose arms the seed also carries
  the frozen noise realization, so seed-level is the generalization-relevant uncertainty).
  Unbiasedness of the multiplicative corruption explains the small absolute magnitude, not a
  flat curve. At tc=1% the same statistic detects the *inversion* (σ_050 beats σ_000, pooled
  −0.0678). The biased points still dominate everything: bs_gamma beats true labels by
  −0.267 [−0.272, −0.261] pooled at tc=0 and beats the **oracle hedger in-model**
  (pooled −0.156 [−0.164, −0.147]). The catastrophic controls are optimization-conflict damage —
  shuffled destroys the *delta* fit too (delta reduction −0.82/−1.20) — they evidence "label
  conflict poisons training", not "gamma correctness drives hedging".
- **The honest statement:** gamma-label correctness monotonically improves delivered gamma
  accuracy; delta-only CVaR95 at any tested tier is insensitive to unbiased gamma-label noise
  up to σ=0.5 and is *improved* by a biased Black–Scholes gamma label that also beats the
  exact-Greek oracle — i.e., this hedging metric does not measure gamma-label correctness.
  Under Q2 the registered verdict is "monotonicity not demonstrated", and the record shows the
  registered criterion was pointed at a metric that cannot demonstrate it.
- **To distinguish "insensitive metric" from "genuine regularization null":** (i) formalize the
  accuracy-axis dose-response (data exists); (ii) dose the **bias**, not the variance
  (damped/mixed gamma labels); (iii) run a minimum-variance-delta oracle comparator (the CF
  oracle already computes vega/vanna) to test whether bs_gamma's advantage is the MV-delta
  correction — if MV-oracle ≈ bs_gamma, the anomaly is explained and "CVaR doesn't reward
  exact-Greek correctness" becomes a theorem-shaped claim; (iv) a higher-frequency rebalancing
  cell (a registered lever) where the discretization term Γ actually binds.

---

## 6. Three candidate stories

**Story 1 — what happened (baseline pathology + cost inversion).** The PDE-residual-only
baseline learns a systematically damped delta: in-model CVaR 37% above oracle, turnover 62% of
oracle, and a *plain price-only feedforward net* beats it by 30% (−0.928 [−1.008, −0.847];
**5 seeds** — this load-bearing number needs the 10-seed upgrade before it anchors prose) —
while sans_pde matches rung3 to within 0.2% (pooled −0.0047 [−0.0068, −0.0025]: sans_pde
marginally *better*, pooled-significant, not seed-robust) and λ_pde=0 scored within 4% of the
selected 0.01 on the baseline's own validation. Any usable training signal fixes most of the
gap; Greek labels polish tracking to the oracle frontier (98.9% of gap at tc=0). Costs invert
the ranking because damped deltas under-trade (WW-efficient); at 1% the forces cancel almost
exactly. Separately and robustly, Greek supervision buys large genuine OOD accuracy gains.

**Story 2 — plausibly artifact.** The 27–31% headline is a property of *this baseline's*
pathology, at *this* λ_pde, *this* architecture: "Sobolev vs standard PINN" generalizes only to
baselines with similarly damped deltas. The registered λ_pde robustness row (rung3-sourced)
was never run; feedforward/bs_gamma/dose cells are 5-seed; the band arm rides on std_pinn
specifically; the exact 1.0005% crossing is a numerical coincidence of this design (a crossing
existing is robust; its location is not meaningful).

**Story 3 — genuinely surprising.** bs_gamma beats the exact-Greeks oracle hedger **in-model**
at tc=0 by ~9% with a tight CI. If real, exact-delta tracking is measurably suboptimal for
discrete CVaR95 hedging under Heston (the MV-delta correction ρξ·∂C/∂v-shaped), and a
BS-gamma-biased fit lands closer to the discrete-optimal hedge. This retro-explains the CVaR
axis's refusal to reward label correctness and is the seed of the follow-up paper.

**I believe Story 1**, with Story 2 absorbed as stated scope limitations and Story 3 flagged as
the one new phenomenon requiring the MV-comparator check before any claim beyond "observed,
unexplained".

---

## 7. Deliverables

### The claim that survives (one sentence)
"Sobolev supervision of a parametric Heston PINN improves held-out out-of-distribution Gamma and
Vega accuracy by 86–93% — with a ~60% Gamma-RMSE cut attributable to gamma labels specifically,
and unreachable by information-matched price-only training — while its delta-only hedging
benefit (+31.5% misspecified CVaR95 at zero transaction cost, 10 seeds, paired CI excluding
zero) is carried by delta supervision, is present without misspecification, decays with
transaction costs to zero at the registered 1% tier, and reverses for the gamma rung there."

### Claims that do not survive
1. "Gamma supervision improves delta-only hedging under misspecified dynamics" — the registered
   claim: null at the confirmatory cell, significant reversal on the order-attribution rung.
2. "The mechanism is model-uncertainty robustness (channel i)" — refuted by the in-model gap
   and severity decay; the coded channel_i is a rule output, not a finding.
3. "The mechanism is transaction-cost/turnover reduction (channel ii)" — the baseline
   under-trades; supervision raises turnover; costs punish the treatment.
4. "Correct gamma labels matter for hedging" — flat over unbiased noise, inverted at biased
   points, bs_gamma beats the oracle.
5. "The registered fail is an artifact of a mis-calibrated tier" — dies as a verdict-changer;
   survives only as the demoted, labeled sensitivity analysis of §2.
6. "OOD Greek endpoint passed" — fails as registered (two-sided parity clause); report as
   failed-on-the-guard with direction stated.
7. "The PDE residual earns its keep" — economically inert given labels (sans_pde is within
   0.2% of rung3; pooled-significantly *better*, not seed-robust — "ties" is the wrong word,
   "≤0.2% either way" is right [corrected — §8]), and the supervision-OFF cell (feedforward,
   5 seeds) beats the residual-only baseline on hedging; on OOD Greek RMSE feedforward is
   worse than std_pinn — state both, claim neither direction globally.

### What the abstract MAY say
- The pre-registered confirmatory hedging test failed, stated first with its numbers.
- The Greek-accuracy claim (with ladder attribution and the info-matching plateau).
- The zero-cost hedging effect exists (+31.5%, CI), is delta-carried, is present in-model,
  decays monotonically with cost, and crosses zero at ≈ the registered tier.
- At A&T's calibrated underlying-leg cost (0.005%) the effect is indistinguishable from
  frictionless — labeled post-hoc, leg attribution stated.
- The registered mechanism menu was incomplete; the registered in-model consistency cell
  detected it; the operative channel is approximation/tracking.
- The band-baseline dominance at ≥1% (Whalley–Wilmott-consistent) and the bs_gamma anomaly,
  reported as findings.

### The three sentences it MUST NOT say
1. "Explicit Gamma supervision improves delta-only hedging PnL under misspecified dynamics."
2. "The improvement operates through the model-uncertainty-robustness channel" (or the
   transaction-cost variant).
3. "The confirmatory test would have passed at a correctly calibrated transaction-cost tier"
   (any phrasing that re-scores the registered fail as an artifact).

(Standing prohibitions from the claim memo remain: no "complete factorial" phrasing beyond
what D1's resolution supports; no "any curvature regulariser would do"; no "three independent
legs against one"; no claim the ADI fix improved boundary error bars.)

### Open items, ranked by whether the paper is defensible without them
1. **CI-convention discipline (required; no GPU) [added — §8]:** the contract already fixes
   the convention (`tail_claim_requires: paired_bootstrap_over_CRN_paths_with_seed_variance_
   separated`): pooled-stratified is the verdict statistic, seed variance the mandatory
   companion. Every contrast in the paper reports **both**; prose vocabulary is fixed as
   "pooled-significant" (these trained models differ on this path population) vs "seed-robust"
   (the treatment replicates across training runs); no generalization claim on
   pooled-significance alone; an economic-relevance floor accompanies words like
   "inert"/"matches" (pooled power resolves 0.2% effects that mean nothing).
1b. **Reporting-layer contract-compliance fixes (required; no GPU):** dose-response note and
   docstring (Q2 wording); mechanism-memo T_ex sentence ("covers 0" ↔ excludes 0); goldilocks
   tier disclosure + re-run at tc=0 (analysis only); sign-guard gap_closed emission;
   registration-ledger annotation for reversal rows. The paper cites this record; it must be
   internally consistent.
2. **MV-delta oracle comparator (required for any bs_gamma sentence beyond "unexplained";
   CPU-cheap):** hedge the same banks with the Heston minimum-variance delta
   (Δ + (ρξ/S)·∂C/∂v). Explains or kills Story 3; the needed citation for MV-delta
   (Föllmer–Schweizer / local-risk-minimization family) is **not in the record** — flag,
   do not cite until obtained and read.
3. **λ_pde robustness row (registered commitment, currently unmet; GPU):**
   `lambda_selection.robustness_row` promises the confirmatory contrast at rung3-sourced
   λ_pde. Its absence is registration debt; the baseline-pathology story leans on λ_pde=0.01
   being std_pinn's own optimum.
4. **Band-on-rung3 / band-on-oracle arms (strengthens the cost story; moderate):** without
   them the paper stays silent on what to deploy at ≥1% — currently the only cost-aware arm
   is built on the worst delta.
5. **10-seed upgrade of feedforward and bs_gamma cells** if either moves into the headline
   (currently 5-seed).
6. **Higher-frequency rebalancing cell + biased-label doses (follow-up paper):** the fair
   hedging test for gamma (Noguer channel) and the designed version of the dose-response.
7. **0.575% tier: RESOLVED — drop from claim text (no run).** Melvin confirmed it was
   interpolated from where our own effect crossed the 10% bar; it has no external provenance
   and stays out of every claim, caption, and "calibrated range" phrase. The data row may
   remain in sweep tables as an unlabeled point.

### Null-outcome / pivot ruling
This is **not** a nothing-paper and it is **not** the registered paper. The registered headline
(gamma supervision → better misspecified delta-only hedging at 1% TC) is dead and must be
reported as a pre-registered failure. What stands is (a) a strong, pre-registered-machinery
Greek-accuracy result with clean ladder attribution and an information-matching defense, and
(b) an unusually well-instrumented dissection of *why* hedging CVaR does not reward Greek
correctness — baseline pathology, cost inversion, tracking channel, menu incompleteness caught
by the design's own consistency cell, and the bs_gamma anomaly. The pivot is of framing
(accuracy paper + adjudicated hedging nulls + design findings), not of substance. Losing the
gamma-hedging title here costs less than losing it in review.

---

## 8. Second-pass correction — CI-convention audit (same date)

An external review of this adjudication caught a real defect in **my own analysis**: I computed
seed-level normal CIs (mean ± 1.96·sd/√n over per-seed paired diffs) for several marginal
contrasts because they were derivable from the per-seed CSVs, while the contract's registered
tail statistic is the pooled-stratified paired bootstrap
(`tail_claim_requires: paired_bootstrap_over_CRN_paths_with_seed_variance_separated`;
`_pooled_stratified` in analyze_results.py). I did not flag the switch, and for two contrasts
it produced the more conservative-sounding reading. The criticism is accepted. I re-ran every
marginal contrast through the registered pooled machinery (`paired_ci_from_npz` on the persisted
PnL npz banks). Verified results:

| contrast | seed-level (what §1–§7 originally used) | pooled (registered) | verdict change |
|---|---|---|---|
| ν increment (rung3−rung2) @1%, 10s | +0.0068 [−0.0008, +0.0144] "null" | +0.0071 [+0.0060, +0.0081] | **flips: ν also reverses at 1%** |
| sans_pde−rung3 @0, 5s | −0.0054 [−0.0281, +0.0172] "ties" | −0.0047 [−0.0068, −0.0025] | **flips: pooled-significant (sans_pde better by 0.2%)** — economically inert wording survives, "ties" does not |
| Γ increment (rung2−rung1) @0.575%, 10s | −0.0119 [−0.0313, +0.0074] "null" | −0.0122 [−0.0163, −0.0080] | **flips the other way: still a (negligible, +0.3% rel) significant improvement** |
| σ_010/025/050 − σ_000 @0, 5s | all include 0 → "flat over unbiased noise" | +0.0112 / +0.0229 / +0.0432, all exclude 0, dose-ordered | **flips: pooled-monotone dose effect at tc=0**; σ=0.5 returns ~half–70% of the Γ(+ν) increment |
| σ_050 − σ_000 @1%, 5s | (not separately claimed) | −0.0678 [−0.0786, −0.0575] | inversion at the registered tier is pooled-significant |
| rung3 − oracle in-model @0, 10s | −0.0062 ± 0.0068 "ties" | −0.0061 [−0.0073, −0.0050] | pooled-significant, but rel +0.35% < the registered 2% tolerance → **sakuma "consistent" stands via the negligibility branch**; "ties" corrected to "differs by +0.35%, inside tolerance" |
| Δ increment (rung1−std) @1%, 10s | −0.0658 [−0.1006, −0.0310] | −0.0655 [−0.0872, −0.0433] | no change (helps under both) |
| bs_gamma − σ_000 @0 / bs_gamma − oracle in-model | −0.268 / −0.156 | −0.267 [−0.272, −0.261] / −0.156 [−0.164, −0.147] | no change (robust under both) |
| order attribution, confirmatory, feedforward, band results | (pooled or far from 0) | — | no change |

**What this changes in the adjudication:** the ladder statement in §4 (ν reverses at 1% under
the registered statistic; Γ's sign flip is sharp between 0.575% and 1%); the dose-response
narrative in §5 (the CVaR axis at tc=0 is not flat under the registered statistic — it shows a
small, dose-ordered, pooled-significant degradation that is not seed-robust; the "unbiased
noise ⇒ designed-in flatness" argument is demoted to explaining magnitude, not shape); and
wording for sans_pde and rung3-vs-oracle ("ties" → bounded-difference statements). **What it
does not change:** every headline ruling — the registered fail, the reversal, the reframe
demotion, the mechanism channel, the Gamma scoping, the bs_gamma anomaly, the pivot ruling.
The registered dose_response verdict ("flat" at the 1% cell) is also unchanged — that
machinery (Spearman over seed-mean CVaRs) ran as registered.

**Convention ruling (now open item 1):** the contract already decides this — pooled decides
threshold verdicts; seed variance is the mandatory companion. The two CIs answer different
questions: pooled = "do these trained hedgers differ on this path population?" (conditions on
the realized training runs and, for dose arms, on the frozen noise realizations); seed-level =
"does the treatment reproducibly produce this difference across training runs?" The paper
reports both for every contrast, reserves generalization language for seed-robust effects, and
pairs pooled-significance with an economic-relevance floor. One caveat back at the review: the
seed-level CIs I used are not an alien statistic — they are the contract's own companion; the
defect was mixing conventions without flagging, not using an unsanctioned one.

Also recorded here: Melvin's confirmation that **0.575% was reverse-engineered** (interpolated
from where our own effect crossed the 10% bar). Prosecution point 3 of §2 is thereby
confirmed, the tier is dropped from all claim text, and the 10%-bar crossing quoted anywhere
is the one interpolated from {0.5%, 1%} only: ≈ 0.66%.
