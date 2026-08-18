# Claim Memo v8 — Sobolev-PINN Greeks for Heston Option Pricing

**Supersedes:** claim_memo_v7.md (Aug 2026), claim_memo_v6.pdf (July 2026)
**Governing documents:** `heston_benchmark_v6.yaml` + amendments Q1, Q2, AM2-2, AM3, amendment 4
**Label artifacts:** `data/frozen/v6-labels-20260812/` — git tag `v6-labels-20260812`, commit `20e3c01`
**Status:** pre-M4. No experimental results exist. Every number below is a design parameter, a
contract value, or a frozen-artifact fact — never an outcome.

> **Reading rule.** Where memo and contract disagree, the contract wins. v8 closes v7's two
> `[PENDING-AM3]` placeholders with values read from the contract and
> `audit/contract_amendment_3_notes.md`, records the freeze, and adds four findings v7 predates.
> One decision (D1) remains genuinely open.

---

## A. Change ledger, v7 → v8

| # | Section | v7 said | v8 says | Source |
|---|---|---|---|---|
| 1 | header | "the staged artifact" | Labels are **FROZEN and TAGGED**: `data/frozen/v6-labels-20260812/`, tag `v6-labels-20260812`, commit `20e3c01`. Freeze notes ship with the artifact | promotion, 2026-08-12 |
| 2 | §4.3 / D2 | ADI at "explicit, **non-default** `vmax`/`nv`", values `[PENDING-AM3]` | `vmax = 1.6`, `nv = 481` — and these are now the **DEFAULTS** in `heston_greeks_adi`, not caller-supplied overrides. "Non-default" is struck | commit `20e3c01` |
| 3 | §7 / D3 | σ-ladder, `clipped_frac_max`, extension contingency all `[PENDING-AM3]` | Supplied in full, §7 below | contract `oracle_headroom_gate`; AM3 notes |
| 4 | §4.3 | (silent) | **The `feller_violating_volvol` vega uncertainty ≈ 1.33 is genuine MC noise, NOT an ADI artifact, and the fix does not reduce it.** Directly governs F7's "effect must exceed oracle noise" | §4.5 |
| 5 | §5 | (silent) | The [0.40, 0.60) Feller band **still masks at 0.2434** after the fix (from 0.5192), against ~0.09 elsewhere | §5.2 |
| 6 | §5 | (silent) | Mask-neutrality panel (b) moved in **both directions**: κ/θ/ξ improved, **ρ and v0 got worse** | §5.2 |
| 7 | §5 | (silent) | The contract's three neutrality checks are **not** the report's three panels; check 2 is arm-independence, which is structural | §5.3 |
| 8 | §4.2 | `feedforward` is the only roster gap | Also: `rung0_price_only` is **dispatched at 10 seeds, never hedged**, and is config-identical to `standard_pinn`. Dispatch is 90 runs ≈ $73 | roster audit |
| 9 | §4.3 | oracle certified | `oracle.py --full` **7/7 PASS at the shipped defaults** — certification and as-run now agree, which they did not before `20e3c01` | re-certification |

Carried unchanged from v7: §0 problem statement; §1 surviving cell and the never-claim-firsts
list; §2–3 competitive picture and positioning; §6 thresholds except as noted; §8–9 triangulation
and novelty matrix.

---

## 0.–3. Unchanged from v7

Problem statement, the three reasons Greek RMSE cannot carry the claim, the surviving distinct
cell, the competitive picture, and the positioning axes are unamended. Sakuma 2026a remains the
nearest method neighbour and threatens the configuration, not the cell.

---

## 4. Evidence contract (v8)

### 4.1 Primary, held-out only

- **OOD type 1** (held-out Heston parameters): Greek accuracy **and** hedging PnL. Two difficulty
  tiers: `near_feller` (interpolation, ball excised) and `strong_neg_corr` (extrapolation beyond
  the sampled ρ range).
- **OOD type 2** (cross-model Bates, Merton): hedging PnL only. Cross-model Greek RMSE stays dropped.

**Naming trap worth one sentence in the paper.** `near_feller` has Feller ratio **1.038** — it is
*above* the boundary despite the name. The regime that actually violates Feller is
`feller_violating_volvol` at **0.444**. Every fourth-leg result below turns on that distinction,
and a reader who assumes `near_feller` is the stressed regime will misread §4.3.

### 4.2 The residual × supervision factorial — three cells, unchanged from v7

|  | PDE residual ON | PDE residual OFF |
|---|---|---|
| **Greek supervision OFF** | standard PINN (baseline 1) — **runs** | price-only NN (baseline 0) — **defined, not dispatched** |
| **Greek supervision ON** | Sobolev-PINN (rung 3) — **runs** | `sobolev_sans_pde` (ω_PDE = 0) — **runs** |

Identified: the residual main effect at supervision-ON (rung 3 vs `sobolev_sans_pde`) and the
supervision main effect at residual-ON (rung 3 vs baseline 1). **Not** identified: the residual
main effect at supervision-OFF, nor the interaction.

The v7 prohibition stands — no draft, slide, or caption may say "complete factorial," "completed
factorial," or "the missing fourth cell" until D1 is resolved.

**Arm-roster facts established by audit (new in v8):**

- `feedforward` (baseline 0) is in `pinn_config.yaml` but in neither the training dispatch nor the
  hedging roster. **D1.**
- `rung0_price_only` is dispatched at **10 seeds and never hedged**, and its `PINNConfig` is
  **bit-identical** to `standard_pinn` (and to `info_matched_baseline`). Ten GPU runs producing
  duplicates of a model that is already trained. Dropping it funds `feedforward` twice over.
- `info_matched_baseline` being config-identical to `standard_pinn` is *by design* — the
  information matching is a **data** manipulation (`n_price_points` swept per `info_matching`), not
  a config flag. Worth stating so a reader does not read the identity as an error.
- `gamma_only` is a YAML alias of `rung2_delta_gamma`; the Vega-marginal reading is the rung2→rung3
  gap and both rungs run. The claim's "Gamma (with Vega)" is not decoration.
- `sigma_000` is a YAML alias of rung 3 and resolves to rung 3's checkpoint, keeping a distinct
  σ = 0 row in the dose-response without a redundant fit.
- Dispatch as it now stands: **90 runs, ≈ $73** at the module's own L40S estimate.

### 4.3 Oracle — fourth leg, settings and certification

Legs 1–3 unchanged (autodiff through the trap-free CF; high-order finite differences on the CF
price grid; Monte-Carlo pathwise/likelihood-ratio), agreement tolerance `1e-3` relative, masking
rule `tol_rel * scale + 3σ_MC` for pairs involving the MC leg.

**Fourth leg: Craig–Sneyd ADI**, required on `near_feller`, `feller_violating_volvol`, and any
sampled point with Feller ratio in [0.40, 0.60]. Grid settings, now the function defaults as of
commit `20e3c01` and the configuration `oracle.py --full` certifies:

    nx = 901,  nv = 481,  xmax = 4.0,  vmax = 1.6,  steps_per_year = 1000,  cx = 0.05

`oracle.py --full`: **7/7 PASS**. Certification and as-run agree; before `20e3c01` they did not.

### 4.4 The cross-validation finding — expanded (worth ~two sentences in the paper's §4)

At the previous default `vmax = 0.8`, the variance domain is truncated inside the mass of the
ξ = 0.60 regime and the artificial boundary contaminates the v-direction derivatives. On
`feller_violating_volvol`:

- the vega disagreement with CF was **systematic** — ADI > CF on **87.9%** of grid points;
- it grew **monotonically with τ**, from 0.0 at τ = 0.04 to 0.0087 at τ = 1.0, the boundary error
  propagating inward;
- it masked **59.4%** of that regime's vega grid at `agreement_tol_rel = 1e-3`.

**It was domain truncation, not resolution.** Four hypotheses, eliminated in order:

| hypothesis | test | result |
|---|---|---|
| MC noise | compare 3σ_MC to the gap | ruled out — MC's band exceeds the disagreement on 85% of points; it cannot vote |
| time stepping | 1000 → 2000 → 4000 steps/yr | ruled out — vega identical to six decimals |
| v-grid resolution | `nv` 241 → 481 | ruled out — vega unchanged **while gamma improved**, so refinement works and vega was not resolution-limited |
| **domain truncation** | `vmax` 0.8 → 1.6 at **unchanged** `nv = 241` | **confirmed** — vega mask 0.5942 → 0.0033 on a *coarser* grid near v = 0 |

`vmax` 1.6 → 3.2 buys almost nothing (median 4.0e-05 → 2.8e-05), so 1.6 is converged. `near_feller`
is **flat** under the same sweep — the control that localises the effect to broad-ξ regimes.
Gates 6/7 improve: fvv worst relative error 3.56e-03 → 1.57e-03 against the 5e-03 bar.

**Honest framing for the paper.** The surviving legs were the externally validated ones (Albrecher
literature prices; Black–Scholes recovery as ξ → 0 — both `cf`/`fd` only). But CF and FD are **not
independent**: FD is FD-on-COS, the same characteristic-function machinery. So this is CF/FD
versus ADI with MC unable to break the tie, not three legs against one. Say it that way.

### 4.5 Oracle noise on the boundary regime — NEW, and it constrains F7

Fixing ADI fixed the **mask**; it did **not** fix the **error bars**. `feller_violating_volvol`
vega `uncertainty_mean` moved only 1.3362 → 1.3299, because
`uncertainty = max(maxdiff, mc_se)` and:

- mean `mc_se` = **1.2549** vs mean `maxdiff` = 0.7438;
- `mc_se` is the **binding** term on **70%** of points;
- mean `|cf − mc|` = 0.7435 while mean `|cf − adi|` is now **0.0007**.

That number is **genuine MC estimator noise at ξ = 0.60**, not a solver artifact. It matters
because `reporting_discipline_F7` requires boundary-regime effects to **exceed oracle noise** — on
that regime the bar is real and high, and no ADI change lowers it. Reducing it would need far more
MC paths on top of the 4× multiplier already applied there. Any draft implying the fix improved
the error bars is wrong.

### 4.6 Hedging backtest — unchanged from v7

Delta-only in the underlying, fixed across arms. ATM call, S₀ = K = 100, τ₀ = 0.25, hedged to
T′ = 0.17 and liquidated. **Rebalancing: 43 steps, `dt_realized = 0.003953488` (252.94/yr)** — T′
does not divide evenly into 1/252 steps, so "daily at dt = 1/252" is wrong as a methods statement
(Q1). 10,000 paths per cell from CRN banks shared by all arms. Costs {0, 1%, 2%}. Confirmatory
cell: combined perturbation, 1% TC, baseline anchor, rung 3 vs standard PINN, 10 seeds.

Mechanism falsifier unchanged: `T_ex = Σ|ΔΔ̂| − Σ|ΔΔ°|` on identical paths. The mechanism is gamma
**error**, not gamma **level** — the differentiator from Sakuma 2026b.

---

## 5. Training parameterization — verified, and the mask facts

### 5.1 Conformance (verified against the frozen artifact)

- All five parameter ranges respected.
- Feller ratio ≥ **0.4061** across the sample, satisfying ≥ 0.40.
- All five anchor balls excised, min normalised distance **0.2310–0.3670** vs the required 0.10
  (2.3–3.7× margin), **zero** points inside any ball.
- Minimum sampled ρ = **−0.7989** against `strong_neg_corr`'s −0.90.

That last is load-bearing and belongs in the paper's methods: it makes the extrapolation claim for
`strong_neg_corr` literally true rather than merely intended.

**Frozen artifact:** 512 parameter points × 64 (S, K, τ) → **23,623 train / 5,936 val** retained
rows against the 5N = 20,480 floor (**PASS**, +15.3% headroom), overall mask rate **0.098**.

### 5.2 Mask-neutrality panels — post-fix, including the counter-movements

**(a) Feller bands.** The ADI band halved; every other band is **bit-identical**, the control
showing the change touched only the fourth leg.

| band | n | before | after |
|---|---|---|---|
| **[0.40, 0.60)** | 26 | 0.5192 | **0.2434** |
| [0.60, 1.0) | 71 | 0.0990 | 0.0990 |
| [1.00, 2.0) | 154 | 0.0896 | 0.0896 |
| [2.00, 4.0) | 137 | 0.0914 | 0.0914 |
| [4.00, ∞) | 124 | 0.0844 | 0.0844 |

The band remains elevated (0.2434 vs ~0.09) — excess down from 5.8× to 2.7×. Part is the
irreducible p99 tail of grid-differenced Greeks against analytic CF (`near_feller` shows the same
floor and does not improve under any refinement); part may be further `vmax` headroom, since the
hypercube reaches ξ = 0.60 at Feller down to 0.40. **Not chased, not floored — state it.**

**(b) KS, masked vs retained marginals — moved BOTH ways.**

| | before | after |
|---|---|---|
| κ | 0.0645 | 0.0287 |
| θ | 0.1787 | 0.1201 |
| ξ | 0.1284 | 0.0383 |
| **ρ** | 0.1242 | **0.1492** |
| **v0** | 0.0832 | **0.1266** |

θ and ξ improving is mechanical — Feller = 2κθ/ξ², so band-concentrated masking *is* a θ/ξ shift,
and the two are one fact, not two. ρ and v0 worsening is expected arithmetic: the masked set shrank
~12%, so its composition changed. Report it; do not present the fix as a uniform improvement.

**(c) |Γ| deciles.** Improved at every decile, including d10 (0.1221 → 0.1010), which is the one
that matters most for gamma supervision. d1 sits at ~0.70 and is unchanged — that is the
relative-tolerance artifact at |Γ| ≤ 1.1e-05, where `tol_rel = 1e-3` demands absolute agreement
near 1e-8. The contract's stated failure mode for check (c) is the rate **rising** with |Γ|; it
falls.

### 5.3 The three neutrality checks are not the three panels — NEW

A mapping the paper must get right or it will mis-describe its own evidence. The contract
(`oracle.masking.neutrality_checks`) declares:

1. "must NOT preferentially remove points where baseline PINN fails" → **panel (a)**
2. "retained-point distribution matched **across arms**" → **not panel (b)**
3. "must NOT preferentially retain smooth-agrees-with-smooth points" → **panel (c)**

Check 2 is **arm-independence** — structural, since the mask derives from the oracle consensus via
`mask_any` and `build_arm_labels` applies the same one to every arm. Panel (b) is a supplementary
statistic, not check 2. `mask_neutrality_report` states its own position: *"Deliberately NO
thresholds: report, human judges."* Neither the panels nor the checks auto-pass.

---

## 6. Pre-registered thresholds (v8)

Unchanged from v7 except where noted.

- **Confirmatory cell**: misspecified delta CVaR95 improvement ≥ 10% relative **and**
  paired-bootstrap 95% CI excludes 0. Headline scale-free number: fraction of the
  baseline-to-oracle-Δ gap closed.
- **Order attribution:** (+Δ+Γ) beats (+Δ) at the cell, CI excludes 0; else honest null.
- **Dose-response (Q2):** monotone via isotonic regression plus rank correlation. A non-monotone
  result is **"monotonicity not demonstrated"** — never "flat," never a "regularization null."
  The accompanying statistic is a **one-sided seed-bootstrap tail probability P(ρ ≤ 0) < 0.05**,
  which is **not a classical p-value** and must not be described as one.
- **OOD-param Γ RMSE** ≥ 15% reduction vs structural PINN; **Vega RMSE** ≥ 15% (secondary); price
  parity within 10%.
- **Mechanism falsifier** (unchanged): gap at 0% TC ⇒ robustness channel (i); gap ≈ 0 at 0% TC
  widening with TC and T_ex → 0 ⇒ cost channel (ii); if T_ex does not drop, channel (ii) is
  rejected mechanically whatever the PnL shows. Both outcomes publishable.
- **In-model hedging** is not pass/fail; it is the (in-model × cost) corner and must reproduce
  Sakuma's ≈ no-gap null as a consistency check, within `sakuma_null_rel_tol = 0.02`.
- **Verdict vocabulary (AM2-2):** `null` = NOT EVALUATED (artifact legitimately absent);
  `error` = evaluation attempted and FAILED (artifact present but corrupt). Never collapsed into
  one column of any table, memo, or figure.

---

## 7. Oracle-headroom gate — D3 RESOLVED

**Framing (AM3).** The gate is a **diagnostic that bounds the maximum detectable effect** on the
primary metric before compute is committed. v6's "gates whether the threshold is physically
achievable" is struck. `spread_threshold_rel = 0.10`.

**Procedure.** Hedge the confirmatory instrument twice — once with oracle Greeks, once with oracle
Greeks under the corruption field. The spread between the two PnL distributions bounds the
detectable effect.

**Declared deviations, both material:**

1. The corruption field is **anisotropic and S-dominated**, not isotropic, with an
   **amplitude-matched iid comparator** alongside. "Noise scaled to plausible PINN gamma error"
   understates this and must not appear in a draft.
2. The decision rule requires **every seed's paired CI to exclude zero**. A pooled CI is not
   sufficient.

### 7.1 The σ ladder — split, with production-scale measurements

| σ_rel | role | `clipped_frac` (production, mean [min, max]) |
|---|---|---|
| 0.05 | **decision** | 0.0027 [0.0024, 0.0029] |
| 0.10 | **decision** | 0.0737 [0.0712, 0.0756] |
| 0.15 | **decision** | 0.1484 [0.1449, 0.1516] |
| 0.20 | **diagnostic** | 0.2470 [0.2429, **0.2513**] — 3 of 10 seeds outside |
| 0.40 | **diagnostic** | 0.8137 [0.8102, 0.8194] |

`region_of_validity.clipped_frac_max = 0.25`, deliberately **not** raised to re-admit 0.20. A rung
whose decision-eligibility depends on which seeds were drawn is not a decision rung. Diagnostic
rows are swept, written and plotted exactly like decision rows and excluded **only** from the
decision scan; they must be labelled diagnostic wherever they appear.

### 7.2 The clip AMPLIFIES — a correctness requirement, not a nicety

AM3-1 measured that the clip **amplifies** the delivered gamma error across the decision band,
inverting the premise AM2-3 was written on:

| nominal σ_rel | 0.05 | 0.10 | 0.15 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| gamma amplification | 1.00 | 1.21 | 1.41 | 1.54 | 1.40 |
| delta counterpart | 1.00 | 1.05 | 1.16 | 1.28 | 1.34 |

So comparing the pilot against **nominal** σ_rel is **anti-conservative**: at nominal 0.15 the
delivered error is 1.41×, and a spread produced by a ~0.21-sized gamma error would be credited to
a 0.15-sized one, returning a GO on an effect the pilot cannot actually deliver.
`effective_sigma_reporting.compare_pilot_against = sigma_gamma_effective` is what prevents this.
The nominal decision band [0.05, 0.15] tops out at **σ_gamma_effective ≈ 0.21 × rms(Γ_oracle)**.

**Why a region of validity exists (the corrected reason).** Above the bound the corrupted arm is
not a weaker perturbation — it is a **structurally different object**: a saturated, bang-bang
hedger pinned at a delta bound over much of the state space, whose gamma error is the oracle's own
−Γ rather than the calibrated field. Its spread cannot be mapped back to a σ at all, because a
trained PINN's Greek error is smooth in state. Do not widen or remove the delta clip.

### 7.3 If the pilot lands outside

**INCONCLUSIVE — neither a pass nor a no-go.** An inconclusive gate **does not authorize training
spend** and must not be read as a failed gate. The ladder must be revisited and the clause
re-decided before any go decision.

**Extension contingency**, declared before any pilot ran: extension rungs **[0.20, 0.25]** enter as
**diagnostic**. A rung becomes decision-eligible only after `clipped_frac` is **re-measured at
production scale** (10,000 paths, the 10 confirmatory seeds, one gate run per seed) with **both**
its seed mean **and** its per-seed **maximum** at or below 0.25. The per-seed max is required
because that is exactly the test 0.20 failed. `clipped_frac_max` **stays 0.25**.

**Expected outcome, stated in advance because it is the likely one:** `clipped_frac` is monotone
non-decreasing in σ_rel and 0.20 already straddles the bound, so an extension will probably **not**
yield an eligible rung. If it does not, the honest reading is that the gate cannot bracket a pilot
that large *at this delta clip*, and the levers are the contract's own — **rebalancing frequency,
then misspecification severity. NOT the bound, NOT the clip, NOT a wider ladder.**

**Reporting.** An extended-ladder result is reported **as** one — which rungs were added, when each
was measured and promoted, and that the original ladder did not bracket the pilot. The headline
keeps the original ladder's answer (INCONCLUSIVE) alongside it.

---

## 8.–9. Unchanged from v7

Three-corner triangulation stands: A&T (gamma hedging = model-uncertainty robustness, at the level
of strategy choice, Greeks exact), Sakuma 2026b (gamma **level** → turnover concentration),
Noguer (gamma **error** → discretization bound, theory only). Our 2×2 plus turnover mediation plus
dose-response is the adjudication none of the three performs. The novelty matrix is unchanged; its
claim does not depend on the missing cell.

---

## B. Open items

| ID | Item | Owner | Blocking | Status |
|---|---|---|---|---|
| **D1** | Add `feedforward` (~$4, 5 seeds) or declare the interaction unidentified | Melvin | M4 dispatch; §4.2 wording everywhere | **OPEN — the only real decision left** |
| D1b | Drop `rung0_price_only` from dispatch (10 seeds, never hedged, config-identical to `standard_pinn`) | Melvin | M4 dispatch cost | OPEN — funds D1 twice over |
| D2 | ADI `vmax`, `nv` | — | §4.3 | **RESOLVED**: 1.6 / 481, now defaults, certified |
| D3 | σ-ladder, `clipped_frac_max`, extension contingency | — | §7 | **RESOLVED**: §7 above |
| D4 | Re-pull staging figures predating the 23,623-row artifact | Melvin | any draft quoting retention/mask rates | **RESOLVED in this memo**; drafts still need sweeping |
| C1 | Hainaut & Casas terminal-condition form | — | identical-ansatz rule | carried open |
| C2 | A&T "model uncertainty" scope | — | adjudication framing | carried open |
| C3 | Armstrong & Ionescu arXiv ID | — | camera-ready bibliography | carried open |

## C. Downstream sync required

`manuscript/main.tex` still encodes v6 and is inconsistent with v7 **and** v8. Sweep for:
"completed factorial" (three places); the `price-only NN` row in the arm grid and results
skeleton; `Δt = 0.003968`; "flat curve means any curvature regulariser would do"; the
"regularisation null" verdict marker; the v6 gate framing ("ceiling on any effect,"
"non-optional," retune-and-re-freeze); the ADI leg with no grid settings; and the Appendix A
verbatim acceptance block, which reproduces the **superseded** `dose_response` line byte-for-byte
and now contradicts Q2. That block must be replaced with the amended contract or carry a dated
amendment record beneath it — byte-fidelity to a superseded pre-registration is worse than no
block.

**New in v8, add to the sweep:** any claim that the ADI fix improved the boundary regime's error
bars (it did not — §4.5); any description of the four-leg oracle as three independent legs against
one (CF and FD are one family — §4.4); and any statement that the mask-neutrality fix was a uniform
improvement (ρ and v0 worsened — §5.2b).
