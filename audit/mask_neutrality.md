# Mask neutrality — is the oracle mask selection on difficulty, and does it touch the pre-registered OOD Greek thresholds?

**Scope.** Read-only on source. Everything below was measured by running the production
code paths unchanged; nothing is carried over from `audit/dataset_sizing.md`. Scripts and
their real outputs are in `audit/repro/` (r10, r11, r12, r13, r14, r15, r16).

---

## Answer up front

1. **The anchor grids ARE masked, and the mask IS applied at evaluation.**
   [eval_greeks.py:130](../eval_greeks.py#L130) — `keep = ~np.asarray(d["mask_any"], bool).ravel()`.
2. **Arm-vs-arm is safe.** Verified by execution on four deliberately different networks:
   bitwise-identical input tensors, identical `n_unmasked`, on every primary regime and
   slice ([r12](repro/r12_output.txt)). This is not at risk and is not reported as such.
3. **Cross-regime IS affected, and the two primary regimes are filtered in *opposite*
   directions.** `near_feller` masks **16.05 %**, `strong_neg_corr` **6.50 %** — a 2.5×
   difference. On `near_feller` the masked points are **5.2× rougher** in Gamma than the
   survivors; on `strong_neg_corr` they are **flat** (the mask there removes only degenerate
   near-zero-Greek states). The `near_feller` eval set is missing the high-curvature Gamma
   region; the `strong_neg_corr` eval set is essentially its full grid minus zeros.
4. **The mask-rate gap between the two primary regimes is driven by LEG ROUTING, not by
   difficulty.** `near_feller` gets the ADI fourth leg, `strong_neg_corr` does not. Within a
   single regime, at identical parameters, adding one leg (the MC subset) raises the mask
   rate 1.6–2.0× ([r16](repro/r16_output.txt)). So yes — the binding-regime selection in
   [analyze_results.py:733-735](../analyze_results.py#L733-L735) compares differently-filtered
   populations, and the filter strength is a function of how many legs voted.
5. **One sanity regime is de facto dropped.** `feller_violating_volvol` masks **68.7 %**, and
   **every tau slice from 0.68 to 1.00 has zero survivors** — the `tau_maturity_holdout`
   slice for that regime is EMPTY, so every metric on it is NaN. The contract's own note
   calls this regime the *"Vega worst-case long-tau slice — reported WITH error bars under
   F7 discipline"*, and F7 says `do_not: drop boundary regimes`. The mask drops exactly the
   part that carries the declared role.
6. **No number in the study is currently wrong**, and the mask cannot be removed — a masked
   point has no trusted label by construction. What is missing is **reporting**: the point
   counts never reach the aggregate tables, the verdict, or any exhibit, so a reader of the
   pre-registered result cannot see that `near_feller` was scored on 84 % of its grid and
   `strong_neg_corr` on 94 %.

A contract change is warranted, and it is a **reporting** change, not a change to the mask.
It is specified in [§7](#7-contract-change-if-warranted-not-made). Nothing was edited.

---

## 0. A correction to the task's premise

The task states *"The Feller band is also the near_feller OOD evaluation regime."* It is not.

| regime | Feller ratio | in `[0.40, 0.60]`? | 4th leg? | why |
|---|---|---|---|---|
| baseline | 1.78 | no | no | — |
| high_variance | 4.00 | no | no | — |
| **near_feller** | **1.04** | **no** | **yes** | named in `oracle.fourth_leg.required_for` |
| strong_neg_corr | 1.78 | no | no | — |
| **feller_violating_volvol** | **0.44** | **yes** | **yes** | both `required_for` AND the band clause |

`near_feller` sits at Feller 1.04, well outside the `[0.40, 0.60]` band whose 0.5211 mask
rate the sizing report measured. The regime that lands in that band is
`feller_violating_volvol` — a **sanity** regime, not a primary one.

The connection the premise was reaching for is real but runs through a different mechanism:
`near_feller` is in `oracle.fourth_leg.required_for`, so it gets the **ADI leg**, and more
legs means more chances to disagree. That is the mechanism, and §3d measures it.

This correction matters for the conclusion. Had `near_feller` been the band regime, the
worry would have been a ~52 % mask rate on a primary metric. The measured rate is 16.05 %,
and the sharp finding is not its size but its **shape** (§3) and the fact that the other
primary regime's mask has the **opposite** shape.

---

## 1. Are the anchor grids masked? — Yes, and the mask is applied

Written by the producer:

- [make_datasets.py:448](../make_datasets.py#L448) — `mask_any = np.any(np.stack([mask[g] for g in LABEL_QUANTITIES]), axis=0)`
- [make_datasets.py:459](../make_datasets.py#L459) — `"adi_leg": np.bool_(fourth), "mask_any": mask_any,` into `{regime}_grid.npz`
- [make_datasets.py:467](../make_datasets.py#L467) — per-Greek `mask_{g}` alongside it
- [make_datasets.py:462](../make_datasets.py#L462) — `_ANCHOR_GRID_KEYS` asserts `mask_any` can never be silently dropped from the artifact

**The line that decides it**, in the consumer:

```python
# eval_greeks.py:130-133
keep = ~np.asarray(d["mask_any"], bool).ravel()          # unmasked = legs agree
wing = (Sf / Kf < WING_LO) | (Sf / Kf > WING_HI)
sel = keep & (np.asarray(restrict, bool).ravel() if restrict is not None else True)
idx = np.flatnonzero(sel)
```

Every RMSE, quantile and `oracle_unc_rms` at
[eval_greeks.py:155-171](../eval_greeks.py#L155-L171) is computed over `idx` only. So the
answer is the first of the task's two branches: **masked**, and roughly one sixth of the
`near_feller` grid is gone. The second branch — evaluating against unvalidated labels — does
not occur.

This is the correct design. The alternative would score arms against a "consensus" that is
the median of legs known to disagree by more than `tol_rel`, i.e. against a number the
oracle itself refuses to certify. There is no third option: a masked point has no trusted
label, which is why §7 recommends a reporting change and not a mask change.

---

## 2. Mask rate per anchor grid — census, not sample

`audit/repro/r10_anchor_grid_masks.py` runs `make_datasets.generate_anchor_grids` **unchanged**
at production settings (contract grid 41×33×16 = 21 648 points per regime, seed 42,
`mc_subset_frac=0.10`, `mc_paths=200 000`, `near_feller_mc_multiplier=4`). Every point of
every regime is counted — this is a **census**, so there is no sampling error to quote.
Wall clock 424 s. → [`r10_output.txt`](repro/r10_output.txt)

```
                  regime  feller  adi  n_grid  mask_any    price    delta    gamma     vega    vanna
----------------------------------------------------------------------------------------------------
                baseline    1.78   no   21648    0.0456   0.0002   0.0004   0.0415   0.0293   0.0342
 feller_violating_volvol    0.44  yes   21648    0.6872   0.1464   0.1345   0.0110   0.5954   0.4292
           high_variance    4.00   no   21648    0.0023   0.0006   0.0005   0.0014   0.0012   0.0000
         strong_neg_corr    1.78   no   21648    0.0650   0.0026   0.0020   0.0463   0.0346   0.0516
             near_feller    1.04  yes   21648    0.1605   0.0007   0.0415   0.0837   0.0589   0.1252
```

Points actually scored, per regime × slice as `eval_greeks` slices them
([`r14_output.txt`](repro/r14_output.txt)):

```
                  regime     role slice  n_slice  survive    frac
                baseline   sanity  full    21648    20661  0.9544
                baseline   sanity  wing    11024    10298  0.9341
                baseline   sanity   tau     1353     1323  0.9778
           high_variance   sanity  full    21648    21599  0.9977
           high_variance   sanity  wing    11024    10995  0.9974
           high_variance   sanity   tau     1353     1353  1.0000
 feller_violating_volvol   sanity  full    21648     6771  0.3128
 feller_violating_volvol   sanity  wing    11024     2821  0.2559
 feller_violating_volvol   sanity   tau     1353        0  0.0000   <-- EMPTY: every metric NaN
         strong_neg_corr  PRIMARY  full    21648    20241  0.9350
             near_feller  PRIMARY  full    21648    18174  0.8395
```

Three things to take from this table.

- **The two primary regimes differ by 2.5× in mask rate** (0.1605 vs 0.0650). That is the
  cross-regime comparability question, answered in §3d and §4.
- **`feller_violating_volvol` retains 31 %.** Per-tau breakdown:
  `tau=0.62` keeps 43/1353, and **`tau ∈ {0.68, 0.74, 0.81, 0.87, 0.94, 1.00}` keep zero**.
  Six of sixteen tau slices are entirely gone. `eval_arm_on_regime` returns the `idx.size == 0`
  branch ([eval_greeks.py:135-138](../eval_greeks.py#L135-L138)) → `n_unmasked: 0` and NaN
  everywhere, which `_write_csv` renders as blank cells.
- **`high_variance` masks essentially nothing** (0.23 %), which is the useful control: the
  pipeline is not masking indiscriminately.

The sizing report's `[0.40, 0.60]`-band number (0.5211) and this table are consistent and
measure different things: that one is per hypercube *parameter point* under 4 legs; this one
is per *grid point* at one fixed parameter vector per regime. `feller_violating_volvol`
(Feller 0.44, in the band, 4-leg-routed) lands at 0.6872 here, the same order.

---

## 3. Is the mask selective on difficulty? — Yes, and the shape differs by regime

`audit/repro/r11_mask_selectivity.py` compares masked vs surviving populations on each grid;
`audit/repro/r15_masked_vs_kept_difficulty.py` adds two network-free difficulty proxies.
→ [`r11_output.txt`](repro/r11_output.txt), [`r15_output.txt`](repro/r15_output.txt)

**Stated caveat:** on a masked point the `consensus` is the median of legs that disagree, so
it is a noisier magnitude proxy than on a surviving point. It is still the best available
estimate, and every comparison below is a ratio of medians or of rms values, never a
point-wise claim.

**Feller ratio** is constant within an anchor grid (one parameter vector per regime), so
that dimension is cross-regime here and is measured within-sample on the hypercube in §6.

### 3a. The two non-ADI regimes: the mask removes degenerate zeros

`strong_neg_corr` (PRIMARY, cf+fd only on 90 % of points):

```
  mask rate by |consensus_gamma| decile (1=smallest): 1:0.328 2:0.055 3:0.167 4:0.055 5:0.009
                                                      6:0.002 7:0.001 8:0.004 9:0.005 10:0.024
  mask rate by tau slice: 0.04:0.681  0.10:0.072  0.17:0.270  then <=0.006 everywhere
  |consensus_gamma| masked  q25/q50/q75 = 2.4e-12 / 6.3e-11 / 2.5e-07
  |consensus_gamma| surviving q25/q50/q75 = 1.1e-05 / 2.2e-03 / 1.6e-02
```

The masked points have Gamma nine orders of magnitude below the survivors — they are the
deep-wing, shortest-tau states where the option value and every Greek collapse to machine
zero, `cross_validate`'s relative comparison degenerates, and the wing floor
(`wing_floor_frac = 1e-2` of the global max) takes over. `baseline` has the same shape
(decile 1: 0.332, decile 10: 0.003).

The roughness proxy confirms it: masked points on `strong_neg_corr` and `baseline` have a
**median `|d²γ/dS²|` of exactly 0.0000** — they sit on a flat, zero part of the surface.

### 3b. `near_feller` is different: the mask removes the ROUGH region

```
near_feller
  mask rate by |consensus_gamma| decile: 1:0.264 2:0.218 3:0.083 4:0.086 5:0.256
                                         6:0.262 7:0.242 8:0.125 9:0.045 10:0.025
  mask rate by moneyness bin: [0.00,0.75): 0.183  [0.75,0.90): 0.515  [0.90,1.00): 0.119
                              [1.00,1.10): 0.011  [1.10,1.30): 0.063  [1.30,inf): 0.068
  mask rate by tau slice: 0.04:0.829  then 0.09-0.21 across every other slice
  gamma roughness |d2gamma/dS2| (normalised): masked median 0.0249, kept median 0.0048,
                                              ratio 5.22x
```

Three separate signals, one story:

- the decile profile is **not** monotone in the way the zero-collapse regimes are — deciles
  5, 6 and 7 (mid-to-large Gamma) are masked at ~25 %, so genuinely large-Gamma states are
  being removed at a quarter of their population;
- **51.5 %** of the `S/K ∈ [0.75, 0.90)` band is removed — a contiguous just-OTM strip, not
  scattered points;
- masked points are **5.2× rougher** in Gamma than survivors.

This is the sharp finding the task asked about, and it points the way the task anticipated:
**the `near_feller` eval set is systematically missing the high-curvature part of the Gamma
surface.** Not missing the largest-|Gamma| points (decile 10 is retained at 97.5 %) — missing
the points where Gamma *changes fastest*.

The mechanism is legible. `near_feller` carries the ADI leg, a finite-difference PDE solve
whose discretisation error is largest exactly where the solution's curvature is largest. So
ADI-vs-CF disagreement tracks surface roughness. The non-ADI regimes have no such leg: their
two "independent" legs are CF and **finite-difference on the CF price grid** (contract
`three_way_validation.leg2`), which are highly correlated and agree everywhere except where
the price itself degenerates.

### 3c. `feller_violating_volvol`: the mask removes the large-Vega states

```
feller_violating_volvol
  rms(consensus) kept / all:  price 0.892, vega 0.900, vanna 1.311, gamma 1.184, delta 1.067
  masked share of sum(c^2):   vega 0.7467, price 0.7512, delta 0.6441, gamma 0.5616, vanna 0.4624
  mask rate by tau slice: rises monotonically from 0.13 at tau=0.10 to 1.000 for tau >= 0.68
```

Here the direction reverses: **75 % of the regime's total squared Vega is on masked points**,
and the surviving set's Vega rms is 10 % *below* the full grid's. This is the "eval set is
systematically missing large-Vega states" case, and it lands on the one regime the contract
explicitly designates as carrying the Vega worst case.

### 3d. Controlling for difficulty: the mask rate tracks LEG COUNT

The regimes above differ in parameters as well as in leg routing, so difficulty and leg count
are confounded across the table in §2. `audit/repro/r16_leg_count_effect.py` breaks the
confound **inside** a single regime. `generate_anchor_grids` folds the MC leg in on a
stratified 10 % row subset only ([make_datasets.py:436-447](../make_datasets.py#L436-L447)),
so each grid contains two populations at identical parameters, identical geometry, differing
only in whether one more leg got a vote. → [`r16_output.txt`](repro/r16_output.txt)

```
                  regime  adi   n_mc  mask|mc  n_nomc mask|no mc   ratio
                baseline   no   2176   0.0813   19472     0.0416     2.0
           high_variance   no   2176   0.0216   19472     0.0001   210.3
 feller_violating_volvol  yes   2176   0.6953   19472     0.6863     1.0
         strong_neg_corr   no   2176   0.0997   19472     0.0611     1.6   PRIMARY
             near_feller  yes   2176   0.1811   19472     0.1582     1.1   PRIMARY
```

Adding one leg, at fixed parameters, raises the mask rate 1.6–2.0× on the regimes that have
room to move. The cleanest per-Greek reading is on `strong_neg_corr`, whose non-MC population
is cf+fd only: `price` and `delta` mask at **exactly 0.0000** there and jump to 0.0257 and
0.0198 the moment the MC leg arrives. CF and finite-difference-on-CF simply never disagree on
price or delta; the MC leg is the first genuinely independent vote, and it is the one that
masks. (On `near_feller` the non-MC population already contains ADI, so its `delta` baseline
is 0.0409 rather than 0 — the ADI leg is doing there what MC does on `strong_neg_corr`.)

The same mechanism explains the gap between the two primary regimes. Their 2-leg (cf+fd)
populations sit at 0.1582 (`near_feller`, which also has ADI) and 0.0611 (`strong_neg_corr`,
which does not). Strip the ADI leg from `near_feller` and its mask rate would be of
`strong_neg_corr`'s order; the 2.5× gap is essentially the ADI leg, i.e. a decision made in
`oracle.fourth_leg.required_for`, not a measurement of which regime is harder.

---

## 4. Does `eval_greeks` compute RMSE over survivors only, and can mask rate drive the binding regime?

**Yes to the first**, per §1. Every metric is over `idx = flatnonzero(keep & restrict)`.

**On the second — plainly: yes, the binding-regime selection could be driven by mask rate
rather than by difficulty, and the two are not separable as the pipeline currently stands.**

The chain:

- [eval_greeks.py:169](../eval_greeks.py#L169) records `n_unmasked` per row;
- `PERSEED_COLS` ([eval_greeks.py:179](../eval_greeks.py#L179)) carries it into
  `ood_param_greeks.csv`;
- **`AGG_COLS` ([eval_greeks.py:185-186](../eval_greeks.py#L185-L186)) does not.** The seed
  aggregate keeps only `n_seeds`;
- `THRESHOLD_COLS` ([eval_greeks.py:188-195](../eval_greeks.py#L188-L195)) does not;
- [analyze_results.py:713](../analyze_results.py#L713) builds the verdict LUT from the
  **agg** CSV, so no point count reaches the verdict;
- [analyze_results.py:733-735](../analyze_results.py#L733-L735) selects the **binding**
  regime as the one with the smallest Gamma reduction, and
  [analyze_results.py:742-746](../analyze_results.py#L742-L746) requires **every** primary
  regime to pass;
- [exhibits.py:36](../exhibits.py#L36) reads only `ood_param_greeks_agg.csv`. And
  `uncertainty_table.csv` — the one artifact that does carry a per-regime, per-Greek
  `mask_rate` ([make_datasets.py:476](../make_datasets.py#L476)) — **has no consumer anywhere
  in the repo** (grep: written, never read).

So the number that decides the OOD Greek verdict is a min over two regimes whose scored
populations are 18 174 and 20 241 points, filtered in opposite directions with respect to
Gamma curvature, and nothing downstream records that.

**What is and is not measurable right now.** No arm is trained, so the *size* of the effect
on the reduction statistic cannot be measured — and it is worth being precise about why the
direction is nonetheless predictable. `reduction_vs_standard_pinn = 1 − RMSE_arm/RMSE_base`
is computed on the **same** point set for both arms, so the mask cancels exactly unless the
arm/baseline error **ratio** differs between masked and kept points. On `near_feller` the
masked points are the high-curvature ones, which is precisely where explicit Gamma
supervision should help a network most relative to a PDE-only baseline. Removing them should
therefore **shrink** the measured Gamma reduction on `near_feller`. That is conservative with
respect to the study's hypothesis — it biases toward the pre-registered null, not toward a
false positive — but it also makes `near_feller` more likely to be the binding (minimum)
regime for a reason that is about oracle leg routing rather than about model performance.

This paragraph is **inference from the measured filter shape, not a measurement.** It becomes
measurable once arms exist, and the check is cheap: recompute the reduction on
`keep ∪ masked` and see whether the arm/baseline ratio moves. That check scores against an
uncertified consensus, so it is a **diagnostic**, never a reportable metric.

Two secondary points, both real but neither verdict-bearing:

- **`rel_rmse` is not comparable across regimes.** `rms(consensus_gamma)` on kept vs all
  points is 1.0718× on `near_feller` and 1.0152× on `strong_neg_corr` — the denominator
  shifts by a different amount in each regime. The pre-registered threshold uses the
  *absolute*-RMSE reduction, so no verdict depends on this; the `rel_rmse` columns in the
  tables do.
- **`price_parity`** is a within-regime ratio of the arm's and baseline's `rel_price_rmse`,
  so the mask and the denominator cancel. Unaffected.

---

## 5. Is the mask arm-independent? — Confirmed by execution

`audit/repro/r12_mask_arm_independence.py` runs the real `eval_greeks.eval_arm_on_regime` on
four deliberately different networks — `standard_pinn` @ init 0, `standard_pinn` @ init 1,
`rung3_delta_gamma_vega` @ init 2, and that rung3 model with **every weight scaled ×50** —
and captures, from inside each call, the exact tensor the model was queried on.
→ [`r12_output.txt`](repro/r12_output.txt)

```
     near_feller / full : n_unmasked= 18174  identical_inputs=True  identical_n=True
                          models_differ=True  equals_~mask_any&restrict=True  -> OK
      gamma rmse per model: standard_pinn/init0=0.02134, standard_pinn/init1=0.0213,
                            rung3/init2=0.02136, rung3/init2 x50=3.16e+05
 strong_neg_corr / full : n_unmasked= 20241  identical_inputs=True  identical_n=True
                          models_differ=True  equals_~mask_any&restrict=True  -> OK
 ... (same on the wing and tau slices of both primary regimes)

ARM-INDEPENDENCE: CONFIRMED
```

Four checks per cell, all passing: the captured input tensors are **bitwise** equal across
all four models; `n_unmasked` is identical; the models genuinely differ (the ×50 model's
Gamma RMSE is seven orders of magnitude off, so the invariance is not vacuous); and the
selection equals `flatnonzero(~mask_any & restrict)` recomputed independently from the npz,
i.e. it is a pure function of the frozen artifact with no model input.

Incidental confirmation from the first run of this probe: `standard_pinn` and
`rung3_delta_gamma_vega` built at the **same** torch seed produce bit-identical Gamma RMSEs,
because arms differ only by loss flags and not by architecture — the contract's "identical
architecture, identical ansatz across arms" invariant, observed rather than asserted. The
script therefore gives the rung3 model its own init seed so the "models differ" guard tests
the mask rather than that identity.

**This is the property that protects the confirmatory contrast, and it holds.**

---

## 6. The training-side mask, for completeness

`audit/repro/r13_hypercube_mask_selectivity.py` runs `make_labels.generate_labels` unchanged
on 148 production parameter points — **all 20** production Feller-band points plus a random
128 non-band points, at `n_skt=64`, seed 42, production leg kwargs. 529 s.
→ [`r13_output.txt`](repro/r13_output.txt)

```
  Spearman rho(feller, point mask rate) = -0.3346  (p=3.23e-05, n=148 points)
  feller [0.40, 0.6): n=  20  mask rate 0.5227
  feller [0.60, 1.0): n=  20  mask rate 0.0602
  feller [1.00, 2.0): n=  41  mask rate 0.0755
  feller [2.00, 4.0): n=  46  mask rate 0.0645
  feller [4.00, inf): n=  21  mask rate 0.0632

  mask rate by |consensus_gamma| decile: 1:0.655 2:0.120 3:0.122 4:0.078 5:0.076
                                         6:0.042 7:0.036 8:0.030 9:0.035 10:0.092
```

- The Feller dependence is **entirely the band**: 0.5227 inside `[0.40, 0.60]`, and flat at
  0.060–0.076 across every band above it. The negative Spearman is that one step, not a
  gradient. (Independent confirmation of the sizing report's 0.5211 ± 0.0522 band census and
  0.0685 plain rate, measured on a different point set.)
- Re-weighted to the true production mix (band weight 0.0446, not this probe's 0.135) the
  composite is **0.0875** — consistent with the sizing report's 0.0893 ± 0.0030.
- Contract neutrality check (c) — *"rate rising with |gamma| would indicate preferential
  retention of smooth-agrees-with-smooth points"* — comes out the **safe** way: the rate
  falls with |gamma|, from 0.655 in the smallest decile to 0.030–0.092 above it. The mild
  uptick in the top decile (0.092) is worth a human eye at freeze time but does not have the
  shape the check is written against.

The training mask is applied identically to every arm ([train_pinn.py:267-268](../train_pinn.py#L267-L268),
[make_labels.py:328-329](../make_labels.py#L328-L329)), so it is a statement about what all
arms learn from, not a comparability threat.

**A structural gap worth naming:** `mask_neutrality_report`
([make_labels.py:231](../make_labels.py#L231)) is called only from `generate_train_val`
([make_datasets.py:288](../make_datasets.py#L288)). `generate_anchor_grids` never calls it.
The contract's three neutrality checks are therefore run on the **training** artifact — where
the mask is arm-neutral by construction and the risk is lowest — and never on the
**evaluation** grids, where the mask actually filters the primary metric.

---

## 7. Contract change, if warranted (NOT made)

Warranted, and it is about **reporting**, not about the mask. The mask itself is correct and
cannot be relaxed: a masked point's label is exactly the number that failed cross-leg
validation, so there is nothing to score against. Three items, in priority order.

**(C-M1) Carry the scored-point count into the aggregate, the verdict, and the exhibit.**
`oracle.masking` currently declares the rule and three neutrality checks but says nothing
about reporting the mask's *extent* alongside a metric computed under it. Proposed addition
under `oracle.masking`: a `reporting` clause requiring that any table or verdict carrying an
OOD Greek metric also carry the number of points it was computed on, and the per-regime mask
rate. Code consequence (a separate change, not made here): add `n_unmasked` to `AGG_COLS` and
`THRESHOLD_COLS`, and surface the per-regime mask rate in the F7 exhibit — which also gives
`uncertainty_table.csv` its first consumer.

**(C-M2) Declare how a regime whose evaluation population is largely removed is reported.**
`reporting_discipline_F7.do_not: drop boundary regimes` is currently satisfied in form —
`feller_violating_volvol` is scored — while 68.7 % of it, and 100 % of the six longest-tau
slices, are removed by the mask. Under `acceptance_thresholds.verdict_vocabulary`, that empty
tau-holdout slice is a **`null`** (not evaluated, artifact legitimately absent), not an
`error` and not a result; `_write_csv` currently renders it as a blank cell, which is neither.
Proposed: state in the contract that a slice with zero surviving points is reported as
`null` with its point count, and that the regime's declared role — *"Vega worst-case long-tau
slice"* — is **not evaluable** at the current `agreement_tol_rel` and ADI settings. That is
an honest null about the oracle's reach, and it should be stated before results, not
discovered in a table.

**(C-M3) State that cross-regime OOD comparison is not like-for-like, and that the binding
regime is selected across differently-filtered populations.** The rule at
[analyze_results.py:733-735](../analyze_results.py#L733-L735) takes the minimum Gamma
reduction across primary regimes. §3d shows the filter strength is substantially a function of
leg routing (ADI present or not; +1 leg raises the mask rate 1.6–2.0× at fixed parameters —
[`r16_output.txt`](repro/r16_output.txt)), so the min is taken over populations selected by
the oracle's configuration. Proposed: a note under `splits.heldout_greek_and_hedging` that
per-regime OOD metrics are computed on regime-specific surviving sets whose sizes must be
reported, and that the binding regime is not, on its own, evidence of relative difficulty.

Optionally, and separately from the contract: run `mask_neutrality_report` (or an anchor-grid
analogue of it) inside `generate_anchor_grids` before the grids are frozen, so §3's numbers
exist as an artifact rather than as an audit script. `generate_train_val` already prints
*"Review mask_neutrality_report.md, then a HUMAN copies this directory to data/frozen"*;
`generate_anchor_grids` prints only the freeze reminder, with nothing to review.

---

## 8. What is NOT wrong

Stated plainly, because a clean answer is a result:

- **Arm-vs-arm comparison is sound.** §5, verified by execution, not by reading.
- **The confirmatory cell, order attribution, dose-response, mechanism adjudication and the
  oracle-headroom gate are untouched.** They are hedging-PnL statistics computed from path
  banks and providers; the anchor grids do not enter them.
- **`price_parity` is mask-invariant** (within-regime ratio of ratios).
- **The training/validation mask is arm-neutral and passes the contract's own neutrality
  check (c)** in the direction the check was written to catch (§6).
- **The masking rule itself is correct**, and the decision to score only on validated points
  is the right one. The alternative — scoring against an uncertified consensus — would be
  worse.
- **`high_variance` masks 0.23 %**, confirming the pipeline is not masking indiscriminately.

---

## 9. Measured vs assumed

**Measured** (executed on this machine, production code paths, outputs in `audit/repro/`):

| Quantity | Value | Where |
|---|---|---|
| Anchor grids carry `mask_any` and eval applies it | code + census | §1, r10 |
| Mask rate per regime × Greek, full contract grid | **census**, 21 648 pts/regime | r10 |
| Surviving points per regime × scored slice | census | r14 |
| `feller_violating_volvol` tau ≥ 0.68 → 0 survivors | census, per-slice | r14 |
| Masked vs surviving moneyness / tau / \|consensus\| distributions | census | r11 |
| Mask rate by \|consensus\| decile, per regime | census | r11 |
| Gamma-roughness ratio masked/kept (5.22× on `near_feller`) | census, interior pts | r15 |
| Masked share of Σ(consensus²) per regime × Greek | census | r15 |
| Leg-count effect at fixed parameters (MC subset vs not) | census | r16 |
| Arm-independence: bitwise-identical inputs, 4 models | execution | r12 |
| Hypercube mask rate vs Feller band | 148 pts, **all 20** band pts censused | r13 |
| Contract neutrality checks (a)(b)(c) on the hypercube | 148-pt probe | r13 |

**Assumed** (stated, not measured):

1. **Seed 42 and the repo's default leg kwargs.** The anchor grids are deterministic given
   `(seed, grid, leg_kwargs)`; the MC subset and its seeds are seed-dependent. A different
   freeze seed redraws the 10 % MC subset — §3d shows that subset is where the extra masking
   lands, so the *composition* would shift slightly. The CF/FD/ADI mask, which dominates,
   is seed-free.
2. **`mc_subset_frac = 0.10` and `mc_paths = 200 000`** — the `make_datasets` defaults, not
   contract-declared. Raising either raises the mask rate (§3d measures the direction: more
   legs, more masking).
3. **The direction of the effect on the Gamma-reduction statistic** (§4) is inferred from the
   measured filter shape, not measured. No arm is trained yet.
4. **The roughness proxy** (second difference of `consensus_gamma` along S, normalised by the
   grid's rms Gamma) is a stand-in for "hard for a smooth approximator". It is not the only
   possible proxy; it was chosen because it needs no network. The oracle-uncertainty proxy in
   r15 is reported alongside it and is less discriminating (ratios 0.2–2.0 across regimes,
   no consistent sign).
5. **The hypercube probe's 128 non-band points** are a random subset of the 428 production
   non-band points at n=448; the 20 band points are a census. The re-weighted composite
   (0.0875) is reported against the sizing report's independent 0.0893 as a consistency
   check, not as a new estimate.

**Reproduce** (total ≈ 25 min single-core; nothing is written outside the scratch dir, and
`generate_anchor_grids` refuses any path containing `frozen`):

```bash
A=<scratch>/anchors_v6
python audit/repro/r10_anchor_grid_masks.py        $A     # ~7 min — builds the grids + census
python audit/repro/r11_mask_selectivity.py         $A     # seconds
python audit/repro/r12_mask_arm_independence.py    $A     # ~2 min
python audit/repro/r14_slice_survival.py           $A     # seconds
python audit/repro/r15_masked_vs_kept_difficulty.py $A    # seconds
python audit/repro/r16_leg_count_effect.py         $A     # seconds
python audit/repro/r13_hypercube_mask_selectivity.py <scratch>/r13 128   # ~9 min
```
