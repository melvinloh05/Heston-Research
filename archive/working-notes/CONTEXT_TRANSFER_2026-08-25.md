# Context transfer — Sobolev-PINN paper, as of 2026-08-25

You are picking up a NeurIPS 2026 workshop submission. You have the repo and the
frozen results. This file is the handover: the standing rules, the claim discipline,
what is settled, what is open, and what was tried and dropped. Read it before
touching `paper.tex`.

---

## 1. What the project is

One pre-registered causal study. Question as registered: does explicitly supervising
the second derivative (with the second-state derivative) of a trained, PDE-retaining,
parametric Heston PINN improve delta-only hedging PnL under misspecified dynamics,
and through which channel? Thresholds, arms, evaluation regimes and the mechanism
falsifier were all frozen before training. `heston_benchmark_v6.yaml` is the single
source of truth and is READ-ONLY.

**The registered answer is no.** Every registered efficacy endpoint failed. The paper
is now about why the standard scorecard could not have found an effect, plus what
supervision demonstrably does buy.

**The paper's one idea (locked by the user on 2026-08-25):** derivative supervision
buys large off-distribution derivative accuracy that the conventional grid metric
confirms, and none of it reaches the downstream task, because the metric measures in
the wrong place. Title:

> *Sobolev Training Improves Off-Distribution Derivatives but Not Downstream Task
> Performance*

Two alternative spines were offered and rejected: "metric-first" (grid RMSE
overstates what supervision buys) and "robustness-first" (supervision stabilises
against the residual weight). 

---

## 2. Current state of the deliverable

`paper.tex` at repo root. Official NeurIPS 2026 workshop format,
`\usepackage[dblblindworkshop]{neurips_2026}`. Backup of the prior draft is
`docs/paper_backup_2026-08-25.tex`; the pre-rewrite draft is
`docs/paper_draft_2026-08-19.tex`.

```
Introduction                      (4 paragraphs, no results, fig:problem + fig:pipeline)
Methodology
Results
   4.1 Held-out derivative accuracy                 (tab:greeks)
   4.2 Pre-registered downstream outcomes           (tab:verdicts)
   4.3 Sensitivity to the residual weight           (fig:lambda)
   4.4 Where the derivative error is measured       (fig:coverage, tab:metric)
   4.5 Reconstruction from the supervised fields
Conclusion
Appendices A-D: acceptance block + amendments / full configuration /
                cost profile + dose-response + exhibits / oracle validation
```

Measured: **3,781 prose words, 7 body floats, 21 references.** Estimated 7.9-9.1
pages depending on assumed density. **The file has never been compiled** (see §9).

Test suite: `284 passed` under `/usr/bin/python3 -m pytest -q`.

---

## 3. Standing rules from the mentors — these hold for EVERY draft

Three separate advisers have given feedback. Their instructions are cumulative and
none has been withdrawn.

### 3.1 Narrative and framing

- **One standard idea per paper.** Do not write the chronology. No "we assumed X,
  we got negative results, so we pivoted to Y". Reverse-engineer the narration from
  the final inference and present it forward.
- Negative results are welcome and must be **honestly admitted**, but as evidence,
  not as a plot twist.
- **Do not include everything that was performed.** Exploratory analysis belongs out
  of the body. The bar is "matching and important for the paper", not "we did it".
- Frame as a **scientific machine learning paper tested on a finance-domain dataset**,
  not a finance paper. The A*STAR reviewers (former chief AI officer + two senior
  researchers) said the workshop audience has no finance background. Finance jargon
  was measured down from 365 to ~107 domain-term hits; keep it there.
- The paper's spine is **derivative supervision**. Be careful and sparing with the
  word "controller"; it should not become the subject of the paper.
- The impact and the reason the problem matters must be visible early, with figures
  carrying it.

### 3.2 Structure and format

- Standard Abstract / Introduction / Methodology / Results / Conclusion.
- **Abstract 14-15 lines maximum.** Shape: 2-3 lines problem statement, 2 lines
  prior approaches, 2-3 lines what we did, close with qualitative and quantitative
  results. Only the numbers that match the title. Not every number.
- **Introduction 4 paragraphs maximum.** Contributions and related work only.
  **No results in the introduction, and few numbers.**
- **No em-dashes anywhere.** Currently zero in the body; keep it that way.
- Formal tone, never conversational.
- **Subheadings must be formal and precise, not claim-shaped.** "Held-out derivative
  accuracy", not "What derivative supervision delivers". Minimise their number.
- Results should be carried by **tables and figures, not heavy text**.
- Tables: **highlight the best value** in each column, in the direction the metric
  runs (bold the lowest where lower is better), and say so in the caption. **Tables
  must not overflow the text block** — reviewers penalise this.
- Follow the page limit exactly.
- A **pipeline / methodology diagram** must appear right after the Introduction.
- Captions are good and should stay substantive.
- Notation must be consistent and correct throughout (the `$v_0$` slot was flagged).
- **Cite every equation, algorithm and technical term that is used but not derived.**
- Hyperlinks stay clickable but must not render coloured or boxed.

### 3.3 References

- **At least 20 references.** Currently 21.
- **No fabricated or hallucinated references.** Every entry must be checked. This is
  a hard rule; the correct response to an unverifiable citation is to flag it, not
  to guess. Precedent: the minimum-variance citation was left as an explicit
  `[CITATION NEEDED]` marker for days rather than invented, then resolved only after
  the source was verified.

### 3.4 Code and integrity

- The codebase will be released on acceptance. Result values and code logic must be
  correct; no mistakes in the numbers.
- Author contributions statement stays.
- Check whether the workshop mandates the NeurIPS paper checklist and include it if so.

### 3.5 The one about tone

An adviser said the draft reads as though the whole content and experiments were fed
to an agent and the result was pasted in, and that this must not be what the work
portrays. Concretely: kill claim-shaped section titles, break up parallel triads, cut
rhetorical set-ups. **The user must read the final draft end to end personally.** Do
not tell them this is fully solved; it is the one item that cannot be finished from
inside the loop.

### 3.6 From the earlier "story" adviser, still binding

- The λ_pde result is the **best** result in the paper, not bad news. Present it as a
  level claim.
- **Stop quoting the third rung's improvement as a percentage** — for hedging
  contrasts. This does *not* extend to accuracy reductions against the undamaged
  control, which survive as percentages.
- Answer the "you trained on oracle Greeks so of course you match oracle hedging"
  objection **early**, not buried. It is currently answered in the Introduction and
  in Methodology's scope conditions.

---

## 4. Claim discipline — non-negotiable

The header comment block of `paper.tex` carries this list. Do not remove it.

### Sentences the paper MUST NOT contain

1. "Explicit Gamma supervision improves delta-only hedging PnL under misspecified
   dynamics." — the registered claim; it failed.
2. "The improvement operates through the model-uncertainty-robustness channel" (or
   the transaction-cost variant) — both channel readings were falsified.
3. "The confirmatory test would have passed at a correctly calibrated
   transaction-cost tier" — any phrasing that re-scores the registered fail.
4. "The hedging and accuracy axes move independently" — artifact of disjoint
   measurement regions; they agree when measured in one region.
5. "The value-only network cannot build the minimum-variance controller" — measured
   false. It recovers 86% of the gain and falls short by 3.40%.

### Numbers that must never be quoted unqualified

- **"+31.5%"** as the effect size. It is a readout of λ_pde, not a treatment property.
- **"closes 98.6% of the gap"** — wrong denominator. Against the controller that
  actually wins it is 69.8%.
- The **91.9 / 6.5 / 1.7** ladder shares — shares of the damage at a dead λ_pde.
- The **0.575%** cost tier — reverse-engineered, appears in no artifact.

### Verified-record rule

Cite nothing that is not in the verified project record or independently checked. If
a claim cannot be sourced, mark it and say so. Two adviser assertions were measured
and **refused** on this basis: "the price-only network can't build Δ_MV" (false) and
"a hedge the oracle can't build" (wrong in form — `mv_oracle` IS oracle-built). An
adviser's "roughly 37×" regret ratio does not reproduce; the correct value is 29.2×.

---

## 5. Load-bearing numbers and where they come from

| Claim | Value | Source |
|---|---|---|
| Confirmatory endpoint | $+0.02\%$ vs $10\%$ bar, pooled CI $[-0.0242,+0.0242]$ | `results/analysis/.../threshold_verdicts.csv` |
| Order attribution (rung 2 vs 1) | $+0.0574$, CI $[+0.0527,+0.0621]$, $1.2\%$ rel, below the $2\%$ floor, harmful in sign | same |
| Held-out 2nd-deriv rel RMSE, supervised | $0.045$ | `results/eval_greeks_full/ood_param_greeks_agg.csv` |
| Held-out 2nd-deriv rel RMSE, residual-only | $0.410$ | same |
| Curvature penalty, no labels | $0.982$ — worse than adding nothing | same |
| Shuffled labels | $0.408$ — indistinguishable from nothing | same |
| More value data, no labels | $0.699$, flat under doubled width | `results/eval_greeks_infomatch/` |
| λ sweep, supervised arm | $2.0758$–$2.0905$ across $100\times$, oracle $2.0794$ | `results/lambda_sweep/`, echoed by `paper_figures.py` main() |
| λ sweep, residual-only arm | $2.1370 \to 3.0518$ | same |
| Loss decomposition at registered weight | residual $53.3\%$ of supervised objective vs $46.2\%$ derivative terms; $42.8\%$ of untreated | `results/reanalysis/loss_decomposition_agg.csv` |
| Repair share | $93.6\%$ of the registered contrast | `results/lambda_sweep/` |
| Residual × supervision interaction | $15.7\times$ | same |
| Supervision vs undamaged control | $0.0611$ loss units, $2.9\%$, CI $[-0.0672,-0.0588]$ | same |
| Metric comparison (2nd deriv) | Spearman $\rho$ grid $0.468$ / region $0.507$ / occupancy $0.707$ | `results/reanalysis/metric_comparison.csv`, `occupancy_metric_agg.csv` |
| Concordance, pooled | $0.733$ region vs $0.710$ grid | `results/reanalysis/metric_validation_agg.csv` |
| Regret decomposition | total $0.4261$; approximation $0.0141$; $96.7\%$ control design; $29.2{:}1$; inverts to $70.8\%$ for untreated | `results/reanalysis/regret_decomposition.csv` |
| MV oracle vs exact reference | $1.670$ vs $2.082$, $19.8\%$ better misspec, $10.6\%$ in-model | `results/mv_delta_full/` |
| MV reconstruction from arms | rung3 $1.6701$ (CI covers zero vs analytic $1.6696$); rung2 $1.6563$ $+0.80\%$; rung1 $-3.27\%$; value-only $-3.40\%$; residual-only $-42.9\%$ | `results/mv_supervised/` |
| Slope deficit | regression slope $0.637$; under-trades reference by $38\%$ | `results/reanalysis/` |
| Residual normalisation | rms $0.527$ vs dominant term $4.674$ $\Rightarrow$ $\approx 79\times$ inflation | Methodology, appendix |
| Cost tier error | registered $1\%$ is $\approx 200\times$ the calibrated rate | Armstrong & Tatlow 2024 |
| Mask asymmetry in decision region | $1.30\times$ the outside rate | `results/reanalysis/mask_hedgebox.csv` |

---

## 6. Reviewer attacks and the standing answers

1. **"You trained on oracle derivatives, so of course you match oracle hedging."**
   The evaluated regimes are excised from training *and* validation by 10%
   relative-radius balls; one lies outside the sampled range entirely; the decision
   trajectory starts 0.011 and ends 0.169 from the nearest labelled point and never
   passes a labelled point at the traded strike. The arms without labels fail at
   exactly those points. Answer early, in the Introduction.

2. **"Did you rig the baseline?"** The residual damage is our normalisation, and we
   say so. Section 4.3 states both selection failures against ourselves, including
   the one a finer grid cannot repair. Do not soften this.

3. **"Your effect is a hyperparameter artifact."** Correct, and the paper says it
   first. That is why the result is reported as a level, not a contrast.

4. **"n=1 domain."** Conceded explicitly: instrumented existence proof with a
   mechanism and a protocol, not a claim about surrogates in general.

5. **"Your recommendation is untested."** It is tested, and it nearly failed — the
   region restriction bought almost nothing, which is why the recommendation is the
   occupancy estimand and the region is named as a crude proxy. This self-refutation
   is a credibility asset. Keep it.

6. **Oracle independence.** Two of the four legs share machinery. Stated in
   Methodology; do not overclaim "four independent witnesses".

---

## 7. Solved

- Spine, title, section order, subsection titles.
- Abstract cut to ~14 lines in the mandated shape.
- Introduction at 4 paragraphs, no results, few numbers.
- `\hypersetup{hidelinks}` — links clickable, no coloured boxes.
- `\workshoptitle` filled with the real workshop name.
- **Figure 1** (`f0_problem.png`): held-out slice, matched values and mismatched
  curvature. Evaluates frozen checkpoints, not a schematic.
- **Figure 2** (`f8_pipeline.png`): the pipeline diagram the adviser asked for.
- `tab:greeks` rebuilt: six columns dropped, best values bolded, derivative notation
  instead of Greek letters.
- `tab:verdicts` shortened and wrapped in `\resizebox` so it cannot overflow.
- `tab:metric` bolds the winning estimand.
- Figure defects fixed: duplicate bar colour, annotation drawn outside the axes,
  legend sitting on the data, a series running off the bottom of a panel, and a
  direct label anchored to the wrong series (a real bug — the loop variable leaked).
- References 13 → 21, every one verified against a real source; both `[VERIFY]`
  flags resolved; the `[CITATION NEEDED]` marker resolved to Föllmer & Schweizer 1991.
- Missing attributions added for CVaR, the QE simulation scheme, Craig–Sneyd, the
  cosine expansion, and the PINN loss-balancing literature.
- Arm naming unified: "value-only", "residual-only", never "price-only" or "PINN".
- Zero em-dashes; no forbidden claim in live text; no undefined refs; no duplicate
  labels; balanced environments and braces.
- `paper_figures.py` regenerates all eight figures from one entry point. Fixed a real
  bug: the `__main__` guard sat above four later definitions so `main()` could never
  dispatch them.
- Suite green at 284.

---

## 8. Not solved — do these

1. **Page limit is unknown.** Not published on the workshop site
   (https://eethanshi.github.io/stochastic-dynamics-2026/) or on the workshop
   tracker. It is on the OpenReview submission page and must be read there. The file
   is built for 8 body pages. **If the limit is 4, restructure rather than trim.**
2. **Never compiled.** See §9. Compile before trusting any length or layout claim.
3. **`neurips_2026.sty` is not on this machine.** Download from neurips.cc and place
   beside `paper.tex`. Never modify it; that is grounds for desk rejection.
4. **Single- vs double-blind not confirmed.** File assumes `dblblindworkshop`.
5. **Checklist requirement not confirmed.** If mandatory, paste the template's
   checklist after the bibliography. Do not invent its wording.
6. **Blank amendment dates** in `tab:amendments`.
7. **The de-agent-ification pass is not finished** and cannot be finished from inside
   the loop. The user must read it end to end.
8. **The missing first-plus-second-state arm** was never trained. It is named in the
   paper as the obvious control and the first next experiment. Necessity of
   second-derivative supervision for the reconstruction is therefore not established,
   only sufficiency.
9. Length is at the edge. If a cut is needed after compiling, in order: the selection-
   failure paragraph in 4.3, then 4.5 down to its headline numbers, then `fig:pipeline`
   last (an adviser explicitly asked for it).

---

## 9. Environment facts that will waste your time if you rediscover them

- **There is no LaTeX toolchain on this machine.** No pdflatex, xelatex, lualatex,
  tectonic or latexmk. Consequences: the pipeline diagram is a rendered PNG rather
  than TikZ, deliberately, so it can be visually verified before shipping; and every
  page count in this project is an **estimate** from word count plus float area, never
  a measurement. Compile on Overleaf with `paper.tex` + `neurips_2026.sty` +
  `figures/`.
- **Use `/usr/bin/python3`**, not `.venv/bin/python3`. The venv is Python 3.9.6
  without numpy or torch. `/usr/bin/python3` has numpy 1.26.4 and torch 2.5.1.
- `timeout` is not available in this shell.
- Provider API: `evaluate(S, v, tau, K)` takes **array** S and v but **scalar** tau
  and K. Arm key is `rung3`, not `rung3_delta_gamma_vega`. Providers are built with
  `build_providers(bm, "results/grid", arms, seed, r, q, include_oracle=True)`.
- Config paths that bite: perturbation directions live at
  `eng["misspecification"]["directions"]`, and psi_c at `eng["simulation"]["psi_c"]`.
- Three hedging runs use **different no-trade bands**, verified from each
  `resolved_config.yaml`: `results/hedging` 0.02, `results/hedging_atc` 0.04,
  `results/hedging_bandtuned` 0.08. Never pool them.
- `mv_delta_comparator.py` **executes its study at import time**. Duplicate the class
  rather than importing it (this is why `mv_supervised_comparator.py` has its own copy).

### Two editing traps that already caused damage once each

- **Never** match a float with `\\begin\{table\}.*?\\label\{tab:x\}` under DOTALL. It
  starts at the *first* table in the file and swallowed the entire Results block into
  the appendix. Walk back from the `\label` line to the nearest `\begin`.
- **Never** use exact string replacement on a paragraph whose line wrapping may have
  changed. Use a whitespace-insensitive matcher:
  `re.compile(r'\s+'.join(re.escape(t) for t in old.split()))`.

---

## 10. Scrapped — one line each, do not resurrect

- **The registered gamma-hedging claim** — dead: real fail plus a signed reversal.
- **The +31.5% effect size** — a monotone readout of λ_pde, 93.6% of it repair of
  self-inflicted damage.
- **The transaction-cost reframe and the 0.575% tier** — reverse-engineered, appears
  in no artifact, withdrawn rather than relabelled.
- **Both registered mechanism channels** — cost channel presumed an over-trading
  baseline and ours under-trades by 38%; robustness channel is 86% present under
  correct dynamics.
- **"Score the metric in the hedging box"** — tested and demoted to a crude proxy
  (0.733 vs 0.710 concordance, worse for the first derivative at the extrapolated
  regime); replaced by the occupancy estimand.
- **The mechanism 2×2 as the paper's spine** — demoted to one row of `tab:verdicts`.
- **"The value-only network cannot build Δ_MV"** — measured false, it recovers 86%.
- **"The hedging and accuracy axes move independently"** — artifact of disjoint
  measurement regions.
- **The old title** ("Learned Derivatives Off-Distribution: What Supervision Buys,
  and Why the Standard Metric Hides It") — rejected as a fancy generic title.
- **The chronological narration** (assumed, failed, pivoted) — rejected by the adviser.
- **The UDE arm** (learned drift correction, `ude.py`) — honest null, ~1.7% worse than
  the residual-only baseline; not in the paper.
- **TikZ for the pipeline diagram** — dropped because it cannot be compiled or
  verified here.
- **`eq:tex`**, the excess-turnover display equation — moved out of the body.
- **The λ-selection procedure paragraph** — moved to Appendix B.
- **`fig:controls`** (label controls) — moved to Appendix C; its numbers stay in prose.
- **The six "hedging box" columns of `tab:greeks`** — dropped; `fig:coverage`b carries
  the reversal.
- **`exhibits.py` E2-E4** — superseded by `paper_figures.py` F3-F5 for manuscript use.
- **"Roughly 37×" regret ratio** (adviser's figure) — does not reproduce; it is 29.2×.
- **`f2_region_reversal.png`** — generated but not used in the paper.

---

## 11. Files

| Path | Role |
|---|---|
| `paper.tex` | the deliverable |
| `docs/paper_backup_2026-08-25.tex` | state before this session's rewrite |
| `docs/paper_draft_2026-08-19.tex` | pre-rewrite draft |
| `paper_figures.py` | every figure; `python paper_figures.py` regenerates all eight |
| `figures/` | f0 problem, f1 lambda, f3 dose, f4 mechanism, f5 infomatch, f6 coverage, f7 controls, f8 pipeline |
| `docs/ADJUDICATION_2026-08-18.md` | the adversarial adjudication that killed the registered claim |
| `docs/CODE_AUDIT_2026-08-20.md` | twelve-component audit of the evaluation paths |
| `docs/PAPER_DISCLOSURES.md` | what must be disclosed in the manuscript |
| `heston_benchmark_v6.yaml` | READ-ONLY contract; if a task needs it changed, STOP and say so |
| `results/reanalysis/` | the Tier-2 re-analyses the user ran personally |
| `mv_supervised_comparator.py` | the MV reconstruction experiment |

Autonomy dial: free to write code and tests, run pytest, make plots, triage logs.
**Propose and wait** before spending money (GPU), freezing artifacts, deleting
data/results, editing any YAML, adding dependencies, or changing the no-trade band.
