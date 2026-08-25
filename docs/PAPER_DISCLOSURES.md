# Paper disclosures and scope conditions — from the 2026-08-20 code audit

Paper-ready text for the three audit findings that must appear in the manuscript, plus
the two scope conditions the current draft does not state. Each block is written to be
lifted into the section named, and each number in it is reproducible from the artifacts
cited. Nothing here re-scores a registered endpoint.

Actions covered: **A1** (report both measurement regions), **A3** (declare the (S,K,τ)
coverage gap), **A4** (disclose the effective-λ inflation and the boundary selection).

---

## A1 — Results: report both measurement regions, and say they disagree

> **For §Results, immediately after the OOD Greek table.**

The out-of-distribution Greek metric is scored on the full contract grid
(S ∈ [50,150], K ∈ [60,140], τ ∈ [0.04,1.0]), while the hedging metric is produced in a
narrow box around the hedged contract: the strike K = 100 the hedge trades, and the
maturities τ ∈ [0.08, 0.25] it walks through between inception and liquidation at
T′ = 0.17. Those regions are disjoint, and the arms do not rank the same way in them.
Restricting the identical Greek metric to the hedging box (Table X, right-hand columns)
reverses the residual-only baseline against the price-only network on both held-out
regimes and on both first- and second-order Greeks:

| held-out regime | Greek | full grid, std_pinn / feedforward | hedging box, std_pinn / feedforward |
|---|---|---|---|
| `near_feller` | Δ | 0.61× (baseline better) | **2.29× (baseline worse)** |
| `near_feller` | Γ | 0.94× | **1.81×** |
| `strong_neg_corr` | Δ | 0.52× | **3.04×** |
| `strong_neg_corr` | Γ | 0.84× | **2.44×** |

The residual-only baseline is the **only** arm that degrades moving from the full grid
into the box (`near_feller` Δ RMSE 0.0537 → 0.0937); every other arm improves there
(rung 3 0.0027 → 0.0037 is within seed noise, `sobolev_sans_pde` 0.0107 → 0.0052,
feedforward 0.0881 → 0.0409). Five seeds; `results/eval_greeks_hedgeslice/`.

**What this licenses, and what it does not.** It licenses the statement that the hedging
result is produced by a *region-specific* error of the baseline that the registered Greek
table cannot see. It does **not** license the draft's claim that the hedging and accuracy
axes move independently. Measured in one region they agree: at the actual hedging path
states the delta-RMSE ordering (rung 3 0.0038 < rung 2 0.0057 < rung 1 0.0202 <
feedforward 0.0484 < standard PINN 0.0989 < shuffled 0.1458) matches the zero-cost
CVaR₉₅ ordering, and the two arms that depart from it — `feedforward` and `bs_gamma` —
both carry a negative hedge-ratio *bias* (−0.023, −0.025) in the minimum-variance
direction, which is the mechanism of §MV, not an RMSE effect. The sentence "the hedging
and accuracy axes move independently" must be replaced by "the two axes rank arms
oppositely **when measured in different regions**, and agree when measured in the same
one."

The registered OOD Greek threshold continues to read the full grid only; the box is
reported beside it and gates nothing
(`test_eval_greeks.py::test_hedge_slice_cannot_move_a_registered_verdict`).

---

## A3 — Scope condition: label coverage of the hedged contract

> **For §Methods (training distribution) and the limitations paragraph.**

The training set is 512 hypercube parameter points crossed with **64 (S, K, τ) triples**
drawn once and shared across every parameter point (`n_skt = 64`, contract amendment
AM4-1). With three grid dimensions and 64 shared points, coverage of any particular
contract is sparse, and the hedged instrument is only partly covered. Measured in
range-normalised (S, K, τ):

| hedge state | nearest training triple | normalised distance | triples within 0.15 |
|---|---|---|---|
| inception, τ = 0.25 | S = 100.4, K = 100.1, τ = 0.2598 | 0.011 | 1 |
| mid-life, τ = 0.17 | same | 0.094 | 1 |
| liquidation, τ = 0.08 | S = 85.1, K = 94.2, τ = 0.1111 | 0.169 | 0 |

The hedge therefore **begins on a labelled point and leaves the labelled region as
maturity decays**; no training triple lies inside the box (K = 100 ± 5, τ ∈ [0.08, 0.25],
S ∈ [65, 125]). PDE collocation is not subject to this — it is redrawn uniformly over the
full grid ranges each epoch — but it is subject to the same anchor-ball excision in
parameter space, so at the hedging anchor the residual-only baseline has physics signal at
other parameters and no label signal at all.

**Scope condition to state:** the hedging results measure *extrapolation in the contract
dimensions as maturity decays*, not interpolation. This is a property of the design and it
bounds the claim: a protocol with denser (S, K, τ) coverage near the hedged contract could
order the arms differently. The parameter-space hold-out is deliberate and pre-registered;
this (S, K, τ) gap is neither, and is disclosed here rather than defended.

---

## A4 — Methods: how the shared PDE weight was set, and two facts about that

> **For §Methods (loss and weight selection).**

**(a) The residual term is normalised against a reference far smaller than its own
dominant term.** Loss terms are scale-normalised by the second moment of their labels;
for the PDE residual the declared reference is `mean((r·price)²)`, the discount term of
the residual evaluated on the labels. Measured on the frozen artifact, that reference has
rms 0.527 while the residual's dominant diffusion term `½ v S² Γ` has rms 4.674 — the
normaliser is 8.9× smaller in rms and **79× smaller in squared units**. A nominal
λ_PDE therefore carries roughly 79× the weight a scale-matched normalisation would give
it, and the registered candidate grid {0, 0.01, 0.1, 1.0} spans effective weights of
about {0, 0.8, 8, 79} in dominant-term units. At the selected λ_PDE = 0.01 the residual
is still **49% of the baseline's total training loss**, and the baseline's in-sample
normalised price MSE is 8.64 × 10⁻⁵ against the price-only network's 3.05 × 10⁻⁶ — the
residual makes the in-sample price fit 28× worse. (Derivative supervision more than
recovers it: rung 3 reaches 1.77 × 10⁻⁶.)

**(b) The weight was selected at the lower boundary of its grid, on a region that
excludes the one the headline is measured in.** λ_PDE was scored on the hypercube
validation split crossed with the full (S, K, τ) grid: 0.1757 at λ = 0, **0.1697 at
λ = 0.01**, 0.3332 at 0.1, 0.3724 at 1.0. The winner is the smallest non-zero candidate,
it beats λ = 0 by 3.4%, and everything larger degrades sharply — so the grid selected at
its own edge and the interval (0, 0.01) was never explored. On the selection region the
choice is right (validation Δ RMSE 0.0418 vs 0.0607 for λ = 0); at the hedging states it
is inverted (0.0989 vs 0.0484). Named anchors are excised from training **and**
validation, so the selection is blind by construction to the region that decides the
primary metric.

**(c) What we did about it.** Because (b) leaves the baseline's quality unresolved inside
the unexplored interval, we report the confirmatory contrast as a function of λ_PDE over
{0, 10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻²} at the confirmatory cell and seed count
(§Sensitivity). The registered λ_PDE remains 0.01 and the registered verdicts are
computed at that value only; the sweep is a sensitivity analysis, not a re-selection.

---

## Three sentences the manuscript must not contain (carried forward)

Unchanged from `docs/ADJUDICATION_2026-08-18.md` §7, plus one added by this audit:

1. "Explicit Gamma supervision improves delta-only hedging PnL under misspecified dynamics."
2. "The improvement operates through the model-uncertainty-robustness channel" (or the
   transaction-cost variant).
3. "The confirmatory test would have passed at a correctly calibrated transaction-cost tier."
4. **New:** "The hedging and accuracy axes move independently" — and any variant asserting
   that Greek accuracy fails to predict hedging performance, unless the two are measured in
   the same region.
