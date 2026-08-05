# contract_amendment_3_notes.md — correcting the region-of-validity rationale and the ladder

Branch `contract/v6-amendment-3`, one commit. **YAML + CLAUDE.md only; no `.py` file was
touched.** Suite before and after: **259 passed** — unchanged, which is the required signal
that nothing read a key it should not have.

Sources read before editing: `audit/fixlog/g2_production_scale.txt`,
`audit/FINDINGS_ADDENDUM.md` N8, the AM2-3 block of `heston_benchmark_v6.yaml`.

Marker convention: `AM3-<item>` in the inline comment, matching `C1` / `Q1-Q7` / `AM2-*`.

**NO CODE CHANGE WAS NEEDED.** Confirmed by execution, not by inspection — see §5.

---

## 0. What triggered this amendment

Two results out of fix batch 3, both of which land on AM2-3:

| | finding | where |
|---|---|---|
| A | The item-2 production-scale re-measurement tripped the STOP condition: `clipped_frac` at `sigma_rel = 0.20` is **0.2470 mean / 0.2513 max** against `clipped_frac_max = 0.25`, with **3 of 10 seeds outside** and seed std 0.0030. | `audit/fixlog/g2_production_scale.txt` |
| B | The item-3 emission falsified the premise AM2-3 was written on: the clip **amplifies** the delivered gamma error across the whole decision band (1.21x at 0.10, 1.41x at 0.15, 1.54x at 0.20), because a clipped hedger is flat in S and its gamma error saturates at the oracle's own −Γ. | `audit/FINDINGS_ADDENDUM.md` N8 |

A is a ladder question (ITEM 2). B is a rationale question (ITEM 1) — and, because the
decision band moved, it forces the contingency question (ITEM 3).

---

## 1. ITEM 1 — the AM2-3 rationale, replaced

### Diff hunk (block comment above `sigma_rel_ladder`)

```diff
   # `amp` is calibrated so std(d_eta/dS) equals sigma_gamma_target on the UNCLIPPED field
-  # and the clip is applied afterwards, so above ~0.2 the DELIVERED corruption diverges
-  # sharply from its label. The clip STAYS as it is: it is what keeps the corrupted arm a
+  # and the clip is applied afterwards, so the DELIVERED corruption diverges from its label.
+  # The clip STAYS as it is: it is what keeps the corrupted arm a
   # plausible hedger, ...
   # ---------------------------------------------------------------------------
+  # AM3-1 — THE DIRECTION OF THAT DIVERGENCE WAS WRONG, AND THE RATIONALE IS REPLACED.
+  #   [measured ratios inline: 0.05 -> 1.00  0.10 -> 1.21  0.15 -> 1.41  0.20 -> 1.54
+  #                            0.40 -> 1.40  0.80 -> 0.62  3.00 -> 0.17
+  #    delta counterpart:      1.00 / 1.05 / 1.16 / 1.28 / 1.34 / 0.82 / 0.23]
+  # WHY the clip AMPLIFIES: ... the clip REPLACES a small calibrated gamma error with a
+  #   larger uncalibrated one ...
+  # WHY A REGION OF VALIDITY EXISTS (the corrected reason): ... a heavily-clipped arm is a
+  #   STRUCTURALLY DIFFERENT OBJECT ... a SATURATED, bang-bang hedger ... a trained PINN's
+  #   Greek error is smooth in state, so the spread-to-gamma-error mapping measured on a
+  #   saturated arm does not transfer to it — in EITHER direction ...
+  # THE ANTI-CONSERVATIVE FAILURE MODE THIS FORBIDS: ... at nominal 0.15 the delivered error
+  #   is 1.41x, so a spread actually produced by a ~0.21-sized gamma error would be credited
+  #   to a 0.15-sized one — returning a GO when the delivered gamma error needed to produce
+  #   that spread EXCEEDS the pilot's ... compare_pilot_against: sigma_gamma_effective is
+  #   what prevents that; it is a correctness requirement, not a reporting nicety.
   # ---------------------------------------------------------------------------
```

The one deleted clause is `above ~0.2 the DELIVERED corruption diverges **sharply** from its
label` — "sharply", together with the G2 note it summarised, carried the direction. What
replaces it states the direction explicitly and gives the numbers. **The old claim is quoted
inside the correction rather than deleted**, so a reader of the frozen contract sees what was
believed, what was measured, and which one governs.

### 1.1 The three other places in the contract that stated the same premise

`grep` on the whole contract for `understat|conservative|smaller than the sigma|clip removed`
plus a read of every `clip`-mentioning line. Four sites total, all corrected:

| # | site | what it said | what it says now |
|---|---|---|---|
| 1 | AM2-3 block comment | "diverges **sharply** from its label" (direction implied downward, via G2) | AM3-1 block: direction stated, measured ratios inline |
| 2 | `effective_sigma_reporting.units_note` | `sigma_delta_effective` "is the direct measure of how much of the intended corruption the clip **removed**" | "how much the clip **CHANGED**… 'removed' is what this note said and it is wrong in the decision band, where the clip ADDS (1.05x to 1.28x)" |
| 3 | `region_of_validity.interpretation` | "the delivered corruption is **dominated by the clip** rather than by the calibrated field… no longer a monotone reading of a gamma error of the labelled size" — left the reader to infer "therefore weaker" | rewritten on the structural reason: saturated bang-bang hedger vs smooth perturbation; "not because the mapping is biased in a known direction, but because the object being measured is no longer the object the gate stands in for" |
| 4 | `region_of_validity.clipped_frac_max` inline comment | "with headroom for seed/cell variation in the measured 0.215" | AM3-2 note: production numbers, bound deliberately unchanged, rung demoted instead |

`hedging_config.yaml` and `pinn_config.yaml`: no hits, untouched.
`docs/` (BASELINE_STATUS.md, CONFIG_AUDIT.md, STATUS_2026-07-29.md): **no hits** —
`grep -rn -i "understat|conservative|clip|clipped" docs/` returns nothing. Nothing to correct.
CLAUDE.md: one new bullet, §4.

The two surviving `conservative` hits in the contract (`lambda_selection.lambda_pde.rationale:353`,
`acceptance_thresholds.dose_response.conservatism:495`) are unrelated — both are "conservative
**wrt the study's hypothesis**", about λ sourcing and the dose-response criterion. Left alone.

### 1.2 Where the false claim still stands — IN CODE, out of scope, reported not fixed

The rules for this amendment are YAML + CLAUDE.md only. The claim is also written in three
`.py` docstrings/comments, which now contradict the contract:

| file:line | text | status |
|---|---|---|
| `gate_headroom.py:106-109` (`_CLIPPED_NOTE`) | "the spread is understated and the gate is conservative (the safe direction for a go/no-go…)" | **partially corrected already** — fix batch 3 appended a `MEASURED CAVEAT` to the same string, so the report prints both. The false sentence is still the first one a reader meets. |
| `gate_headroom.py:337-338` (`NoisyOracleProvider.__init__` comment) | "the DELIVERED gamma error is smaller than sigma_gamma_target and the measured spread is understated (conservative, but the sigma axis is then not the axis it is labelled with)" | **uncorrected** |
| `gate_headroom.py:375` (`clipped_fraction` docstring) | "so the arm's spread is understated (audit G2)" | **uncorrected** |
| `test_gate_headroom.py:161-162` (`test_binding_delta_clip_is_reported` docstring) | "the measured spread is understated and the gate is conservative — safe direction, wrong axis label" | **uncorrected** |

None of these is *required* for this amendment to be correct or complete: they are prose, they
change no behaviour, and the contract now governs. They are a **documentation-only follow-up
for a code batch** — recommended before the freeze tag, since a frozen repo whose docstrings
contradict its own pre-registration is the exact failure mode amendment 2's §5 warned about.
The correcting test already exists
(`test_gate_headroom.py::test_effective_gamma_can_EXCEED_the_nominal_in_the_decision_band`),
so the follow-up is text only.

`train.py:142` and `gate_headroom.py:413` also match `conservative` and are unrelated
(λ sourcing; "a rung named in BOTH lists is diagnostic — the conservative reading").

---

## 2. ITEM 2 — `sigma_rel = 0.20` demoted to diagnostic

### Diff hunk

```diff
   sigma_rel_ladder:
-    decision: [0.05, 0.10, 0.15, 0.20]        # the ONLY rungs the DECISION section may fire on
-    diagnostic: [0.40]                        # ... ~78% clipped, outside region_of_validity
+    decision: [0.05, 0.10, 0.15]              # AM3-2 — ... 0.20 DEMOTED, see production_scale_measurement
+    diagnostic: [0.20, 0.40]                  # ... 24.7% and 81.4% clipped at production scale
     spacing_rationale: > ...unchanged...
+    production_scale_measurement: >
+      [the 10-seed production table, then the two consequences]
```

`spacing_rationale` is left **verbatim**: it is the reasoning that produced the AM2-3 ladder,
it was honest about its own evidence base ("0.05 and 0.15 are NOT separately measured"), and
overwriting it would erase the record. `production_scale_measurement` sits beside it and
supersedes it where they differ, and says so.

### What it records

- **The measurement**, mean [min, max] over the 10 confirmatory seeds at `n_paths = 10000`:
  `0.05 -> 0.0027 [0.0024, 0.0029]`, `0.10 -> 0.0737 [0.0712, 0.0756]`,
  `0.15 -> 0.1484 [0.1449, 0.1516]`, `0.20 -> 0.2470 [0.2429, 0.2513]`,
  `0.40 -> 0.8137 [0.8102, 0.8194]`; production scale is systematically ~11-20% relative
  higher than the smoke-size numbers the ladder was cut against.
- **0.05 and 0.15 are now measured**, at 0.0027 and 0.1484 — AM2-3a could only *bound* them
  (`<=0.061`, `<=0.215`) by the monotonicity argument. The argument held; those rungs need no
  further defence. (Requested explicitly by the amendment task.)
- **Why 0.20 goes**: mean inside the bound, per-seed values straddling it, 3 of 10 outside,
  max 0.2513. "A rung whose decision-eligibility depends on which seeds were drawn is not a
  decision rung." The **small** seed std (0.0030) is what settles it — this is the rung sitting
  ON the bound, not noise around a comfortable value.
- **The bound was deliberately NOT raised.** Stated in `clipped_frac_max`'s own comment:
  AM2-3c chose 0.25 as the smallest round bound that admitted the then-measured ladder, i.e.
  **fitted to the ladder**; re-fitting it against the data that broke it so the same ladder
  still passes is circular. The bound stays, the rung moves.
- The decision band is now nominal `[0.05, 0.15]`, which in DELIVERED terms tops out at
  `sigma_gamma_effective ~ 0.21 x rms(Gamma_oracle)` (0.15 x the 1.41 amplification of AM3-1).

---

## 3. ITEM 3 — `ladder_extension_contingency`, declared before any pilot exists

New key, placed after `region_of_validity` (it depends on both that block and
`effective_sigma_reporting`) and before `motivation` / `runs_before`.

| field | content |
|---|---|
| `trigger` | pilot `sigma_gamma_effective` above the top decision rung's (~0.21 x rms(Γ)); that state is already reported INCONCLUSIVE by `region_of_validity.if_pilot_outside` |
| `extension_rungs: [0.20, 0.25]` | the ONLY rungs addable, in that order. 0.20 re-enters as a **candidate**, not restored. Nothing above 0.25 — 0.40 is 81% clipped and diagnostic by pre-registration |
| `added_as: diagnostic` | a new rung is swept/reported/plotted but NOT decision-eligible on arrival |
| `preconditions_for_decision_eligibility` | production-scale `clipped_frac` re-measurement (`n_paths_per_cell`, `seeds_confirmatory_cell`, one gate run per seed, the `audit/repro/e2_g2_production_scale.py` protocol) with **both** the seed mean **and the per-seed maximum** at or below the bound; measurement written to `audit/fixlog/` before promotion |
| `clipped_frac_max_stays: 0.25` | fixed for that re-derivation; widening it would make the precondition vacuous and repeat the circularity ITEM 2 refused |
| `expected_outcome` | **stated in advance because it is the likely one**: `clipped_frac` is monotone in `sigma_rel` and 0.20 already straddles, so with the bound fixed an extension will probably NOT yield an eligible rung. Then the honest reading is that the gate cannot bracket a pilot that large AT THIS CLIP, and the levers are the contract's own (`decision_rule`: rebalancing frequency, then misspecification severity) — not the bound, not the clip, not a wider ladder |
| `reporting` | an extended-ladder result is reported AS one (ladder executed, rungs added, when each was measured and promoted, and that the original ladder did not bracket the pilot); the headline keeps the original ladder's INCONCLUSIVE alongside |
| `declared_before_pilot: true` | `data/` and `results/` do not exist; no pilot fit, no gate run, no `sigma_gamma_pilot` |

The per-seed-maximum requirement is not decoration: it is precisely the test 0.20 failed, so
the precondition is calibrated to a case we have actually seen rather than to a hypothetical.

`expected_outcome` is the part worth a second read. A contingency that quietly expects to
succeed invites the bound to be renegotiated when it does not; naming the likely failure now
turns "the gate cannot bracket this pilot" into a pre-registered result.

---

## 4. ITEM 4 — CLAUDE.md sync

One existing bullet edited (`decision rungs ≤0.20 only, 0.40 diagnostic-only` ->
`decision rungs ≤0.15 only, 0.20 and 0.40 diagnostic-only`) and three added, one per item,
each naming the contract keys and nothing more:

- AM3-1: the clip AMPLIFIES in the decision band; the "understates, so conservative" premise is
  measured false; comparing against nominal σ_rel would be anti-conservative.
- AM3-2: 0.20 is diagnostic; `clipped_frac_max` stays 0.25, never raise it to re-admit a rung.
- AM3-3: a pilot above the band follows `ladder_extension_contingency`.

---

## 5. Does any code path change behaviour today?

| key | read by any `.py`? | behaviour change today |
|---|---|---|
| `sigma_rel_ladder.decision` / `.diagnostic` | **YES** — `Hedging_backtest.contract_thresholds:131,134` -> `gate_headroom._resolve_ladder` | **YES, when the gate next runs.** The DECISION scan now considers `[0.05, 0.10, 0.15]`; `0.20` is swept, hedged, written to `headroom.csv` and shown in the report table as `DIAGNOSTIC`, and can no longer be selected as a decision row. **No gate has ever run** (`results/` does not exist), so no existing artifact changes and no recorded verdict moves. |
| `sigma_rel_ladder.spacing_rationale` / `.production_scale_measurement` | no | none — documentation with contract force |
| `effective_sigma_reporting.units_note` / `.emission_status` | no | none |
| `effective_sigma_reporting.compare_pilot_against` | YES (`contract_thresholds:139`) | **unchanged value** (`sigma_gamma_effective`); AM3-1 only states why it is load-bearing |
| `region_of_validity.clipped_frac_max` | YES (`contract_thresholds:144`) | **unchanged value** (0.25) — that is the point of ITEM 2 |
| `region_of_validity.interpretation` / `.if_pilot_outside` | no (`if_pilot_outside` is IMPLEMENTED, but by its semantics, not by reading the string) | none |
| `ladder_extension_contingency.*` | no | none — declaratory until a pilot triggers it |

Verified rather than asserted — the executed ladder read back through the real code path
after the edit:

```
contract decision  : [0.05, 0.1, 0.15]      executed decision  : [0.05, 0.1, 0.15]
contract diagnostic: [0.2, 0.4]             executed diagnostic: [0.2, 0.4]
thresholds decision/diagnostic: (0.05, 0.1, 0.15) (0.2, 0.4)
source: contract | 0.2 decision-eligible: False
```

### NO CODE CHANGE WAS NEEDED — explicit confirmation

`git diff --stat` for this commit is `CLAUDE.md` and `heston_benchmark_v6.yaml` **only**. The
demotion took effect with zero code edits because fix batch 3 ITEM 1 had already made the
ladder contract-read and carried `decision_eligible` per rung all the way into the decision
scan; ITEM 2's demotion is therefore a data change, exactly as this task predicted.

`python -m pytest -q` -> **259 passed**, identical to the pre-amendment baseline. Three of
those tests read the ladder or the bound out of the YAML dynamically
(`test_sigma_ladder_comes_from_the_contract`, the `gate_sigma_rel_*` parity rows,
`test_pilot_outside_the_region_of_validity_is_inconclusive`), so they track the new values
rather than pinning the old ones — which is why the count is unchanged even though a live
input moved. No test asserted the old four-rung decision list.

---

## 6. Compliance with the task's rules

- Branch `contract/v6-amendment-3`, one commit. ✅
- YAML + CLAUDE.md only; **no `.py` touched**. ✅
- Existing structure, key ordering, comment style (`# MARKER — text`, `>` folded blocks) and
  quoting preserved; nothing reformatted; `spacing_rationale` kept verbatim with a superseding
  sibling rather than an overwrite. ✅
- `python -m pytest -q` -> 259 passed, unchanged from baseline. ✅
- Nothing merged, nothing tagged. ✅
