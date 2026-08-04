# contract_amendment_2_notes.md — the second and final pre-freeze amendment

Branch `contract/v6-amendment-2`, one commit. **YAML + CLAUDE.md only; no `.py` file was
touched.** Suite before and after: **240 passed** — unchanged, which is the required
signal that nothing read a key it should not have.

Sources read before editing: `audit/contract_requests.md`, `audit/FINDINGS_ADDENDUM.md`,
`audit/FIXLOG.md` ("Open for the human"), `audit/fixlog/g2_measured.txt`.

Marker convention: new keys carry `AM2-<item>` in their inline comment, matching the
existing `C1` / `Q1` / `Q2` / `Q3` markers from amendment 1.

---

## 0. Read this first — three judgment calls and one live divergence

| # | What | Where |
|---|---|---|
| A | Item 2 as specified names four verdict values; the code emits **eleven**. I declared the universal pair plus the per-threshold outcome sets. | §2.1 |
| B | `null` is BOTH "not evaluated" AND a genuine adjudicated outcome for two thresholds. Documented, **not resolved** — resolving it is a code change. | §2.2 |
| C | Item 3(b)'s "effective sigma" is a delta-error std, but `sigma_gamma_pilot` is a gamma RMSE. I declared **two** effective fields and named the gamma one as the pilot comparison target. | §3.2 |
| **D** | **LIVE DIVERGENCE:** the contract now declares a ladder the code does not implement. `gate_headroom.py:82 _SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)` still stands. | §3.1, §5 |

Under the CLAUDE.md rule "when code and YAML disagree, the YAML wins and the code changes",
**D is a defect from this commit forward until Prompt E lands.** It is deliberate and
sanctioned by the task (YAML only), but it must not be left standing across the freeze tag:
a frozen contract that declares an unimplemented ladder is worse than either alternative.

---

## 1. ITEM 1 — `acceptance_thresholds.confirmatory_cell_rel_min`

### Diff hunk

```diff
@@ -317,6 +371,7 @@ acceptance_thresholds:
 acceptance_thresholds:
   confirmatory_cell_pass: "misspec delta CVaR95 improvement >=10% relative AND paired-bootstrap 95% CI excludes 0 (combined,1% TC,baseline,rung3 vs standard_pinn)"
+  confirmatory_cell_rel_min: 0.10             # AM2-1 — the ">=10% relative" named above, AS A NUMBER. Was the last surviving Python literal after C1 (Hedging_backtest.contract_thresholds, marked TODO(C1)); requested in audit/contract_requests.md. Prose above unchanged; the number joins it
   headline_scale_free: fraction_of_baseline_to_oracle_gap_closed
```

**What it resolves.** `audit/contract_requests.md` item 1 — the single value that survived
the C1 single-sourcing pass. The key name is exactly the one that request document proposed,
so request and response bind without a rename. Placed immediately after the prose it
quantifies, mirroring how `sakuma_null_rel_tol` sits under `in_model_hedging`.

**Behaviour change today: NONE.** `Hedging_backtest.contract_thresholds:115` still reads
`"confirmatory_rel_threshold": 0.10,  # TODO(C1): prose-only in the contract`. The value is
identical (0.10), so no verdict moves; the literal is simply now redundant rather than
irreplaceable.

**Follow-up (code, not this task).** Replace that literal with
`float(at["confirmatory_cell_rel_min"])`, drop the `TODO(C1)` from the docstring, and add the
key to `test_contract_thresholds.py`'s parity table. Until then the duplicate is live and
C1's hazard (a human edits the YAML, the verdict keeps the old number) is **not yet closed**
for this one value — declaring the key is necessary but not sufficient.

---

## 2. ITEM 2 — `acceptance_thresholds.verdict_vocabulary`

### Diff hunk

```diff
@@ -336,6 +391,25 @@ acceptance_thresholds:
   tail_claim_requires: paired_bootstrap_over_CRN_paths_with_seed_variance_separated
+  verdict_vocabulary:                         # AM2-2 — permitted values of the `verdict` column of results/tables/threshold_verdicts.csv. Fix batch 2 (A4) introduced `error`; the reporting schema is part of the pre-registration, so the vocabulary is declared here rather than discovered from the code
+    universal:                                # available to EVERY threshold row
+      "null":  "NOT EVALUATED. ..."
+      "error": "evaluation was ATTEMPTED and FAILED: ..."
+    outcome_values:                           # each threshold emits exactly one of ITS pair/set in place of a universal value
+      confirmatory_cell: [pass, fail]
+      order_attribution: [pass, fail]
+      ood_greek_thresholds: [pass, fail]
+      dose_response: [monotone, flat]
+      sakuma_null_consistency: [consistent, flag]
+      mechanism_adjudication: [channel_i, channel_ii, decomposition, "null"]
+      goldilocks_bates: [decision_relevant_regime_located, "null"]
+    must_not_collapse: >
+      `null` and `error` must NEVER be merged in any table, memo or figure. ...
+    yaml_note: "`null` and `error` are QUOTED because bare null is a YAML null literal; ..."
```

*(elided strings above are verbatim in the YAML; the full text is in the file.)*

**What it resolves.** Fix batch 2's A4 introduced `error` as a value distinct from `null`.
Verdict values are the reporting schema of a pre-registered study, so the vocabulary is now
declared rather than inferred from `analyze_results.py`.

### 2.1 Deviation from the item as specified — CHECK THIS

The task says *"Declare the permitted set — pass, fail, null, error."* **The verdict column
does not carry four values; it carries eleven.** Enumerated from
`analyze_results.py` (`grep -n 'verdict = '` plus `_mechanism_reading`'s reading strings,
which `mechanism_adjudication` passes straight into `_verdict`):

```
pass  fail  null  error  monotone  flat  consistent  flag
channel_i  channel_ii  decomposition  decision_relevant_regime_located
```

Declaring only four as "permitted" would have made the contract factually wrong about its own
outputs on day one — a pre-registration that a conforming run violates immediately. I
declared the structure the code actually has: a **universal** pair (`null`, `error`) available
to every row, plus **per-threshold outcome sets**. The distinction the item was written to
protect — null vs error, never collapsed — is stated in full and given its own
`must_not_collapse` clause.

If you intended the narrower literal reading, this is the one hunk to send back.

### 2.2 Unresolved collision: `null` means two different things

`mechanism_adjudication` and `goldilocks_bates` emit the string `"null"` as a **genuine
adjudicated outcome** — "no channel is present", "no decisive severity row was located" —
which is a *result*, not an absence. The same string is the universal not-evaluated value.
A reader scanning `threshold_verdicts.csv` cannot tell them apart without the notes column.

This is pre-existing (it predates A4) and **the amendment documents it rather than fixing
it**, because fixing it means renaming a verdict string, which is a code change. It is the
exact shape of `FINDINGS_ADDENDUM.md` N3, which suggested a distinct `incomplete` value for
the A3 shortfall and was deferred for the same reason.

**Recommendation for the code batch that acts on this:** rename the two adjudicated values to
`no_channel` and `no_decisive_regime`, leaving `null` to mean only "not evaluated". That is a
one-line change in each function plus the contract list here. Doing it BEFORE the freeze is
much cheaper than after.

**Behaviour change today: NONE.** No code reads `verdict_vocabulary`
(`grep -rn verdict_vocabulary --include='*.py'` → no hits). It is documentation with contract
force, and a future schema test can bind it.

---

## 3. ITEM 3 — `oracle_headroom_gate`: ladder, effective sigma, region of validity

The measurement this rests on (`audit/fixlog/g2_measured.txt`, field mode, confirmatory cell,
`n_paths=256`, 2 seeds):

| sigma_rel | clipped_frac | spread_rel_mean @ tc=0.01 |
|---|---|---|
| 0.1 | 0.061 | +0.446 |
| 0.2 | 0.215 | +1.065 |
| 0.4 | **0.780** | +2.051 |
| 0.8 | **0.960** | +2.839 |

The clip was **not** widened or removed, per instruction.

### 3.1 (a) `sigma_rel_ladder` — decision rungs `[0.05, 0.10, 0.15, 0.20]`, diagnostic `[0.40]`

**Spacing justification, against the table.** The binding fraction roughly *triples* from 0.1
to 0.2 and then saturates. Resolution below 0.2 buys information about where the mapping
degrades; resolution above it buys nothing, because 0.78 and 0.96 are both "the clip is
running the hedger". Hence 0.05 spacing across `[0.05, 0.20]`, one diagnostic rung at 0.40 to
exhibit the saturation in the report, and 0.80 dropped entirely.

**On the two unmeasured rungs — stated as a bound, not a guess.** 0.05 and 0.15 were not
separately measured. Every arm scales ONE common frozen RFF field by `amp`
(`gate_headroom.py:301`, `gate_seed = meta.global_seed`), so `|err|` is monotone in `amp` and
`clipped_frac` is therefore monotone non-decreasing in `sigma_rel`. The new rungs are
consequently **bounded** by their measured neighbours (`≤0.061` and `≤0.215`), not
interpolated. The YAML says exactly this.

**Behaviour change today: NONE — and that is the problem.** See §5. The gate still runs
`_SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)`.

> The task brief predicted "item 3 changes the declared ladder, which changes gate behaviour
> when the gate next runs". **That is not accurate as things stand:** the ladder is a Python
> literal that no contract read reaches, so the gate would next run on the OLD ladder unless
> the operator passes `--sigma-rel 0.05 0.10 0.15 0.20 0.40` by hand. Behaviour changes when
> Prompt E wires the key, not when this commit lands.

### 3.2 (b) `effective_sigma_reporting` — and the units call

Declared field names:

- `sigma_delta_effective` — std over the reference cloud of the delivered post-clip delta
  error, `clip(delta + eta) - delta`. **This is the quantity the task named.**
- `sigma_gamma_effective` — std of `d/dS` of that delivered delta error; the post-clip
  counterpart of the calibration statistic `std(d_eta/dS)`.
- `report_alongside: sigma_rel`, `compare_pilot_against: sigma_gamma_effective`.

**Why two fields.** The task says the effective value "is the one the pilot point is compared
against". But `sigma_gamma_pilot` (written by `train.py --pilot`, handed over via
`--sigma-gamma-from-runlog`) is a **gamma RMSE**, and the nominal axis
`sigma_gamma_target = sigma_rel * rms(Gamma_oracle)` is likewise in gamma units — while a
delta-error std is in delta units. Comparing the pilot against `sigma_delta_effective` would
be a units error. So the amendment declares both post-clip quantities (they come from one
measurement pass) and names the **gamma** one as the pilot's comparison target, with a
`units_note` spelling out why. `sigma_delta_effective` is retained because it is the direct
measure of how much of the intended corruption the clip removed.

**If you disagree, this is a one-line edit:** change `compare_pilot_against`.

**Behaviour change today: NONE.** Emission is a code change (Prompt E). Today the gate reports
nominal `sigma_rel` plus the `clipped_frac` that fix batch 2 added; the YAML says so
explicitly so the gap is not mistaken for an implemented feature.

### 3.3 (c) `region_of_validity` — `clipped_frac_max: 0.25`

**How the bound was chosen, stated honestly in the YAML itself:** it is the smallest round
number that admits the entire *decision* ladder as measured (0.215 at its top rung, 0.20)
while excluding the diagnostic rung (0.780 at 0.40). It is fitted to the ladder with headroom
for seed- and cell-to-cell variation in that 0.215 — **not derived from an independent
principle**, and the comment says so rather than implying a derivation that does not exist.

**Caveat on the evidence base.** The 0.215 comes from a smoke-size run (`n_paths=256`,
2 seeds), not the full-size gate. The bound therefore rests on a small measurement, and the
seed-to-seed spread of `clipped_frac` at `sigma_rel=0.2` is unknown. If the full-size gate
shows 0.20 landing materially above 0.25, `clipped_frac_max` — not the ladder — is what
should be revisited, and that revision would itself be post-hoc.

**Consequence clause, as required.** `if_pilot_outside`: a pilot landing above the region of
validity makes the gate **INCONCLUSIVE** — explicitly neither a pass nor a no-go — does not
authorize training spend, and forces the ladder (and the clause) to be revisited before any
go decision.

**Pre-pilot record, as required.** The block comment states that at the time of the amendment
no pilot fit and no gate run existed (`data/` and `results/` do not exist), so the region of
validity was fixed without knowing which side of it `sigma_gamma_pilot` would land on.

**Behaviour change today: NONE.** No code reads these keys.

---

## 4. Keys placed where a downstream read will be awkward

1. **`sigma_rel_ladder` is split `decision` / `diagnostic`; `run_gate` takes one flat
   `sigma_rel_list`.** The signature is
   `run_gate(cfg, sigma_rel_list: tuple = _SIGMA_REL_DEFAULT, ...)` and the DECISION scan
   (`gate_headroom.py:367-372`) walks `summary` with no notion of a rung being
   decision-ineligible. Wiring this needs *two* changes, not one: concatenate the lists for
   the sweep, AND carry a per-arm `decision_eligible` flag so the `next(...)` that picks the
   decision row skips diagnostic rungs. A naive `sigma_rel_list = decision + diagnostic`
   would let 0.40 — the rung this amendment exists to demote — fire the gate. **This is the
   single most likely way to implement the amendment and get it backwards.**

2. **`region_of_validity.if_pilot_outside` is not yet checkable as written.** It is stated in
   terms of the *effective* sigma, which nothing emits (§3.2). A proxy check IS available
   today: fix batch 2 already emits per-arm `clipped_frac` in the summary and report, so the
   pilot arm's own `clipped_frac` vs `clipped_frac_max` is computable in the current code.
   Prompt E should implement the proxy at minimum, and say in the report which of the two it
   used.

3. **`verdict_vocabulary.universal` uses quoted keys.** `"null"` must stay quoted — bare
   `null:` is a YAML null literal and would parse as the key `None`. Verified after the edit:
   `list(vv['universal'])` → `['null', 'error']`, `None in vv['universal']` → `False`. Anyone
   hand-editing this block must preserve the quotes; the `yaml_note` key says so in-file.

4. **`_DELTA_CLIP = (-0.05, 1.05)` remains a Python literal.** The new `region_of_validity`
   block *refers* to it (in `statistic`'s comment) but does not declare it as a contract
   number. I did not declare it: the task did not ask, and adding it would create exactly the
   duplicate-literal hazard C1 spent a batch eliminating — with the twist that this one is
   load-bearing for the corruption model, not just for a verdict. **Candidate for a third
   amendment if one happens**; otherwise it should be threaded from a contract key by the same
   Prompt E that reads the ladder.

---

## 5. Summary: does any code path change behaviour today?

| Key | Read by any `.py` today? | Behaviour change today |
|---|---|---|
| `acceptance_thresholds.confirmatory_cell_rel_min` | No | **None** — duplicate of the live literal, same value 0.10 |
| `acceptance_thresholds.verdict_vocabulary.*` | No | **None** — documentation with contract force |
| `oracle_headroom_gate.sigma_rel_ladder.*` | No | **None today**; see the divergence below |
| `oracle_headroom_gate.effective_sigma_reporting.*` | No | **None** — the fields do not exist yet |
| `oracle_headroom_gate.region_of_validity.*` | No | **None** — nothing evaluates the clause |

Verified: `grep -rn "sigma_rel_ladder\|clipped_frac_max\|verdict_vocabulary\|confirmatory_cell_rel_min" --include="*.py" .` returns **no hits** outside `audit/`.
This is also what the unchanged suite count (240) attests.

### The divergence this commit opens

```
contract  oracle_headroom_gate.sigma_rel_ladder = decision [0.05, 0.10, 0.15, 0.20] + diagnostic [0.40]
code      gate_headroom.py:82 _SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)
```

Both `run_gate`'s default and the `--sigma-rel` argparse default still point at the old tuple.
Per CLAUDE.md the YAML wins and the code must change. Until it does, **a gate run launched
with default arguments will sweep the superseded ladder, including the 0.8 rung this
amendment deleted** — and will do so silently, since nothing cross-checks.

Same class of divergence, smaller: `confirmatory_cell_rel_min` vs the `TODO(C1)` literal
(§1) — harmless only because the two values are equal.

**Recommendation: do not apply the freeze tag until Prompt E closes both.** A frozen
pre-registration whose declared gate ladder is not the one the code runs is a worse artifact
than either the pre-amendment or the post-Prompt-E state.

---

## 6. Compliance with the task's rules

- Branch `contract/v6-amendment-2`, one commit. ✅
- YAML only among source files; `CLAUDE.md` (item 4) and this notes file are the other two.
  **No `.py` touched** — `git diff --stat` shows `CLAUDE.md` and `heston_benchmark_v6.yaml`
  only. ✅
- Existing structure, key ordering, comment style (`# MARKER — text`, `>` folded blocks) and
  quoting preserved; nothing reformatted, nothing removed, no existing line edited except the
  two CLAUDE.md bullets item 4 required. ✅
- Every new key carries an inline comment naming what it resolves. ✅
- `python -m pytest -q` → **240 passed**, unchanged from baseline. ✅
