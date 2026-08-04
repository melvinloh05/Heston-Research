# FIXLOG — audit fix batch 1 (pre-compute blockers)

Branch `fix/audit-batch-1` off `contract/v6-amendment`. Baseline before the batch:
**189 passed**. After the batch: **226 passed** (37 new tests, 0 pre-existing tests edited).

Nothing in this batch changes a numerical result. Every value was already correct; what
changed is where values come from and whether wrong states are detectable. The two
exceptions where a *verdict string* can now differ are **A2** and **A3** — both in the
conservative direction (an affirmative claim becomes `null`), and both only in states the
study hopes never to be in.

Per-finding: diff summary, test name, captured pre-fix failure, captured post-fix pass,
full-suite count after the commit. Raw captures are in `audit/fixlog/*.txt`.

---

## C1 + A1 + G1 — single-source the contract constants

**Commit** `4ba1a10` · 7 files, +573 / −83.

**Diff summary.**
- `Hedging_backtest.contract_thresholds(cfg) -> dict` (new): every pre-registered numeric
  the downstream layers act on, read from the resolved contract. Accepts a `resolve_config`
  output OR a bare contract dict (eval_greeks loads only the contract) — the three
  engine-supplement keys (`n_boot`, `baseline_arm`, `oracle_arm`) are simply absent in the
  latter case rather than silently defaulted.
- `analyze_results`: `DEFAULT_LEVEL` / `DEFAULT_BOOT` / `DEFAULT_SEED` **removed** (not kept
  as fallbacks). Every verdict function (`paired_ci_from_npz`, `confirmatory_cell`,
  `order_attribution`, `dose_response`, `ood_greek_thresholds`, `sakuma_null_consistency`,
  `mechanism_adjudication`, `goldilocks_bates`) now takes `thresholds` plus per-value
  arguments that default to `None` = "the contract". `run_analysis` resolves ONCE and
  threads the dict through all of them; `--contract` / `--engine` added to the CLI, and
  `--n-boot` / `--seed` default to the contract instead of a module literal.
  `ood_greek_thresholds`' single `red_threshold` split into `gamma_red_threshold` /
  `vega_red_threshold` — the contract declares the two minima separately.
- `gate_headroom.run_gate`: the hardcoded `0.10` at line 361 became the
  `spread_threshold_rel` parameter, defaulting to
  `oracle_headroom_gate.spread_threshold_rel`; it is returned in the result dict and the
  DECISION section of `headroom_report.md` now prints the value it actually used.
- `eval_greeks._threshold_rows(..., thresholds)`: the `0.15 / 0.15 / 0.10` at lines 288-289
  come from the contract; `run_greek_eval` derives them from its own `cfg` by default.
- `run_info_matching.run_saturation_sweep(tol=None)`: the plateau tolerance is read from
  `information_matching.plateau_tol`.

**Left as a literal (with `TODO(C1)`).** `confirmatory_rel_threshold = 0.10` — the contract
states ">=10% relative" in prose only. It now lives at exactly ONE site (inside
`contract_thresholds`) and is requested in `audit/contract_requests.md`.

**Left as a constant + parity test, deliberately.** Three values whose only call sites are
pure functions unit-tested without a contract in hand: `analyze_results._MISSPEC_FILTER` /
`_INMODEL_FILTER` (the cell *selector*), `eval_greeks.WING_LO/WING_HI`, and
`run_info_matching.PLATEAU_TOL` (the pure `plateau_multiplier` helper's default). Each is
asserted equal to the contract by the new parity test, which converts a silent divergence
into a loud test failure. Threading them would have required editing pre-existing tests.

**Test.** `test_contract_thresholds.py` (new, 30 cases): parametrized parity over every
threshold in `heston_benchmark_v6.yaml` + `hedging_config.yaml`, navigated independently out
of the YAML; plus consumption proofs — the OOD Greek verdict flips at exactly the contract's
reduction and parity values, `eval_greeks._threshold_rows` flips at the same, and one smoke
gate run shows the decision follows an injected threshold (`1e9` → no tier flagged, `-1e9` →
every tier flagged) while the default equals the contract.

**Pre-fix (captured, `audit/fixlog/c1_pre.txt`).**
```
_________________ ERROR collecting test_contract_thresholds.py _________________
test_contract_thresholds.py:38: in <module>
    TH = hb.contract_thresholds(CFG)
E   AttributeError: module 'Hedging_backtest' has no attribute 'contract_thresholds'
1 error in 0.89s
```

**Post-fix (`audit/fixlog/c1_post.txt`).**
```
..............................                                           [100%]
30 passed in 3.81s
```

**Full suite after the commit: 219 passed.**

---

## A2 — the mechanism reading was sign-blind

**Commit** `a5c12ca` · 2 files, +50 / −7.

**Diff summary.** `analyze_results._mechanism_reading`:
`present_i = _excludes_zero(...)` → `_excludes_zero(...) and g0["ci_hi"] < 0.0`; and
`widening`'s `abs(gmax["diff"]) > abs(g0["diff"])` → `gmax["ci_hi"] < 0.0 and
gmax["diff"] < g0["diff"] - 1e-12` (more negative). This is the same direction guard
`confirmatory_cell:312` and `order_attribution:335` already used. Docstring states the
requirement explicitly so it cannot be "simplified" back.

**Test.** `test_analyze_results.py::test_mechanism_wrong_direction_gap_is_not_a_channel` —
the exact MIRRORS of the existing four patterns must read `null`; a gap that *shrinks*
toward zero as TC rises is not widening; and the original negative patterns still read
`channel_i` / `channel_ii` (no over-correction). The pre-existing four-pattern test is
extended, not replaced.

**Pre-fix (captured, `audit/fixlog/a2_pre.txt`).**
```
        worse_i = ar._mechanism_reading(
            [_gap(0.0, +1.0, +0.5, +1.5), _gap(0.02, +1.0, +0.5, +1.5)],
            _tex(-0.1, -0.5, 0.3))
>       assert worse_i["reading"] == "null" and worse_i["present_i"] is False
E       AssertionError: assert ('channel_i' == 'null'
E         - null
E         + channel_i)
1 failed, 18 deselected in 0.10s
```

**Post-fix (`audit/fixlog/a2_post.txt`).** `4 passed, 15 deselected in 0.37s`
(all four mechanism tests, old and new).

**Full suite after the commit: 220 passed.**

---

## A3 — the confirmatory verdict ignored the seed count

**Commit** `d5909ea` · 2 files, +55 / −9.

**Diff summary.** `confirmatory_cell` gains `n_seeds_required` (default
`meta.seeds_confirmatory_cell` = 10). A short cell returns `verdict="null"` — deliberately
NOT `"fail"`: nothing about the hypothesis has been tested — with the shortfall leading the
notes ("NOT EVALUATED: 3 of the pre-registered 10 seeds present …"). `n_seeds` is added to
`VERDICT_COLS` as a structured column (it previously existed only inside free-text notes)
via a new `_verdict(..., n_seeds=None)` parameter, and is populated for
`order_attribution` and `sakuma_null_consistency` too.

**Test.** `test_analyze_results.py::test_confirmatory_requires_the_preregistered_seed_count`
— a 3-seed (resumable-partial) directory yields `null` with `row["n_seeds"] == 3`; the
complete 10-seed cell still passes with `n_seeds == 10`.

**Pre-fix (captured, `audit/fixlog/a3_pre.txt`).**
```
        v = ar.confirmatory_cell(d, tc=0.01, n_boot=200, seed=3)
>       assert v["verdict"] == "null"
E       AssertionError: assert 'pass' == 'null'
E         - null
E         + pass
1 failed, 19 deselected in 0.16s
```

**Post-fix (`audit/fixlog/a3_post.txt`).** `1 passed, 19 deselected in 0.23s`

**Full suite after the commit: 221 passed.**

---

## I1 — a row-capped rung could masquerade as an information plateau

**Commit** `6ef5b3f` · 2 files, +102 / −6.

**Diff summary.**
- `subsample_train` still caps at the frozen row count (no invented rows — the existing
  cap test is untouched) but now REPORTS it: the returned view carries
  `subsample_requested` and `subsample_capped`.
- `_one` records both on every per-seed row; they are written to
  `saturation_curve_per_seed.csv` (`n_train_rows_requested`, `subsample_capped`) and
  aggregated onto `saturation_curve.csv` (`subsample_capped` = any seed).
- `run_saturation_sweep`, after applying the plateau rule: if ANY width-1.0 rung at or
  below `plateau_index` was capped, it sets `plateau_capped=True`, forces
  `plateau_reached=False`, and warns (`RuntimeWarning`) that the flat segment is a row-cap
  artifact. Capping strictly ABOVE the plateau index is left alone — the plateau fired on
  full rungs, which is a genuine result.
- The reported paragraph, the console line and the per-seed `info_matched_budget.json`
  sidecar (`data_capped_multipliers`, `plateau_capped`, and a distinct `note`) all say so;
  the pre-existing "pinned at the cap" wording is preserved for the genuinely-not-reached
  case, so `test_sidecar_pins_at_cap_when_plateau_not_reached` still passes unchanged.

**Test.** `test_run_info_matching.py::test_row_cap_cannot_masquerade_as_a_plateau` (a ladder
requesting 10 000·m rows from a fixture holding ~25) plus
`test_uncapped_ladder_reports_no_capping` as the discriminating control.

**Pre-fix (captured, `audit/fixlog/i1_pre.txt`).** The unfixed sweep reports a reached
plateau on bit-identical data:
```
E       Failed: DID NOT WARN. No warnings of type (<class 'RuntimeWarning'>,) matching the regex were emitted.
E        Regex: capped
----------------------------- Captured stdout call -----------------------------
[info-matching] N=10000, multipliers=[1, 2] (cap 5), seeds=[0]; plateau m=2 (reached). ...

>       assert res["plateau"]["capped_rungs"] == []
E       KeyError: 'capped_rungs'
2 failed, 11 deselected, 1 warning in 2.47s
```

**Post-fix (`audit/fixlog/i1_post.txt`).** `2 passed, 11 deselected, 1 warning in 2.44s`

**Full suite after the commit: 223 passed.**

---

## Q3 — λ_pde is now sourced from the baseline arm

**Commit** `49e9607` · 3 files, +131 / −21.

**Diff summary.** `train.py::_run_select_lambdas` implements the amended contract's
`lambda_selection` section as a STAGED, validation-only search:

1. **stage 1** — 1-D over `lambda_pde.candidates`, scored on
   `lambda_selection.lambda_pde.source_arm` (`standard_pinn`), each candidate fitting that
   arm on its own train split and scoring `_val_greek_score` on its own val split;
2. **stage 2** — 2-D over (`lambda_gamma`, `lambda_vega`) candidates on
   `lambda_{gamma,vega}.source_arm` (`rung3_delta_gamma_vega`) at the stage-1 `lambda_pde`.

Both stages run behind the same `LockedTestSet` and reuse `train_pinn.select_lambdas`
unchanged as the guarded grid-search primitive (called twice; its docstring now warns
against collapsing the two calls back into one joint search). Candidate lists come from the
contract when not given on the CLI. `lambdas_selected.yaml` gains `sources`
(per-λ source arm), a `selection` block (protocol, tune_on, candidates, seed, steps) and
`scores_table_pde` (stage 1); `scores_table` is now the stage-2 table at the selected
`lambda_pde`. `_apply_lambdas` is unchanged — the shared λ_pde still reaches every
PDE-live arm, which is the intended control; only its SOURCE moved.

**Test.** `test_train.py::test_select_lambdas_is_staged_and_records_its_source_arms` — the
emitted yaml records a source arm per λ matching the contract, the two arms differ, stage 1
is 1-D on the baseline (`scores_table_pde` carries only `lambda_pde`/`score`), stage 2 is
the 2-D grid at the selected `lambda_pde`, and that λ_pde reaches `standard_pinn` and
`rung3` identically through the untouched `_apply_lambdas`.

**Pre-fix (captured, `audit/fixlog/q3_pre.txt`).**
```
>       assert lam["sources"]["lambda_pde"] == lam_sel["lambda_pde"]["source_arm"]
E       KeyError: 'sources'
----------------------------- Captured stdout call -----------------------------
selected lambda_pde=1.0 lambda_gamma=1.0 lambda_vega=0.3 (lambda_delta fixed 1.0); scored 4 combos on validation
1 failed, 13 deselected, 1 warning in 2.26s
```
(the "4 combos" is the joint 2×1×2 rung3 grid the fix replaces)

**Post-fix (`audit/fixlog/q3_post.txt`).** `1 passed, 13 deselected, 2 warnings in 2.19s`

**Full suite after the commit: 224 passed.**

> **Not yet done, and out of scope for this batch:** λ selection has not been RERUN, and
> `lambdas_selected.yaml` does not exist. The protocol is now correct; the artifact is still
> to be produced (GPU spend — needs sign-off).

---

## Two tests with no fix attached

**Commit** `672cc6f` · 1 file, +71.

Neither can show a pre-fix failure (there is no fix), so both were checked for
non-vacuity instead — captured in `audit/fixlog/extra_probe.txt`:

```
realized dt = 0.003953488; contract dt_realized 0.003953488; |diff| 3.72e-10
vs OLD declared 1/252 = 0.003968254; |diff| 1.48e-05 (> 1e-6 -> test would FAIL pre-amendment)
regime=baseline                 feller=1.778 states=5504 saw exact v==0: False
regime=feller_violating_volvol  feller=0.444 states=5504 saw exact v==0: True
```

- **`test_realized_dt_matches_the_contract`** — asserts `|T'/n_steps - dt_realized| < 1e-6`,
  that `n_steps == rebalancing.n_steps` (43), that the simulator's actual `np.diff(times)`
  is `dt_realized` throughout, and that `times[-1] == T'` exactly. **It passes**: the Q1
  amendment and the code agree. It would have failed by 1.5e-5 against the pre-amendment
  `dt: 0.003968`.
- **`test_qe_exponential_branch_atom_reaches_the_providers`** — the baseline regime produces
  **0** exact zeros in 5504 provider-visible states, which is exactly why no engine test
  exercised the QE exponential branch's atom. The test swaps in the contract's own
  `feller_violating_volvol` anchor (Feller 0.44, in-memory only) at smoke size, asserts the
  provider genuinely receives exact `v == 0`, that it stays finite (asserted inside the
  provider on every call), and that every emitted `cvar` / `mean_pnl` is finite.

**Full suite after the commit: 226 passed.**

---

## Rules compliance

- One commit per finding, prefixed with the finding ID. ✅
- `heston_benchmark_v6.yaml`, `hedging_config.yaml`, `pinn_config.yaml` untouched
  (`git diff a5a08a1..HEAD -- '*.yaml'` is empty; `main..HEAD` shows only the human's own
  amendment commit `a5a08a1`, which this branch is based on). ✅
- Every fix is test-first with a captured pre-fix failure. ✅
- No pre-existing test was edited to make a fix pass; no pre-existing test failed. ✅
- No refactors beyond the named defects; everything else noticed went to
  `audit/FINDINGS_ADDENDUM.md`. ✅

---
---

# FIXLOG — audit fix batch 2 (reporting integrity and provenance)

Branch `fix/audit-batch-2` off `fix/audit-batch-1` (`8cb1d37`). Baseline before the batch:
**226 passed**. After the batch: **240 passed** (14 new tests).

**Nothing in this batch changes a number that is currently correct.** Every fix changes what
happens when something is missing, corrupt, or partial. Three exceptions worth naming, all
in states the study hopes never to be in:

- **A4** can now emit a verdict string (`"error"`) that did not previously exist — it
  replaces a `"null"` that meant the opposite thing.
- **Q4** changes `rel_improvement` only when the baseline CVaR is negative, where the old
  value had the WRONG SIGN.
- **G2** adds a measurement, and that measurement came back non-trivial. See its entry.

Per-finding: diff summary, test name, captured pre-fix failure, captured post-fix pass,
full-suite count after the commit. Raw captures in `audit/fixlog/*.txt`.

---

## X1 — a missing exhibit cell was drawn as a hard 0.0

**Commit** `3ccbe9d` · 4 files, +219 / −22.

**Diff summary.** New `exhibits._nan_if_missing(v)`: `_num(v)`, with a blank/non-finite cell
as NaN rather than 0.0 (matplotlib omits NaN bars; the 2×2 inset in the same function already
did exactly this and prints `"n/a"`). Applied at the five named sites — E2's T_ex value and
err, E4's `gap_closed_mean`, E4's cost/directional split, E4's vanna inset value and err.

**Deliberately NOT changed.** `exhibits.py:291, 320, 342, 429, 499` also carry `or 0.0`, but
those are error bars whose VALUE is present and only the seed-std is blank, and a NaN `yerr`
drops the point itself in some matplotlib paths. Not among the finding's five sites.

**Test.** `test_exhibits.py::test_e2_missing_t_ex_is_not_drawn_as_zero`,
`::test_e4_missing_gap_closed_is_not_drawn_as_zero`,
`::test_e4_missing_vanna_reduction_is_not_drawn_as_zero` — they spy on `Axes.bar` and assert
the DRAWN height (not the CSV cell, which was always right) is NaN for the blank arm and
unchanged for the populated ones. The fixture was EXTENDED with an optional per-method
`blanks` argument that defaults to populating every field, so no pre-existing test moves.

**Pre-fix (`audit/fixlog/x1_pre.txt`).**
```
E       AssertionError: blank gap_closed_mean drawn at 0.0; 0.0 reads as 'closed none of
        the oracle gap', an affirmative negative result
E       assert False
E        +  where False = _is_nan(0.0)
3 failed, 16 deselected in 0.65s
```

**Post-fix (`audit/fixlog/x1_post.txt`).** `3 passed, 16 deselected in 0.75s`

**Full suite after the commit: 229 passed.**

---

## A4 — `_guard` turned any exception into "null"

**Commit** `c2a801e` · 4 files, +129 / −11.

**Diff summary.** New `analyze_results._failure_verdict(exc, tid, cell)` classifies the
failure: `FileNotFoundError` (module constant `_ABSENCE_EXC`) keeps `verdict="null"` —
absence, nothing evaluated, the legitimate non-alarming state `ood_greek_thresholds` already
uses that word for. Everything else — a renamed column, a shape mismatch, a failed import —
gets a distinct `verdict="error"` with an `"ANALYSIS ERROR (artifact present but unusable)"`
note. Applied at all four sites (`_guard`, `dose_response`, `goldilocks_bates`).

`KeyError` is deliberately NOT in `_ABSENCE_EXC` globally — a KeyError from a renamed CSV
column is a defect. It is treated as absence at exactly one site, `_measured_label_error`,
where `SobolevPINN.load_arm` does `raw["arms"][arm]` on an arm that is not in
`pinn_config.yaml`. That handler was narrowed from `except Exception` to
`(FileNotFoundError, KeyError)`, so a corrupt-but-present label artifact now propagates to
the error verdict instead of silently becoming a dose reference line.

**Test.** `test_analyze_results.py::test_corrupt_artifact_is_an_error_verdict_not_a_null` —
a syntactically valid npz whose two arms carry different-length PnL arrays (tripping the
engine's CRN shape assertion in `paired_bootstrap_cvar_diff`) yields `verdict="error"` with
`AssertionError` in the notes, in memory and in `threshold_verdicts.csv`; an empty PnL dir
still yields `"null"`. The pre-existing `test_run_analysis_missing_artifacts_degrade_to_null`
is untouched and passes.

**Pre-fix (`audit/fixlog/a4_pre.txt`).**
```
>       assert conf_v["verdict"] != "null", (
            "a corrupt artifact rendered as the same 'null' used for absence")
E       AssertionError: a corrupt artifact rendered as the same 'null' used for absence
E       assert 'null' != 'null'
1 failed, 20 deselected in 0.24s
```

**Post-fix (`audit/fixlog/a4_post.txt`).** `1 passed, 20 deselected in 0.24s`

**Full suite after the commit: 230 passed.**

---

## T2 — `last.pt` was labelled `matched_epochs` even when the arm early-stopped

**Commit** `fb2a078` · 5 files, +165 / −8.

**Diff summary.** `train_pinn.train_model` always emits `checkpoints["last"]`, and adds
`checkpoints["matched_epochs"]` (a copy of the same record) ONLY when `reached_max_steps`.
`train.py`'s completion line names which of the two this `last.pt` is
(`"last, EARLY-STOPPED"` vs `"matched_epochs"`).

**The default early-stop behaviour is deliberately unchanged** — `train.py`'s
`early_stop = not args.matched_epochs` is a run-book decision, not a code one. The `last.pt`
FILE is still written in both modes; only the runlog label moved.

**Test.** `test_train.py::test_early_stopped_run_has_no_matched_epochs_checkpoint` — a
default-mode `train.main` run forced to early-stop (a non-improving `_eval_val` plus a
shrunk val cadence, since the contract's `val_every 500 / patience 10` needs 5500 steps to
trip) has `"last"` with `reached_max_steps` False and no `"matched_epochs"` key.

**One pre-existing assertion updated, unavoidably.** The fix changes the runlog schema, so
`test_matched_epochs_checkpoint_exists_alongside_best`'s
`set(ck) == {"best", "matched_epochs"}` became `{"best", "last", "matched_epochs"}`. Its
substantive assertions (path, step 30, `reached_max_steps` True) are unchanged, and it gains
`ck["last"] == ck["matched_epochs"]` — `matched_epochs` is the same checkpoint, not a second
file.

**Left alone (YAML, not mine to edit).** `pinn_config.yaml:94-95` still documents the old
label: *"last.pt (matched_epochs) is ALSO recorded for the contract's report_both"*. That
comment is now stale for default runs. Flagged, not changed.

**Pre-fix (`audit/fixlog/t2_pre.txt`).**
```
>       assert "matched_epochs" not in ck, (
E       AssertionError: an early-stopped checkpoint was filed as matched_epochs; the
        report_both table would compare unequal budgets as if matched
E       assert 'matched_epochs' not in {'best': {...}, 'matched_epochs': {'path': 'last.pt',
        'reached_max_steps': False, 'step': 10, 'val_total': 2.0}}
2 failed, 13 deselected, 4 warnings in 2.75s
```
(`reached_max_steps: False`, filed as `matched_epochs` — the defect verbatim.)

**Post-fix (`audit/fixlog/t2_post.txt`).** `2 passed, 13 deselected, 4 warnings in 2.32s`

**Full suite after the commit: 231 passed.**

---

## R1 — `resolved_config.yaml` recorded the last cell's trimmed config

**Commit** `c03fb60` · 4 files, +86.

**Diff summary.** `run_hedging._run_program` calls `hb.log_resolved_config(prog, run_root)`
once after the cell loop, so the untrimmed program-level config is the surviving copy. The
per-cell trimming itself is intended and unchanged (`_resolved_config_hash`'s docstring
already anticipates the hash mismatch). Nothing numerical moves.

**Test.** `test_run_hedging.py::test_resolved_config_describes_the_program_not_the_last_cell`
— the 10-seed, 2-magnitude confirmatory run's surviving `resolved_config.yaml` carries all
ten seeds and both magnitudes, still records the program's own trim (combined only, no
cross-model), and re-enumerates through `hb._iter_sim_cells` to the exact 20 cells the run
executed.

**Pre-fix (`audit/fixlog/r1_pre.txt`).**
```
>       assert cfg["derived"]["seeds"] == CONFIRMATORY_SEEDS          # all 10, not the last
E       assert [51] == [42, 43, 44, 45, 46, 47, ...]
E         At index 0 diff: 51 != 42
1 failed, 11 deselected, 2 warnings in 3.31s
```
(`seeds: [51]` — exactly the state the finding predicted.)

**Post-fix (`audit/fixlog/r1_post.txt`).** `12 passed, 2 warnings in 5.13s`

**NOT addressed, out of scope for a one-commit fix.** The finding's smaller sibling: a HARD
INTERRUPTION mid-loop still leaves single-cell `headline_delta_only_per_seed.csv` / `_agg.csv`
on disk, since those are also engine-overwritten per cell and only restored from
`_rows_master.csv` at the end. A resumed run repairs them; a crashed one does not.

**Full suite after the commit: 232 passed.**

---

## G2 — the delta clip silently shrank the delivered corruption

**Commit** `903003f` · 5 files, +157 / −7.

**Diff summary.** The clip is UNCHANGED, as instructed. `NoisyOracleProvider` counts, per arm
over the whole run, the fraction of delta evaluations `_DELTA_CLIP` bound on
(`np.count_nonzero((raw < lo) | (raw > hi)) / raw.size`), exposed as `.clipped_fraction` and
carried into every `run_gate` summary row (`clipped_frac`), the `headroom_report.md` table,
and the result dict — with a module-level `_CLIPPED_NOTE` stating what a non-zero value means,
so a reader cannot meet the number without it. `evaluate()` stays a pure function of state;
the counters are bookkeeping, and the frozen-field determinism test still passes unchanged.

`clipped_frac` is per ARM, repeated on each tc row: positions are built once per method and
reused across tiers, so there is no per-tc clipping to report.

> ### This answers the audit's one "unable to verify" item — and the answer is NOT inert.
>
> Measured (`audit/fixlog/g2_measured.txt`), field mode, confirmatory cell, n_paths = 256,
> 2 seeds:
>
> | sigma_rel | clipped_frac |
> |---|---|
> | 0.1 | 0.061 |
> | 0.2 | 0.215 |
> | 0.4 | **0.780** |
> | 0.8 | **0.960** |
>
> iid mode clips far less (0.001 at 0.1, 0.117 at 0.8). The upper rungs of the default sigma
> ladder deliver substantially less corruption than their label, so the field-mode ceiling
> those rows report is understated. The direction is safe (a conservative gate), but the
> sigma axis of the reported table is not the axis it is labelled with, at 0.4 and above.
>
> **This commit only reports it.** No threshold, no clip and no decision rule changed.
> Whether the ladder should be re-centred on its low end, or the clip widened, is a human
> call — see "Open for the human" at the end of this batch.

**Test.** `test_gate_headroom.py::test_binding_delta_clip_is_reported` (a large-sigma provider
over a deep-ITM/OTM state cloud reports a non-zero fraction; sigma=0 reports exactly 0; NaN
before the first `evaluate`) and `::test_clipped_fraction_reaches_the_summary_and_report`
(every summary row carries it, it is constant per arm across tc tiers, and it reaches the
report).

**Pre-fix (`audit/fixlog/g2_pre.txt`).**
```
>       assert p.clipped_fraction != p.clipped_fraction          # NaN before any eval
E       AttributeError: 'NoisyOracleProvider' object has no attribute 'clipped_fraction'
>       assert all("clipped_frac" in s for s in res["summary"])
E       assert False
2 failed, 6 deselected in 2.02s
```

**Post-fix (`audit/fixlog/g2_post.txt`).** `2 passed, 6 deselected in 2.30s`

**Full suite after the commit: 234 passed.**

---

## T1 — the pilot→gate handoff was a human copy-paste off a known-wrong first line

**Commit** `e362ad0` · 5 files, +253 / −7.

**Diff summary.**
- `gate_headroom.sigma_gamma_pilot_from_runlog(path)` reads `sigma_gamma_pilot` BY NAME from
  a `--pilot` runlog.json, raising with an explicit *"do NOT substitute
  sigma_gamma_pilot_prefix_bug"* message when the key is absent, and on a non-finite or
  non-positive value. Exposed as `--sigma-gamma-from-runlog <runlog.json>`, mutually
  exclusive with `--sigma-gamma`, and echoed before the run.
- `train.py`: the BEFORE line moves behind `--show-prefix-bug` (tagged
  `[DO NOT FEED TO THE GATE]` when shown). It is not lost — it is always written to the
  runlog as `sigma_gamma_pilot_prefix_bug{,_relative}`. Default stdout now carries only the
  correct value plus the exact gate command to run.
- `headroom_report.md`'s "pilot-calibrated point" instruction names the runlog flag instead
  of a hand-typed float.

**Test.** `test_train.py::test_gate_reads_sigma_gamma_from_the_runlog` — a real `--pilot`
runlog drives `gh.main` with `run_gate` stubbed; the gate receives exactly
`runlog["sigma_gamma_pilot"]`. At smoke size best == last step so the two values coincide and
a wrong pick would hide, so a doctored runlog forces them apart (0.25 vs 0.40) and proves the
read is by name. A runlog with no pilot entry raises rather than silently falling back to the
sweep.

**One pre-existing assertion updated, unavoidably.** The fix's whole point is the BEFORE line
leaving default stdout, so `test_pilot_prints_finite_sigma_gamma`'s
`assert "BEFORE fix" in printed` became `not in`, plus a `--show-prefix-bug` run asserting it
comes back and a check that the runlog retains the value.

**Pre-fix (`audit/fixlog/t1_pre.txt`).**
```
>       assert "sigma_gamma_pilot BEFORE fix" not in printed
E       AssertionError: 'sigma_gamma_pilot BEFORE fix' is contained here:
E           sigma_gamma_pilot BEFORE fix (last-step model, gamma (arm label)) = 0.0157787 ...
E           sigma_gamma_pilot AFTER  fix (best-step model, gamma_ref) = 0.0157787 ...
2 failed, 14 deselected, 6 warnings
```

**Post-fix (`audit/fixlog/t1_post.txt`).** `2 passed, 14 deselected, 6 warnings in 2.95s`

**Full suite after the commit: 235 passed.**

---

## A5 — `dose_response` fitted on common seeds but reported `cvar_mean` on each arm's own

**Commit** `e98adb9` · 4 files, +226 / −9.

**Diff summary.** One helper `_arm_ys(i)` returns an arm's y values on the seed set the FIT
uses (`common` when it exists), and both the fit and the reported point now go through it.
`cvar_seed_std` moves with the mean — the error bar on a point must span the same seeds as
the point. `n_seeds` still reports the arm's OWN count so the shortfall stays visible, and a
new `n_seeds_common` column (added to `DOSE_COLS`) records the fit set. Reference arms
(`gradient_penalty_only`, off the label-error axis) carry `""` for it — they are not in the
fit, so no common-seed restriction is claimed. `bs_gap` differences two `cvar_mean` values, so
it inherits the fix. Identical whenever all dose arms have all seeds, the intended state.

**Test.** `test_analyze_results.py::test_dose_cvar_mean_uses_the_same_seeds_as_the_fit`
(`sigma_050` short two seeds; the discriminating arm is `shuffled`, which HAS all five and
varies with the seed — own-seed mean 50.0 vs common-seed mean 45.0) and
`::test_dose_bs_gap_uses_the_common_seeds` (common-seed gap 18.0 vs the pre-fix mixed-seed
16.0; the label-error axis is stubbed there because the fake labels npz cannot serve the
`bs_gamma` arm's implied-vol inversion, and only the seed set is under test).

**Pre-fix (`audit/fixlog/a5_pre.txt`).**
```
>       assert byarm["shuffled"]["cvar_mean"] == pytest.approx(45.0)
E       assert 50.0 == 45.0 ± 4.5e-05
>       assert byarm["sigma_000"]["cvar_mean"] == pytest.approx(12.0)
E       assert 14.0 == 12.0 ± 1.2e-05
```

**Post-fix (`audit/fixlog/a5_post.txt`).** `2 passed, 21 deselected in 0.91s`

**Full suite after the commit: 237 passed.**

---

## H1 — realized dt (no code change)

Per the Q1 amendment, `dt` is a DERIVED quantity (`rebalancing.dt_realized = 0.003953488`,
`n_steps = 43`) and the code and contract agree. **No code change was expected and none was
made.** The batch-1 test
`test_hedging_backtest.py::test_realized_dt_matches_the_contract` was re-run against the
amended contract and **passes**: it asserts `|T'/n_steps − dt_realized| < 1e-6`,
`n_steps == rebalancing.n_steps`, that the simulator's own `np.diff(times)` is `dt_realized`
throughout, and that `times[-1] == T'` exactly. Confirmed as part of every full-suite run in
this batch.

---

## Q4 — `rel_improvement` divided by a signed CVaR

**Commit** `f6ab4b9` · 5 files, +112 / −6.

**Diff summary.** New `Hedging_backtest.rel_improvement(cvar_a, cvar_b)` is the single
definition both the engine and `analyze_results._pooled_stratified` call (the expression was
duplicated verbatim): `|cvar_b|` in the denominator, NaN at exactly 0, and a `RuntimeWarning`
naming the state when `cvar_b < 0`. **Both** options the question offered, because they answer
different halves: the `abs()` stops the sign inversion, the warning stops the state passing
unnoticed. Numerically inert wherever `cvar95(loss) > 0`, which is every state the study
anticipates.

Measured on the new test: a 1.0-unit improvement against a −4.02 baseline previously read as
`rel = −0.249`, i.e. `confirmatory_cell`'s `rel >= 0.10` would have called a real improvement
a failure.

**Test.** `test_analyze_results.py::test_rel_improvement_does_not_invert_on_a_negative_baseline_cvar`
— positive `rel_improvement` and the warning at both the engine and the pooled-analysis site;
the ordinary positive-loss-tail case reproduces the unchanged signed formula exactly.

**Pre-fix (`audit/fixlog/q4_pre.txt`).**
```
>       assert st["rel_improvement"] > 0.0, (
E       AssertionError: an improvement against a negative baseline CVaR read as a regression
E       assert -0.2487825561423195 > 0.0
```

**Post-fix (`audit/fixlog/q4_post.txt`).** `1 passed, 23 deselected, 2 warnings in 0.15s`

**Full suite after the commit: 238 passed.**

---

## Q6 — anchor-grid `params` was an unnamed vector

**Commit** `66f6bdc` · 5 files, +105 / −4.

**Diff summary.** `make_datasets` writes `param_names` alongside `params`, exactly as the
label artifact does (`make_labels.py:187`), plus a module-level `_ANCHOR_GRID_KEYS` schema
asserted at write time so a key cannot be dropped silently. `eval_greeks.eval_arm_on_regime`
reads the grid's own `param_names` and binds the regime dict off it; a grid whose names are
not the five Heston parameters raises rather than being zipped positionally into the wrong
slots.

**Test.** `test_eval_greeks.py::test_anchor_grid_params_are_bound_by_name` — `params` and
names permuted TOGETHER give bit-identical metrics, the writer's schema carries the key, and a
mislabelled grid raises. `_fake_grid` gains a `param_names` argument defaulting to the real
tuple, so no existing test moves.

**Pre-fix (`audit/fixlog/q6_pre.txt`).**
```
>           assert m_perm[g]["rmse"] == pytest.approx(m_ref[g]["rmse"]), g
E           AssertionError: price
E           assert 31.642108191452852 == 31.433799676588464 ± 3.1e-05
```

**Post-fix (`audit/fixlog/q6_post.txt`).** `1 passed, 10 deselected in 8.18s`

**Full suite after the commit: 239 passed.**

---

## Q7 — the Spearman seed-bootstrap used a bare seed

**Commit** `78474e0` · 4 files, +55 / −1.

**Diff summary.** `_spearman_seed_bootstrap`'s `np.random.default_rng(int(seed))` becomes
`np.random.default_rng([int(seed), _STREAM_SPEARMAN])`, with a new `_STREAM_SPEARMAN = 8`
distinct from every stream id in use (0–3 engine and providers, 7 pooled, 11/12 gate).
Numerically inert — the two were already different streams, and the two bootstraps resample
different things (seeds vs paths). CLAUDE.md's frozen-exception list does not name this site,
which marks it as oversight rather than intent.

**Test.** `test_analyze_results.py::test_spearman_bootstrap_uses_the_stream_convention` —
spies on `np.random.default_rng` and asserts the `[seed, _STREAM_SPEARMAN]` form is what gets
constructed, and the bare-seed form never is.

**Pre-fix (`audit/fixlog/q7_pre.txt`).**
```
>       assert ar._STREAM_SPEARMAN != ar._STREAM_POOLED
E       AttributeError: module 'analyze_results' has no attribute '_STREAM_SPEARMAN'
```

**Post-fix (`audit/fixlog/q7_post.txt`).** `1 passed, 24 deselected in 0.47s`

**Full suite after the commit: 240 passed.**

---

## Q5 — left alone, as instructed

The decorative `LockedTestSet` guard around λ selection is untouched. It is a run-book
question (should production selection runs be required to pass `--anchor-grids <dir>`), not a
code one, and there is no actual leakage.

---

## Open for the human

1. **G2's measured clipped fractions.** 78% at `sigma_rel = 0.4` and 96% at 0.8 means the top
   half of the default `_SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)` ladder is not delivering
   the corruption its label claims. The gate is conservative either way, so nothing is
   invalidated — but before the gate is run for the go/no-go, decide whether to re-centre the
   ladder on its low end, widen `_DELTA_CLIP`, or simply report the fractions alongside the
   table. Changing the clip or the ladder is a design decision, not a fix.
2. **`pinn_config.yaml:94-95`** documents the pre-T2 checkpoint label. YAML is read-only to
   the agent; the comment should say `last.pt` is `matched_epochs` only under
   `--matched-epochs`.
3. **Verdict consumers must learn `"error"`.** Any downstream reader that branches on
   `pass` / `fail` / `null` now has a fourth value to handle. Nothing in this repo does today
   (the memo prints the string verbatim), but a paper-side table might.

---

## Rules compliance

- One commit per finding, prefixed with the finding ID (10 commits: X1, A4, T2, R1, G2, T1,
  A5, Q4, Q6, Q7; H1 is a no-change confirmation, recorded above). ✅
- `heston_benchmark_v6.yaml`, `hedging_config.yaml`, `pinn_config.yaml` untouched
  (`git diff 8cb1d37..HEAD -- '*.yaml'` is empty). ✅
- Every fix is test-first with a captured pre-fix failure. ✅
- No refactors beyond the named defects. ✅
- **Two pre-existing assertions edited**, both unavoidable because the fix changes the
  observable the assertion names, both documented in their entries above: T2's runlog-schema
  set, and T1's BEFORE-line stdout check. In both cases the test's substantive assertions are
  unchanged and were EXTENDED, not weakened. No other pre-existing test was touched; none
  failed.
- `audit/FIXLOG.md` appended, not replaced. ✅

---
---

# FIXLOG — audit fix batch 3 (close the contract/code divergences before the freeze)

Branch `fix/audit-batch-3` off `contract/v6-amendment-2`. Baseline before the batch:
**240 passed**.

Contract amendment #2 (`audit/contract_amendment_2_notes.md`) declared keys no `.py` file
read — its own §5 names the live divergence and recommends withholding the freeze tag until
it closes. This batch closes it: the gate ladder, both effective-sigma quantities, the
region-of-validity consequence, the confirmatory 10%, the `null`-means-two-things collision,
the pre-flight cap check, and the two addendum scope calls.

Per item: diff summary, test name, captured pre-fix failure, captured post-fix pass,
full-suite count after the commit. Raw captures are in `audit/fixlog/*.txt`.

---

## ITEM 1 (E1) — the gate ladder comes from the contract

**Diff summary.**
- `gate_headroom._SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)` **deleted**. In its place a
  comment naming what it got wrong: it swept 0.8 (which AM2-3a deletes) and treated 0.4 as
  a decision rung (which AM2-3a demotes to diagnostic-only).
- `Hedging_backtest.contract_thresholds`: two new keys, `gate_sigma_rel_decision` and
  `gate_sigma_rel_diagnostic`, from `oracle_headroom_gate.sigma_rel_ladder.{decision,
  diagnostic}`. Both added to `test_contract_thresholds._EXPECTED`.
- `gate_headroom._resolve_ladder(cfg, sigma_rel_list, sigma_rel_diagnostic)` (new) returns
  `[(sigma_rel, decision_eligible), ...]` ascending plus the source. `sigma_rel_list=None`
  (the new default, and what the CLI passes unless a human overrides) = the CONTRACT ladder.
- **The trap the amendment notes flagged (§4.1), handled:** eligibility is carried PER ARM
  all the way to the decision scan. Arms are 4-tuples `(label, sigma_rel, sigma_gamma_abs,
  decision_eligible)`; every record and summary row carries `rung_role` (`decision` /
  `diagnostic`) and `decision_eligible`; the `next(...)` that picks the decision row now
  requires `s["decision_eligible"]`. Diagnostic rungs are still swept, hedged, written to
  `headroom.csv` and printed in the report table — they are excluded from the DECISION scan
  and NOWHERE else.
- Report: a ladder line naming source, decision rungs and DIAGNOSTIC-ONLY rungs; a `role`
  column in the spread table; the DECISION section states the exclusion and why
  (`region_of_validity`). CLI gains `--sigma-rel-diagnostic`; `--sigma-rel` defaults to
  None = contract.
- An explicit `sigma_rel_list` is treated as an OPERATOR OVERRIDE (those rungs are decision
  rungs, `sigma_rel_diagnostic` defaults to empty) — but passing a rung the CONTRACT calls
  diagnostic raises a loud `RuntimeWarning` naming it, and the report prints
  "**LADDER OVERRIDDEN — this is NOT the contract ladder**". This is what keeps the three
  pre-existing tests that deliberately pass `sigma_rel_list=(0.4,)` meaningful instead of
  silently re-promoting 0.4; their warnings in the suite output are that signal firing.

**Tests.** `test_gate_headroom.py::test_sigma_ladder_comes_from_the_contract` (no module
literal survives; `_resolve_ladder` returns the contract's decision/diagnostic split; 0.8 is
absent from what a default invocation executes) and
`::test_diagnostic_rung_cannot_fire_the_decision_scan` (in-memory ladder decision=[0.05],
diagnostic=[0.8]; the 0.8 arm clears the scan's own predicate — spread >= threshold AND
`ci_excludes_zero_frac == 1.0` — and is asserted NEVER to be the decision row at any tier,
while its `diagnostic` label reaches `headroom.csv` and the report).

**Pre-fix (`audit/fixlog/e1_pre.txt`).**
```
>       assert {s["arm"] for s in res["summary"]} == {"s0.05", "s0.8"}
E       AssertionError: assert {'s0.1', 's0....s0.4', 's0.8'} == {'s0.05', 's0.8'}
E         Extra items in the left set:
E         's0.2'
E         's0.1'
E         's0.4'
2 failed, 8 deselected in 3.59s
```
(the default invocation swept the Python literal, not the contract ladder.)

**Post-fix (`audit/fixlog/e1_post.txt`).** `2 passed, 8 deselected in 2.64s`

**Full suite after the commit: 244 passed.**

---

## ITEM 3 (E3) — both post-clip quantities, and the pilot compared against the right one

**Diff summary.**
- `gate_headroom._base_delta_cloud(base, states, K)` (new): one pass of UNCORRUPTED delta
  AND gamma on the reference cloud, shared by every arm.
- `gate_headroom.effective_sigmas(provider, states, base_cloud)` (new): the two DELIVERED
  quantities the contract declares, measured on `err = clip(delta + eta) - delta` over the
  reference cloud —
  `sigma_delta_effective = std(err)` (delta units) and
  `sigma_gamma_effective = std(d err/dS)` (gamma units). Contract field names, verbatim.
  `d err/dS` is taken ANALYTICALLY and almost everywhere: `eta_dS` where the clip is slack,
  exactly `-Gamma` where it binds. A finite difference was tried first and rejected — it
  straddles the clip boundary on an O(h) set of states and turns each kink into an ~eta/h
  spike (+11% at sigma_rel = 0.1, h = 1e-3 of the S range), i.e. it measures the estimator,
  not the corruption. iid mode returns NaN for the gamma quantity WITH a reason string (a
  per-call redraw has no d/dS); its measurement draws come from a dedicated `_STREAM_EFF`
  so measuring can never advance the strawman's hedging stream.
- Both quantities are attached to every `records` row, every `summary` row, `headroom.csv`
  and the report table (which now shows NOMINAL and both EFFECTIVE columns), plus a new
  `_EFFECTIVE_NOTE` spelling out the units and why the pilot uses the gamma one.
- `_pilot_comparison(cfg, sigma_gamma_abs, eff, clipped_frac)` (new) + result key
  `pilot_comparison`: reads the comparison FIELD NAME from the contract
  (`effective_sigma_reporting.compare_pilot_against` via the new
  `contract_thresholds["gate_compare_pilot_against"]`, parity-tested) and carries
  `{compare_against, nominal, effective, sigma_delta_effective, sigma_gamma_effective,
  clipped_frac}`. The report gets a "Pilot point vs the DELIVERED corruption" section.

**Tests.** `test_effective_sigmas_track_the_nominal_until_the_clip_bites` (unclipped arm
delivers its label to 2%; 3x-rms arm delivers < 0.5x in BOTH quantities),
`test_effective_sigmas_reach_summary_csv_and_report`, and
`test_pilot_is_compared_against_sigma_gamma_effective` — which asserts on the VALUE the
comparison carries (`pc["effective"] == eff["sigma_gamma_effective"] !=
eff["sigma_delta_effective"]`), not merely that both fields exist.

**Pre-fix (`audit/fixlog/e3_pre.txt`).**
```
>           assert all(key in s for s in res["summary"])
E           assert False
...
>       pc = res["pilot_comparison"]
E       KeyError: 'pilot_comparison'
3 failed, 10 deselected in 4.36s
```

**Post-fix (`audit/fixlog/e3_post.txt`).** `4 passed, 10 deselected in 5.06s`

**A MEASURED SURPRISE, recorded as `audit/FINDINGS_ADDENDUM.md` N8 (P1).** Emitting the
quantity immediately falsified the assumption it was introduced to report. Across the whole
DECISION ladder the delivered gamma scale is LARGER than the nominal label — 1.21x at
sigma_rel 0.10, 1.41x at 0.15, 1.54x at 0.20 — and only drops below it past ~96% clipped.
Where the clip binds, the corrupted hedger is FLAT in S, so its gamma error there is the
ORACLE's own `-Gamma`, bigger than the calibrated field's. G2's "delivered error is smaller,
so the gate is conservative" (still quoted in the contract's AM2-3 block comment) is
therefore not true in the band the gate decides on. Nothing was widened or re-tuned: the
clip and the ladder are untouched, `_CLIPPED_NOTE` gained the measured caveat, and
`test_effective_gamma_can_EXCEED_the_nominal_in_the_decision_band` locks the direction into
the suite. **The contract-side consequence is the human's call (see "Open for the human").**

**Full suite after the commit: 249 passed.**

---

## ITEM 4 (E4) — a pilot outside the region of validity is INCONCLUSIVE, not None

**Diff summary.**
- `contract_thresholds["gate_clipped_frac_max"]` (new, parity-tested) from
  `oracle_headroom_gate.region_of_validity.clipped_frac_max`.
- `_pilot_comparison` gained `clipped_frac_max` and `in_region`. The region is DECLARED in
  terms of `clipped_frac` (AM2-3c `statistic: clipped_frac`), so that is the test: the pilot
  arm's own clipped fraction over its hedging life vs the contract bound.
- `_inconclusive(pc)` (new) builds the DECISION entry: `{"inconclusive": True, "reason":
  ..., "pilot_comparison": ...}` where the reason names the measured `clipped_frac`, the
  bound it exceeded, the delivered vs nominal sigma, and states in words that the gate is
  neither a pass nor a no-go, authorizes no training spend, and forces the ladder and the
  clause to be revisited.
- `run_gate`: when the pilot is out of region, EVERY tier's decision becomes that entry.
  `None` is left alone to mean exactly what it meant before — no swept arm cleared the
  threshold.
- `decision_status(entry)` (new, public): the three readings a DECISION entry can carry —
  `no_arm_cleared` / `inconclusive` / `cleared`. The report renders all three distinctly and
  closes with a paragraph saying they must stay distinct.

**Test.** `test_pilot_outside_the_region_of_validity_is_inconclusive`: (a) a 3x-rms pilot
clips past `clipped_frac_max`, and at EVERY tier the decision is not None, carries
`inconclusive=True`, and its reason names both `clipped_frac` and the bound; (b) an in-region
pilot with an unreachable threshold still yields the plain `None`; (c) the three statuses are
three distinct strings, so a reader (or a downstream table) cannot conflate a no-go with an
inconclusive.

**Pre-fix (`audit/fixlog/e4_pre.txt`).**
```
>       assert res["pilot_comparison"]["in_region"] is False
E       KeyError: 'in_region'
1 failed, 14 deselected in 2.35s
```

**Post-fix (`audit/fixlog/e4_post.txt`).** `1 passed, 14 deselected in 3.63s`

**Full suite after the commit: 251 passed.**

---

## ITEM 5 (E5) — `null` stopped meaning two incompatible things

**The narrow YAML edit (the batch's one authorized contract change).** Exactly two lines of
`heston_benchmark_v6.yaml`, both in `acceptance_thresholds.verdict_vocabulary.
outcome_values`:

```diff
-      mechanism_adjudication: [channel_i, channel_ii, decomposition, "null"]   # `null` here is the ADJUDICATED no-channel reading, which coincides with the universal not-evaluated string; disambiguate via notes
-      goldilocks_bates: [decision_relevant_regime_located, "null"]             # same coincidence; `null` = no decisive severity row located
+      mechanism_adjudication: [channel_i, channel_ii, decomposition, no_channel]   # `no_channel` is the ADJUDICATED "neither channel is present" reading; renamed from "null" (fix batch 3 ITEM 5) so it no longer collides with the universal not-evaluated string
+      goldilocks_bates: [decision_relevant_regime_located, no_decisive_regime]      # `no_decisive_regime` = severity rows present, none decisive; renamed from "null" for the same reason. NO rows at all is the universal `null` (not evaluated)
```

`git diff main..HEAD -- '*.yaml'` is exactly these two lines: **2 insertions, 2 deletions,
one file**. The `universal` pair, `must_not_collapse`, `yaml_note` and every other threshold's
outcome set are untouched. Each line's trailing comment moved WITH its string, because the old
comments documented the collision this item removes — a comment describing a collision that no
longer exists would be a false statement inside the frozen pre-registration. If that reading
of "only those strings" is too liberal, the comment halves are the only revertible part.

**Code diff.**
- `_mechanism_reading`: the fall-through reading `"null"` -> `"no_channel"`. This is also the
  value A2's wrong-direction gaps land on (a significantly WORSE arm, or one degrading faster
  with TC), which is exactly what `no_channel` means there.
- `goldilocks_bates`: rows present but none decisive -> `"no_decisive_regime"`. **No rows at
  all still returns `"null"`** — that case is NOT an adjudication, it is not-evaluated, and
  the notes now say so ("NOT EVALUATED: no bates severity cells found").
- `mechanism_memo`'s legend line: "decomposition = both; no_channel = neither (an ADJUDICATED
  reading, distinct from the universal `null` = not evaluated)".
- Consumers checked: `analyze_results` (the two emitters, `_verdict`, `run_analysis`'s
  `_guard`/absent-artifact rows — all remaining `"null"`s are not-evaluated cases and stay),
  `mechanism_memo` (renders the string verbatim; legend updated), `exhibits.py` (names
  `threshold_verdicts.csv` in its module docstring only — no exhibit branches on a verdict
  string, so nothing to change), and every test asserting on the old strings.

**Tests.** New: `test_adjudicated_no_channel_is_not_the_not_evaluated_null` and
`test_goldilocks_no_decisive_regime_is_not_the_not_evaluated_null` (each builds a genuine
adjudication AND the not-evaluated row of the SAME threshold, and asserts the two strings
differ and that the adjudicated one is not `"null"`), plus
`test_contract_thresholds.py::test_adjudicated_verdicts_are_the_contracts_vocabulary`, which
binds the emitted strings to the contract's `outcome_values` sets in both directions (every
declared value is reachable from `_mechanism_reading`, and `"null"` is in neither set).

**One deviation from the item as written.** The item asks that the not-evaluated mechanism
row also not be `"null"`. It is `"null"`, deliberately: `null` IS the universal not-evaluated
value the amendment declares (`verdict_vocabulary.universal`), and renaming that would undo
AM2-2. What the tests assert is the property the item exists to protect — the adjudicated
result and the not-evaluated row are DIFFERENT strings, and the adjudicated one is no longer
`null`.

**Five pre-existing assertions edited** (unavoidable: the fix changes the string the
assertion names): four `reading == "null"` in `test_mechanism_*` and one
`verdict == "null"` in `test_goldilocks_locates_and_null`. Every one of them keeps its
substantive check; only the expected string moved.

**Pre-fix (`audit/fixlog/e5_pre.txt`).**
```
>       assert vd["verdict"] == "no_decisive_regime" != "null"
E       AssertionError: assert 'null' == 'no_decisive_regime'
E         - no_decisive_regime
E         + null
2 failed, 25 deselected in 0.32s
```

**Post-fix (`audit/fixlog/e5_post.txt`).** `2 passed, 25 deselected in 0.32s`

**Full suite after the commit: 254 passed.**

---

## ITEM 2 (E2) — clipped_frac re-measured at production scale, before any pilot exists

**What was run.** `audit/repro/e2_g2_production_scale.py` (new): the CONTRACT ladder
(decision 0.05/0.10/0.15/0.20 + diagnostic 0.40), field mode, confirmatory cell,
`n_paths = 10000` (the engine's production per-cell count), the 10 confirmatory seeds
42-51 — **one seed per gate run**, so the seed-to-seed spread of `clipped_frac` (explicitly
unknown when AM2-3c set the bound) is measured rather than pooled away. 5241s wall clock.
Output: `audit/fixlog/g2_production_scale.txt`, with the 256-path/2-seed column beside it.
No pilot fit and no gate go/no-go run exists, so no value here could be chosen knowing
which side of the bound `sigma_gamma_pilot` lands on.

| sigma_rel | role | production mean | seed std | min | max | 256-path/2-seed |
|---|---|---|---|---|---|---|
| 0.05 | decision | 0.0027 | 0.0002 | 0.0024 | 0.0029 | not measured |
| 0.10 | decision | 0.0737 | 0.0012 | 0.0712 | 0.0756 | 0.0614 |
| 0.15 | decision | 0.1484 | 0.0021 | 0.1449 | 0.1516 | not measured |
| 0.20 | decision | **0.2470** | 0.0030 | 0.2429 | **0.2513** | 0.2153 |
| 0.40 | DIAGNOSTIC | 0.8137 | 0.0031 | 0.8102 | 0.8194 | 0.7800 |

**### STOPPING POINT — this is the human's decision, not a code fix.**

Production scale moves `clipped_frac` UP at every rung by a consistent ~11-20% relative.
At the top DECISION rung the seed MEAN (0.2470) stays inside `clipped_frac_max = 0.25`,
but **3 of the 10 seeds are outside it** (0.2504, 0.2505, 0.2513). The measured seed std is
0.0030 — small — so this is not noise around a comfortable value: the rung sits ON the
bound. AM2-3c chose 0.25 as "the smallest round bound admitting the whole DECISION ladder as
measured (0.215 at its top rung)"; that 16% of headroom is 1.2% at production scale.

Nothing was adjusted: the bound, the ladder and `_DELTA_CLIP` are exactly as the contract
declares them. The options (drop 0.20 to diagnostic, raise `clipped_frac_max`, or accept a
straddling top rung and say so in the gate report) are all contract decisions and are
recorded here for the human to make BEFORE the gate runs.

**Also worth noting:** the two rungs AM2-3a could only BOUND (0.05 and 0.15, from the
monotonicity argument) are now measured, and both are comfortably inside — 0.0027 and
0.1484 against bounds of "<= 0.061" and "<= 0.215". The monotonicity argument held.

**Suite unchanged by this item (no `.py` under test was touched): 254 passed.**
