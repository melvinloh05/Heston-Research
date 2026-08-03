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
