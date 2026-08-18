# FINDINGS — v1 pipeline correctness audit

Baseline: `python -m pytest -q` → **189 passed, 19 warnings, 45.48s** (2026-07-28).
Repo state at audit time: **no `data/` and no `results/` directory exist** — nothing has
been run, nothing frozen. Every finding below is therefore pre-compute: fixing them costs
nothing but a code edit.

Read-only audit: no source file was modified. All reproductions live in `audit/repro/`
with their real captured output in `*_output.txt`.

---

## Tier 1 — `analyze_results.py`

### A1 · P1 · CONFIRMED — Every pre-registered threshold is a Python literal; the module never opens the contract

**`analyze_results.py:53-56`**
```python
DEFAULT_LEVEL = 0.95
DEFAULT_BOOT = 2000
DEFAULT_SEED = 42                                        # contract meta.global_seed
_STREAM_POOLED = 7                                       # pooled-bootstrap rng stream
```
**`analyze_results.py:301-304`**
```python
def confirmatory_cell(pnl_dir, *, tc: float = 0.01, level: float = DEFAULT_LEVEL,
                      n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED,
                      arm: str = "rung3", baseline: str = "standard_pinn",
                      rel_threshold: float = 0.10) -> dict:
```
**`analyze_results.py:554-556`**
```python
def ood_greek_thresholds(greek_agg_csv, *, regimes=("near_feller", "strong_neg_corr"),
                         arm: str = "rung3", red_threshold: float = 0.15,
                         parity_tol: float = 0.10, binding: bool = True) -> dict:
```
**`analyze_results.py:718-721`**
```python
def mechanism_adjudication(pnl_dir, per_seed_csv, *, tcs=(0.0, 0.01, 0.02),
```

**Mechanism.** This module is the *only* place the pre-registered verdicts are computed, and
it contains no `yaml` import, no `resolve_config` call, and no reference to
`heston_benchmark_v6.yaml` or `hedging_config.yaml` (verified by source inspection in R01).
Eleven contract quantities are re-typed as Python defaults. `heston_benchmark_v6.yaml` is
declared READ-ONLY and authoritative in CLAUDE.md; the moment a human is *permitted* to edit
it (the one legitimate edit path), every verdict here keeps using the stale number, silently.

**Blast radius.** All of them currently agree — I diffed them numerically (R01):

| quantity | analyze_results.py | contract/engine | match |
|---|---|---|---|
| cvar level | 0.95 | 0.95 | OK |
| bootstrap B | 2000 | 2000 | OK |
| global seed | 42 | 42 | OK |
| confirmatory rel threshold | 0.1 | 0.1 | OK |
| confirmatory tc tier | 0.01 | 0.01 | OK |
| ood gamma/vega reduction | 0.15 | 0.15 | OK |
| price parity tol | 0.1 | 0.1 | OK |
| tc sweep tiers | (0.0, 0.01, 0.02) | (0.0, 0.01, 0.02) | OK |
| ood regimes | ('near_feller','strong_neg_corr') | ('near_feller','strong_neg_corr') | OK |
| baseline arm name | standard_pinn | standard_pinn | OK |

So this is a **drift hazard, not a live mismatch** — P1 per the audit's own rule, not P0.
The number that would move on drift is *any* verdict string in `threshold_verdicts.csv`.

Two of these defaults have **no contract counterpart at all** and were invented in Python:
- `sakuma_null_consistency(..., rel_tol=0.02)` — the contract says
  `in_model_hedging: NOT_PASS_FAIL`, so a 2% band is a code-local operationalization.
- `dose_response(..., spearman_p_max=0.05)` — the contract says only
  `"monotone (isotonic + rank correlation); flat = regularization null"`; requiring
  `p < 0.05` is *stricter* than pre-registered and can turn a monotone curve into a
  `flat` (= "regularization null") verdict. See Q2.

**Why the tests missed it.** `test_analyze_results.py` constructs synthetic PnL and asserts
verdicts flip at the boundary (e.g. `test_confirmatory_pass_and_fail`), but it passes the
thresholds it is testing *as literals of its own* or relies on the same defaults. No test
opens `heston_benchmark_v6.yaml` and asserts equality against these constants. Grep confirms
`heston_benchmark` appears nowhere in `analyze_results.py`.

**Reproduction.** `audit/repro/r01_analyze_constants_vs_contract.py` → `r01_output.txt`.

**Confidence.** CONFIRMED (executed; all values printed side by side).

**Disconfirmer.** Show me a call site that passes these thresholds in from `resolve_config`
— e.g. a `run_analysis` wrapper that reads the YAML and overrides every default. I found
none: `run_analysis` (line 959) passes only `level`, `n_boot`, `seed` through, and `main`
(line 1057) sources those from argparse defaults that are themselves the module constants.

**Fix (one line).** Have `run_analysis` call `hb.resolve_config(...)` once and thread the
contract's `acceptance_thresholds` / `cvar_convention` / `transaction_costs.tiers` into
every verdict function instead of the module-level defaults.

---

### A2 · P1 · CONFIRMED — `_mechanism_reading` is sign-blind: a gap in the *wrong* direction reads as the robustness channel

**`analyze_results.py:662-666`**
```python
    present_i = _excludes_zero(g0["ci_lo"], g0["ci_hi"])
    widening = (_excludes_zero(gmax["ci_lo"], gmax["ci_hi"])
                and abs(gmax["diff"]) > abs(g0["diff"]) + 1e-12)
    t_ex_reduced = _excludes_zero(t_ex["ci_lo"], t_ex["ci_hi"]) and t_ex["diff"] < 0.0
    present_ii = bool(widening and t_ex_reduced)
```
with **`analyze_results.py:124-128`**
```python
def _excludes_zero(lo: float, hi: float) -> bool:
    """True iff the CI [lo, hi] lies strictly on one side of 0 (both finite)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo > 0.0 or hi < 0.0
```

**Mechanism.** `gaps` are documented at line 647 as *"rung3 - standard (loss units, `<0` =
rung3 better)"*. `present_i` asks only whether the tc=0 CI excludes zero — it accepts a CI
lying entirely **above** zero, which means rung3 is significantly **worse** at 0% TC.
`widening` compounds this with `abs()`: a gap that grows in magnitude *in the wrong
direction* counts as widening. Note the contrast with the sibling verdict functions, which
*do* check direction — `confirmatory_cell:312` and `order_attribution:335` both use
`_excludes_zero(...) and p["ci_hi"] < 0.0`. The direction guard was simply not carried into
`_mechanism_reading`.

**Blast radius.** The `verdict` column of the `mechanism_adjudication` row in
`threshold_verdicts.csv`, and the `**Pre-registered reading: ...**` line plus the entire
"Three candidate stories" narrative of `mechanism_adjudication_memo.md` (written from
`reading` at lines 908-912 and 933-938). Direction of error: a study where second-order
supervision *hurts* delta-only hedging at zero cost — a perfectly publishable pre-registered
null — is reported as `channel_i`, i.e. **"robustness channel present"**, the affirmative
mechanism claim. E2 (the mechanism figure) is built on this reading.

**Why the tests missed it.** `test_analyze_results.py` contains
`test_mechanism_reading_channel_i`, `..._channel_ii`, `..._null` and `..._decomposition`.
Every one of them feeds gaps with **negative** `diff` (rung3 better) — the direction the
study hopes for. No test feeds a positive gap, so the missing sign guard is never exercised.

**Reproduction.** `audit/repro/r02_confirmatory_seed_count_and_direction.py` →
`r02_output.txt`, part (b). Real output:
```
gaps (rung3 - standard, loss units; POSITIVE = rung3 worse):
   tc=0.0   diff=+0.420  CI=[+0.200, +0.650]
   tc=0.01  diff=+0.300  CI=[+0.100, +0.500]
   tc=0.02  diff=+0.250  CI=[+0.050, +0.450]
T_ex diff CI covers 0 (turnover unmoved).

_mechanism_reading -> {'reading': 'channel_i', 'present_i': True, 'present_ii': False, ...}

the MIRRORED case (rung3 genuinely better at every tier):
_mechanism_reading -> {'reading': 'channel_i', 'present_i': True, 'present_ii': False, ...}
```
and for the `widening` half (`r02b_output.txt`):
```
rung3 WORSE and worsening with TC, but trading less ->
  {'reading': 'channel_ii', 'present_i': False, 'present_ii': True,
   'widening': True, 't_ex_reduced': True}
```
i.e. an arm that is *worse at every TC tier and degrades faster as costs rise* is labelled
**cost channel (ii)**.

**Confidence.** CONFIRMED (executed; identical reading for mirrored inputs).

**Disconfirmer.** If the memo or a downstream consumer independently gates on the sign
before printing the reading, the label never reaches a reader unqualified. It does not:
`mechanism_memo` at line 908 prints `reading.get('reading')` verbatim with no sign test, and
`run_analysis:1023` copies the reading straight into the verdict CSV.

**Fix (one line).** In `_mechanism_reading`, replace `_excludes_zero(g0[...])` with
`g0["ci_hi"] < 0.0` and gate `widening` on `gmax["diff"] < g0["diff"]` (more negative), so
both channels require the gap to be an improvement.

---

### A3 · P1 · CONFIRMED — The confirmatory verdict does not check that the pre-registered 10 seeds are present

**`analyze_results.py:308-323`**
```python
    res = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                             slug_filter=_MISSPEC_FILTER)
    p = res["pooled"]
    rel = p["rel_improvement"]
    ok_ci = _excludes_zero(p["ci_lo"], p["ci_hi"]) and p["ci_hi"] < 0.0
    ok_rel = math.isfinite(rel) and rel >= rel_threshold
    verdict = "pass" if (ok_ci and ok_rel) else "fail"
```
and **`analyze_results.py:197-208`** (what "present" means):
```python
    for path in sorted(glob.glob(os.path.join(str(cell_dir), "*.npz"))):
        parsed = _parse_slug(path)
        if not _slug_matches(parsed, slug_filter):
            continue
```

**Mechanism.** `_gather_cell_arrays` globs whatever npz files happen to be on disk. The
contract pre-registers `seeds_confirmatory_cell: 10` and `run_hedging.run_confirmatory`
enforces it at *run* time — but the analysis layer re-derives the seed set from the
filesystem and never compares it to the contract. `run_hedging._run_program` is explicitly
**resumable** (`_ledger_done`, "a completed (cell, seed) is skipped on resume"), so a
partially-completed confirmatory directory is a normal, expected on-disk state, not an
exotic one. Pointing `--confirmatory-dir` at it yields a fully-formed `pass`.

**Blast radius.** `verdict` and `statistic` of the `confirmatory_cell` row — the headline
number of the paper — computed on an under-powered seed set. The pooled-stratified CI
*narrows* with fewer seed blocks only through the path count, so the CI can still exclude 0
on 3 seeds; the seed-to-seed component (the one the contract's
`tail_claim_requires: paired_bootstrap_over_CRN_paths_with_seed_variance_separated` cares
about) is simply estimated from 3 points. `n_seeds` is reported **only inside the free-text
`notes` string** — no structured column carries it.

**Why the tests missed it.** `test_run_hedging.py::test_confirmatory_uses_exactly_10_seeds`
covers the *runner*, not the analysis. On the analysis side,
`test_analyze_results.py::_confirmatory_dir` does write `for s in range(10)` — so
`test_confirmatory_pass_and_fail` happens to run at the contract's seed count — but it
asserts only `verdict == "pass"` / `"fail"`; nothing ties the verdict to the seed count.
And the sibling `test_order_attribution_pass_and_null` builds its null case with
`for s in range(6)` and asserts a normal `fail` verdict, so a non-10 seed set is already an
*accepted* input in the test suite. No test asserts that a short seed set is refused or
flagged.

**Reproduction.** `audit/repro/r02_confirmatory_seed_count_and_direction.py` →
`r02_output.txt`, part (a). Real output:
```
seeds present = 3  [42, 43, 44]
  verdict   : pass
  statistic : 0.2938  (pooled rel improvement)
  CI        : [-0.6, -0.6]
  row keys  : ['cell', 'ci_hi', 'ci_lo', 'notes', 'statistic', 'threshold_id', 'verdict']

seeds present = 10  [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
  verdict   : pass
  statistic : 0.2886  (pooled rel improvement)
```

**Confidence.** CONFIRMED (executed).

**Disconfirmer.** If `run_analysis` refused to start against a directory whose ledger shows
incomplete cells, the operator could not reach this state. It does not read `run_ledger.csv`
at all — `run_analysis:967-972` builds paths from `confirmatory_dir` and never inspects the
ledger.

**Fix (one line).** In `confirmatory_cell`, require
`res["n_seeds"] == contract meta.seeds_confirmatory_cell` and emit `verdict="null"` (with
the shortfall in `notes`) otherwise; add `n_seeds` to `VERDICT_COLS`.

---

### A4 · P2 · CONFIRMED (static) — `_guard` converts *any* analysis exception into a `null` verdict

**`analyze_results.py:978-983`**
```python
    def _guard(fn, tid, cell):
        try:
            return fn()
        except Exception as exc:                        # missing artifact -> null row
            return _verdict(tid, cell, float("nan"), None, "null",
                            f"not evaluated: {type(exc).__name__}: {exc}")
```
(plus the same pattern at 488, 998, 1031)

**Mechanism.** The comment scopes the intent to "missing artifact", but the clause catches
everything: a `KeyError` from a renamed CSV column, a `ValueError` from a shape mismatch in
`_pooled_stratified`, a `scipy` import failure inside `_spearman_seed_bootstrap`. All become
the string `null`, which in this codebase means *"not evaluated"* — a legitimate,
non-alarming outcome (`ood_greek_thresholds:598-599` uses `null` for exactly that).

**Blast radius.** The `verdict` column of any of the seven threshold rows. A genuine
analysis bug is indistinguishable, at a glance, from an intentionally-absent artifact. The
exception type *is* preserved in `notes`, which is what keeps this at P2 rather than P1 —
the evidence survives, but only for a reader who reads every `notes` cell.

**Why the tests missed it.** `test_analyze_results.py` exercises the missing-artifact path
(`test_run_analysis_missing_artifacts_degrade_to_null`) and asserts `verdict == "null"` —
i.e. the test *codifies* the broad catch as intended behaviour. There is no test in which a
present-but-corrupt artifact must raise.

**Confidence.** CONFIRMED (static reading; the code path is unambiguous). Not separately
reproduced — it is a one-line control-flow reading, and I am not claiming a number moves.

**Disconfirmer.** If `null` were rendered distinctly from a real "not applicable" in the
memo, a reader could not confuse them. `mechanism_memo:874` prints every verdict identically
as `**{verdict}**`.

**Fix (one line).** Narrow to `except (FileNotFoundError, KeyError) as exc` for the
artifact-absent case and let everything else propagate — or emit `verdict="error"` (a value
distinct from `null`) when the exception is not an absence.

---

### A5 · P3 · CONFIRMED (static) — `dose_response` fits the curve on the common seed set but reports `cvar_mean` over each arm's own seeds

**`analyze_results.py:483-493`**
```python
        cvars = {int(r["seed"]): _num(r.get("cvar")) for r in cell
                 if _num(r.get("cvar")) is not None and r.get("seed") != ""}
        y_mean, y_std, n = _finite_mean_std(list(cvars.values()))
        ...
        rows.append({"arm": arm, ..., "cvar_mean": y_mean, ...})
```
**`analyze_results.py:503-512`**
```python
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []

    def _arm_y_mean(i):
        if common:
            return float(np.mean([y_by_seed[i][s] for s in common]))
        return float(np.mean(list(y_by_seed[i].values())))

    if len(fit_arms) >= 2:
        ys = [_arm_y_mean(i) for i in range(len(fit_arms))]
        iso = isotonic_increasing(xs, ys)
```

**Mechanism.** Restricting the fit to `common` seeds is the *right* call (it keeps the
isotonic/Spearman comparison paired). But `cvar_mean` — the column written to
`dose_response.csv` and plotted as the data points of E3 — is the mean over each arm's own
seed set. When the seed sets differ (one dose arm's run failed on one seed), the plotted
point and the fitted line for that arm come from different seed subsets. `bs_gap` at line
537 has the same asymmetry.

**Blast radius.** `cvar_mean` vs `isotonic_fit` columns of `dose_response.csv`, hence the
E3 "money figure". Identical whenever all dose arms have all seeds, which is the intended
state. Direction of error: unsigned.

**Why the tests missed it.** `test_analyze_results.py::test_dose_response_monotone` builds
every dose arm with the same seed list, so `common == each arm's own seeds` and the two
means coincide exactly.

**Confidence.** CONFIRMED (static; the two code paths are plainly different aggregations).

**Disconfirmer.** If `run_full_sweep`'s ledger guaranteed all-or-nothing per seed across
arms, the sets could never diverge. It cannot: the ledger unit is `(cell, seed)`, and all
arms for a cell are run in one engine call — so in practice divergence needs a partial
resume, same precondition as A3.

**Fix (one line).** Report `cvar_mean` over `common` too (or add a `n_seeds_common` column
and compute both).

---

### `analyze_results.py` — checked and CLEAN

- **Bootstrap resampling unit.** `_pooled_stratified:213-239` resamples path indices *within
  each seed block* and applies the same indices to both arms, then re-pools — the correct
  stratified paired scheme. Per-seed CIs are reported *separately* (`per_seed`,
  `per_seed_summary`). This matches the contract's
  `tail_claim_requires: paired_bootstrap_over_CRN_paths_with_seed_variance_separated`
  exactly. Not a finding.
- **Pooling convention.** Both readings are emitted, and the pooled one is documented as the
  confirmatory statistic (docstring lines 12-20). No hidden mean-of-ratios.
- **Cell selection.** `_MISSPEC_FILTER`/`_INMODEL_FILTER` omit a `regime` field, which is
  *correct*: `Hedging_backtest._iter_sim_cells:768` builds every DGP from
  `bm["regimes"][mis["train_params"]]` (= `baseline`) only, so there is no regime axis in
  the hedging sweep to confuse. Verified by reading `_iter_sim_cells` and `_cell_slug`.
- **Arm naming.** `"rung3"`, `"rung2"`, `"rung1"`, `"standard_pinn"` are the *engine*
  method names, and `pinn_provider._ARM_DIR:43-49` maps them to checkpoint dirs. Consistent
  on both sides; `_measured_label_error:365` uses the same map. Not a finding.
- **`ood_greek_thresholds` binding-regime tracking** (lines 588-592) correctly keeps the
  *minimum* gamma reduction across regimes and reports its CI.
- **Confirmatory vs exploratory separation.** `run_analysis` reads the confirmatory verdicts
  from `conf_pnl` and the dose/goldilocks rows from `full_dir`; `run_hedging` writes them to
  separate directories (`out_dir/confirmatory`, `out_dir/full_sweep`). No path by which an
  exploratory cell reaches the headline verdict.

---

## Tier 1 — `Hedging_backtest.py`

### H1 · P3 · CONFIRMED — The realized rebalance `dt` is 0.3657% off the contract's declared `dt`

**`Hedging_backtest.py:941`**
```python
    n_steps = int(round(T_prime * eng["rebalancing"]["frequency_per_year"]))
```
against **`heston_benchmark_v6.yaml:209`**
```yaml
  rebalancing: {frequency: daily, dt: 0.003968, fixed_across_arms: true}
```
and **`:208`** `horizon: {T_prime: 0.17, ...}`.

**Mechanism.** `0.17 * 252 = 42.84`, which is not an integer, so `round` gives 43 steps over
`[0, 0.17]` and `np.linspace` (line 177) makes every step `0.17/43 = 0.003953488`. The
contract declares `dt: 0.003968` (= 1/252). The realized rebalancing frequency is
**252.94/year**, not 252.

**Blast radius.** Every absolute PnL/CVaR level (43 rebalances instead of 42.84 worth,
each 0.37% shorter). It is applied identically to every arm and every cell, so **no
arm-vs-arm comparison moves** — hence P3. What moves is the number you would write in the
paper's methods section: "daily rebalancing, dt = 0.003968" is not what ran.

**Why the tests missed it.** `test_hedging_backtest.py` asserts the engine exposes
`horizon.T_prime == 0.17` and `frequency_per_year == 252` (per `docs/CONFIG_AUDIT.md` §6),
and separately that positions/PnL shapes follow `n_steps`. No test computes
`T_prime / n_steps` and compares it to `hedging_simulation.rebalancing.dt`.

**Reproduction.** `audit/repro/r03_engine_grid_and_crn.py` → `r03_output.txt`:
```
  contract hedging_simulation.rebalancing.dt = 0.003968
  T_prime * freq                             = 42.84   (not an integer)
  n_steps = round(...)                       = 43
  realized dt = T_prime / n_steps            = 0.003953488
  relative difference                        = 0.3657%
  => realized rebalances per year            = 252.9412
```

**Confidence.** CONFIRMED (executed).

**Disconfirmer.** Nothing here is fixable by code alone — the contract's own `T_prime=0.17`
and `dt=1/252` are mutually inconsistent, so *some* rounding is forced. This is flagged so
the reported `dt` matches what ran, not to suggest the engine is wrong. See Q1.

**Fix (one line).** Either report `dt = 0.003953` in the paper, or (a contract edit, your
call) set `T_prime = 42/252 = 0.166667` so the grid is exactly daily.

---

### `Hedging_backtest.py` — checked and CLEAN (this is the strongest module in the repo)

I went through the whole file line by line against the audit's hunt list. Every item on that
list checks out:

- **Premium convention, including the oracle's own row.** `_eval_methods:636-641` computes
  `oracle_cache = delta_positions(..., providers[oracle_name], ...)` with *no*
  `premium_override`, then sets `prem_override = oracle_cache[1]`; line 662-664 gives every
  other arm that same override, and the oracle reuses `oracle_cache`. So all arms —
  oracle included — settle against one identical per-path premium vector. **No free lunch**,
  and the smoothed variant inherits the same `premium` (line 667-669). The absent-oracle
  fallback warns loudly and stamps `premium_convention_ok=False` on every row (648-654, 702).
- **CRN.** Paths are simulated *once per cell*, outside the arm loop (`_run_sweep:991`), and
  every arm indexes the same `(times, S, v)` arrays. No arm consumes RNG at all. Verified
  bitwise in R03: same seed re-simulated → `S True v True`.
- **QE scheme.** `simulate_heston_qe:179-213` is textbook Andersen (2008): `c1`/`c2`, the
  `psi <= psi_c` quadratic branch (`b2 = 2/psi - 1 + sqrt(2/psi)sqrt(2/psi-1)`,
  `v = m/(1+b²)(b+Z)²`), the exponential branch with its atom at zero
  (`np.where(ue <= pp, 0.0, ...)`), and the central log-spot discretisation
  `K0..K4` with `gamma1 = gamma2 = 0.5`. All correct. **The martingale correction is
  deliberately absent and documented** (line 173-174) as a bias common to all hedgers — see
  Q3. Crucially, `z_v`, `u` and `z_s` are drawn **unconditionally** (lines 198-199, 211,
  comment *"always drawn: stream alignment"*), so the branch mix cannot desynchronise the
  stream across cells.
- **`lambda_j = 0` recovers Heston bit-for-bit** — asserted at both write time
  (`bank_write:878-882`) and run time (`_run_sweep:992-996`), and independently confirmed in
  R03 (`S True v True`).
- **The `v = 0` atom.** The `GreekProvider` docstring (line 96-97) makes flooring the
  provider's responsibility, and both providers honour it at the same floor:
  `HestonCFProvider` `v_floor=1e-6` (providers.py:58, 72) and `PINNProvider`
  `v_floor=1e-6` (pinn_provider.py:78, 114). Symmetric across arms.
- **Transaction costs.** `_settle_core:383-406` charges `tc * price * |Δposition|` at
  inception (`|pos_0|`), at every rebalance (`|pos_i - pos_{i-1}|`) and at the final unwind
  (`|pos_last|`, gated by `charge_final_unwind`). Proportional on traded *notional*, on the
  **absolute** change. Matches `transaction_costs: {kind: proportional_on_traded_notional}`.
- **PnL decomposition.** `tcf` is accrued forward at `r` alongside `cash`, so
  `pnl_total = pnl_directional - tc_paid_fv` holds *exactly*; it is re-derived by an
  independent frictionless settlement and asserted every time (`settle_delta:448-450`).
  Verified numerically in R03 at all three tiers (`decomposition ok=True`).
- **`dt` bookkeeping / off-by-one.** `dt = np.diff(times)` has exactly `n_steps` entries;
  the loop consumes `dt[0..n_steps-2]` and the final block `dt[-1]` — each used once, no
  double-count, no omission. The position held over the last interval is `pos[:, -1]`,
  liquidated at `px[:, n_steps]`.
- **Discounting.** One convention throughout: everything is a *future value at T'*
  (`cash * exp(r·dt)` forward, liability marked at T'). Never mixed with a present value.
- **Terminal mark.** `_liability:945-963` **raises** rather than silently settling at
  expiry when a `T'` horizon is declared (two separate guards, lines 950-953 and 960-962),
  exactly as CLAUDE.md's live-convention invariant requires.
- **`T_ex`.** Computed against `ctx.oracle_name = eng.get("oracle_provider_name", "oracle")`
  — read from `hedging_config.yaml`, **not hardcoded** (line 970). `_total_traded:419-426`
  is the sum-including-endpoints definition CLAUDE.md declares. Positions are built once,
  outside the tc loop (`_eval_methods:656`, comment *"positions once per method/variant
  (never per tc)"*), so `analyze_results._t_ex_diff`'s tc-invariance premise is sound —
  verified in R03 (`t_ex source _total_traded equal: True`).
- **No-trade band.** `smooth_positions:335-353` is a pure symmetric band
  (`|raw - held| > band`), with `ema_alpha` accepted and explicitly `del`-ed — matching
  CLAUDE.md's "PURE no-trade band (no EMA)". Selected on validation only; see the
  `run_hedging` section.
- **Aggregation.** `add_gap_closed:1053-1088` groups on `(tag fields, tc, seed)` and
  `aggregate_over_seeds:1091-1157` groups on `(tag fields, tc, method)` — correct units.
  `gap_closed` is emitted **both** ways (`gap_closed_mean` = ratio-of-mean-gaps,
  `gap_closed_mean_of_ratios` = legacy), with the denominator floored on the CVaR scale
  rather than by an absolute epsilon (`_gap_denominator_floor:1045-1050`). No silent
  NaN→0; undefined cells become `""`.
- **Contract-target lock.** `_assert_contract_targets:140-163` asserts that
  `magnitude = 1.0` along every engine direction lands within 1e-12 of the contract's
  perturbation targets. Confirmed in R03: `combined m=1 -> xi=0.44999999999999996,
  rho=-0.8`.
- **Path banks.** `_BankLoader.load:814-840` hard-checks sha256, DGP kind, DGP params and
  array shape, and raises on any mismatch — *"the engine NEVER silently resimulates"*. True
  as written.

---

## Tier 1 — `gate_headroom.py`

### G1 · P1 · CONFIRMED — The gate's decision threshold is a hardcoded `0.10`, not read from the contract

**`gate_headroom.py:355-362`**
```python
    # smallest sigma_rel whose mean spread clears the pre-registered 10%
    # relative threshold WITH every seed's paired CI excluding 0
    decision = {}
    for tc in tiers:
        decision[tc] = next(
            (s for s in summary if s["tc"] == tc
             and s["spread_rel_mean"] >= 0.10
             and s["ci_excludes_zero_frac"] == 1.0), None)
```

**Mechanism.** The audit asks this question directly, so: the threshold is **hardcoded**.
`run_gate` reads `cfg["benchmark"]` for the instrument, tiers, seeds and regimes, but the
one number that decides the gate is a Python literal. The contract's
`oracle_headroom_gate.decision_rule` ("spread < pre-registered 10% CVaR95 threshold") and
`acceptance_thresholds.confirmatory_cell_pass` (">=10% relative") are the source; neither is
read. Same root cause as **A1**, different file — listed separately because it needs its own
edit and because the gate runs *before all training*, so a drift here mis-sizes the entire
compute commitment.

**Blast radius.** The `decision` dict and the "## DECISION (per tc tier)" section of
`headroom_report.md` — which is the input to the human go/no-go that gates every GPU dollar
in the project. Currently numerically correct (0.10 == the contract's 10%).

**Why the tests missed it.** `test_gate_headroom.py` exercises `run_gate` on a smoke-size
cell and asserts the returned dict has the expected shape and that the decision is `None`
when no arm clears — it never compares `0.10` to the contract.

**Confidence.** CONFIRMED (verbatim quote; also covered by R01's finding that no gate/analysis
module opens the contract for thresholds).

**Disconfirmer.** A `bm["acceptance_thresholds"]` lookup anywhere in `run_gate`. There is
none — `bm` is dereferenced only for `hedging_simulation`, `regimes`, `grid` and `meta`.

**Fix (one line).** Read the threshold from
`cfg["benchmark"]["acceptance_thresholds"]` (or a new `oracle_headroom_gate` numeric key)
and pass it into `run_gate` as a parameter.

---

### G2 · P2 · CONFIRMED (static) — The delta clip silently makes the delivered noise smaller than the calibrated noise

**`gate_headroom.py:81`** and **`:241-248`**
```python
_DELTA_CLIP = (-0.05, 1.05)
```
```python
    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float,
                 K: float) -> dict:
        out = dict(self.base.evaluate(S, v, tau, K))
        delta = np.asarray(out["delta"], float)
        err = (self.eta(S, v, tau) if self.mode == "field"
               else self._rng.normal(0.0, self.sigma_delta, delta.shape))
        out["delta"] = np.clip(delta + err, *_DELTA_CLIP)
        return out
```
against the calibration at **`:206-209`**
```python
        self.amp = 1.0                          # unit amp for calibration pass
        unit_std = float(np.std(self.eta_dS(*ref_states)))
        self.amp = (self.sigma_gamma_target / unit_std
                    if self.sigma_gamma_target != 0.0 else 0.0)
```

**Mechanism.** `amp` is calibrated so that `std(dη/dS)` over the reference cloud equals the
requested `sigma_gamma_target` — measured on the **unclipped** field. The clip is then
applied to `delta + η` at evaluation time. Wherever the clip binds (an ATM call's delta
already sits near 0.5, but deep ITM/OTM path states sit near 1 and 0, exactly where the
`[-0.05, 1.05]` bounds are), the realized delta error — and therefore the realized *gamma*
error, which is what the gate is calibrating — is **smaller** than `sigma_gamma_target`.
Nothing measures or reports the clipped fraction.

**Blast radius.** `spread_rel_mean` per (sigma_rel, tc) in `headroom.csv`, and hence which
`sigma_rel` row the DECISION section flags. Direction of error: the delivered corruption is
**weaker** than labelled, so the measured spread is **understated** and the gate is
**conservative** — it will demand a larger nominal `sigma_gamma` than truly required. That
is the safe direction for a go/no-go, but it means the σ axis of the reported gate table is
not the axis it is labelled with, and the pilot-calibrated point (`--sigma-gamma`) may be
judged against a corruption it did not actually receive.

**Why the tests missed it.** `test_gate_headroom.py` asserts the field is frozen
(`evaluate` is a pure function of state, same input → same output) and that `amp` scales the
calibration statistic. It never checks how much of the *delivered* delta error survives the
clip, and its smoke sizes use small σ where the clip rarely binds.

**Confidence.** CONFIRMED (static; the calibration and the clip are plainly on opposite
sides of the noise application, and no code measures the gap). I did **not** run the
full-size gate to quantify the clipped fraction — that is a GPU-scale run and needs your
sign-off; see SUMMARY "unable to verify".

**Disconfirmer.** If the reference cloud's delta distribution is concentrated well inside
`[-0.05, 1.05]` at every swept σ, the clip never binds and this is inert. Measuring
`mean(clip binds)` over the reference cloud at each `sigma_rel` settles it in one cheap run;
I could not, because it needs `HestonCFProvider` over a full reference cloud (~minutes) and
the answer is regime-dependent.

**Fix (one line).** Record `float(np.mean((delta + err < lo) | (delta + err > hi)))` per arm
into the summary/report, so a binding clip is visible rather than silent.

---

### `gate_headroom.py` — checked and CLEAN

- **CRN between the clean and corrupted legs.** `run_gate:309-315` builds ONE providers dict
  (`{oracle: base, noisy_*: ...}`) and makes ONE `hb.run_headline` call, so every leg is
  evaluated on the same simulated paths within a cell, and the `pnl_vs_oracle_*` paired
  bootstrap resamples both arms with the same indices (`_eval_methods:703-711`). The spread
  is genuinely paired — **not** contaminated by independent MC noise.
- **`sigma_gamma_pilot` units.** `--sigma-gamma` is documented ABSOLUTE, and line 304 uses it
  directly as `sigma_gamma_target`, which the calibration matches against
  `std(dη/dS)` — a quantity in gamma units. **Units are consistent.** (The *handoff* from
  `train.py` is a separate concern — see T1.)
- **Comparison direction/strictness.** `"ci_excludes_zero": ci_lo > 0.0` (line 337) is
  one-sided *in the correct direction*: it requires the corrupted arm to be significantly
  **worse** than the oracle. The name understates what it does; the semantics are right.
  `>= 0.10` and `== 1.0` implement the stricter "every seed" reading that CLAUDE.md declares
  as deviation #2.
- **Anisotropic field + amplitude-matched iid** are the two declared deviations in CLAUDE.md
  and the code matches the declaration exactly (`_BANDWIDTH = (1.0, 0.1, 0.1)` line 80;
  `self.sigma_delta = float(np.std(self.eta(*ref_states)))` line 211, no `sqrt(dt)`).
  Not a finding — declared design.
- **One frozen field per gate.** `gate_seed = int(bm["meta"]["global_seed"])` (line 301) is
  shared by every σ arm, so the ladder is a common-field ladder — the right choice for
  monotonicity across σ.

---

## Tier 2 — `run_hedging.py`

### R1 · P2 · CONFIRMED (static) — `resolved_config.yaml` records the LAST CELL's trimmed config, not the program's

**`run_hedging.py:194-196`**
```python
        cell_cfg = _cell_cfg(prog, tag, seed)
        t0 = time.perf_counter()
        rows = hb.run_headline(cell_cfg, providers_cache[seed], run_root)
```
**`run_hedging.py:124-136`** (what the trim does)
```python
    c = copy.deepcopy(cfg)
    c["derived"]["seeds"] = [int(seed)]
    ...
        mis["perturbations"] = {tag["direction"]: src_perts[tag["direction"]]}
        mis["cross_model"] = []
        c["engine"]["misspecification"]["magnitudes"] = [tag["magnitude"]]
```
**`Hedging_backtest.py:1004-1005`** (what the engine then writes)
```python
    if out_dir:
        log_resolved_config(cfg, out_dir)
```

**Mechanism.** `_run_program` calls the engine once per (cell, seed) with a **trimmed**
config, and the engine unconditionally writes `resolved_config.yaml` into the shared
`run_root` on every call. The file is overwritten each time, so what survives is the last
cell's trim: one perturbation direction, one magnitude, one seed. `bank_write` stamps a
`config_hash` from the *full* resolved config, and `_resolved_config_hash`'s docstring
already anticipates the mismatch ("a trimmed sub-sweep … legitimately hashes differently"),
so the trimming is intended — but the *provenance artifact* still claims to describe the run.

**Blast radius.** No reported statistic moves. What moves is the run's provenance record: a
reader (or a future reproduction attempt) reconstructing the confirmatory run from
`confirmatory/resolved_config.yaml` would read `seeds: [51]` and `magnitudes: [1.0]` and
conclude the confirmatory cell was a single-seed, single-magnitude run. P2 rather than P3
because reproducibility of a pre-registered study is load-bearing.

**Same-shaped, smaller sibling.** `headline_delta_only_per_seed.csv` and `_agg.csv` are also
overwritten per cell by the engine (`Hedging_backtest.py:1008-1012`) and only restored at the
end of `_run_program` from `_rows_master.csv` (`run_hedging.py:205-212`). A hard interruption
therefore leaves a *parseable but single-cell* per-seed CSV on disk. A resumed run repairs
it; a crashed one does not, and `analyze_results` reads that exact filename
(`PER_SEED_CSV`) for `_t_ex_diff` and `dose_response`.

**Why the tests missed it.** `test_run_hedging.py` asserts the ledger resumes and that the
final CSVs contain every cell — it checks the *end state* of a completed run, which is
correct. No test inspects `resolved_config.yaml`'s contents, and none interrupts a run.

**Confidence.** CONFIRMED (static; the overwrite is unconditional and the trim is explicit).

**Disconfirmer.** If `_run_program` re-wrote the untrimmed `prog` config after the loop, the
last-cell version would be replaced. It does not — lines 205-212 rewrite only the two metric
CSVs.

**Fix (one line).** Call `hb.log_resolved_config(prog, run_root)` once in `_run_program`
after the cell loop, so the program-level config is the surviving copy.

---

### `run_hedging.py` — checked and CLEAN

- **Band selection is validation-only, and provably so.** `select_band_width:302` pins
  `magnitudes = [0.5]`, and lines 327-328 *assert* every consumed row is magnitude 0.5
  ("falsifier guard: the band is tuned ONLY on the mid-severity cell"). The band's target
  (`standard_pinn_smoothed`) is not one of the confirmatory contrast arms, so no selection
  decision touches the reported cell.
- **Per-seed providers.** Both `_run_program:190-193` and `select_band_width:313-332` rebuild
  `build_providers` per seed and hold one seed's arms at a time; the docstring calls out the
  failure mode they are avoiding ("Reusing a single providers dict across seeds would hedge
  every seed with seed 0's network, faking the seed diversity"). Correct.
- **Seed derivation / cell collisions.** Cells are enumerated by
  `hb._iter_sim_cells(prog)` — the *same* generator the engine and `bank_write` use — and
  keyed by `hb._cell_slug(tag, seed)`, which embeds every tag field plus the seed. Two
  different cells cannot collide on a slug, and the trimmed per-cell call reproduces the
  untrimmed program's slug exactly (that is what makes resume align).
- **Within a cell, all arms get identical paths**: guaranteed upstream by the engine (one
  simulation per cell, all providers indexed against it).
- **Confirmatory vs full sweep are structurally separate directories**
  (`out_dir/confirmatory` vs `out_dir/full_sweep`), each with its own ledger, row master, PnL
  dir and CSVs. An exploratory cell cannot reach the confirmatory verdict.
- **Missing checkpoints raise loudly** (`pinn_provider.build_providers:173-179`) and list
  every `best.pt` that does exist — no silent substitution.

---

## Tier 2 — `run_info_matching.py`

### I1 · P2 · CONFIRMED — A plateau caused by the training-row cap is indistinguishable from an information plateau

**`run_info_matching.py:71-85`**
```python
def subsample_train(train_ds: ArmDataset, n_rows: int, seed: int) -> ArmDataset:
    ...
    n_keep = min(int(n_rows), train_ds.n_rows)
    perm = np.random.default_rng([int(seed), 40961]).permutation(train_ds.n_rows)
    idx = torch.as_tensor(perm[:n_keep], dtype=torch.long)
```
**`run_info_matching.py:452-456`**
```python
    for m, cfg in zip(mults, sweep_cfgs):
        for seed in seeds:
            row, best_state = _one(cfg, seed, m * N)
```
**`run_info_matching.py:121-146`** (the rule that reads the resulting curve)
```python
        impr = (prev - cur) / prev if prev > 0 else float("nan")
        rel.append(impr)
        if plateau_i is None and (not math.isfinite(impr) or impr < tol):
            plateau_i = i
```

**Mechanism.** The budget ladder requests `m * N` rows (N = 4096, m up to 5, so 20480 at the
cap). `subsample_train` silently caps at `train_ds.n_rows`. If the frozen train split holds
fewer rows than `5N`, the top rungs are trained on **bit-identical data**, the Gamma-RMSE
curve is exactly flat *by construction*, and `plateau_multiplier` returns
`plateau_reached=True` — the same value a genuine information plateau produces. Nothing
asserts, warns, or records that the cap bound; `saturation_sweep_configs` asserts only that
`max(m) <= cap_multiplier`, which is about the contract's 5N cap, not about data
availability.

**Blast radius.** `plateau_multiplier` / `plateau_reached` in
`info_matching_agg.csv` and the `_budget_sidecar`'s `selected_multiplier`, and the reported
paragraph built by `build_paragraph` — whose whole claim is the contract's wording
*"plateau bounds what THIS architecture/protocol extracts from prices"*. Under a binding
cap that sentence is false: the plateau bounds what the **label artifact contains**.
Downstream, `run_hedging`'s full sweep loads the plateau-m checkpoint, so the
`info_matched_baseline` arm's budget would be set by the artifact size.

**Why the tests missed it.** `test_run_info_matching.py` unit-tests `plateau_multiplier` on
**synthetic curves** (the docstring says so: "pure; unit-tested on synthetic curves") where
the flatness is stipulated, and drives the sweep end-to-end only with
`base_n_price_points` shrunk so small that the cap never binds. The two conditions are never
combined.

**Reproduction.** `audit/repro/r05_info_matching_cap_plateau.py` → `r05_output.txt`:
```
  frozen train rows = 8199   (N = 4096, cap 5N = 20480)
  rung -> rows actually trained on: {1: 4096, 2: 8192, 3: 8199, 4: 8199, 5: 8199}
  rungs whose data is BIT-IDENTICAL to the rung below: [4, 5]

  mean Gamma rel-RMSE by rung : [0.4, 0.3, 0.24, 0.24, 0.24]
  plateau_multiplier ->  {'plateau_multiplier': 4, 'plateau_index': 3,
                          'plateau_reached': True,
                          'rel_improvement': [nan, 0.25, 0.2, 0.0, 0.0]}

  For contrast, a genuine information plateau at the same rung:
  curve [0.4, 0.3, 0.24, 0.238, 0.237] -> {'plateau_multiplier': 4, ...,
                          'plateau_reached': True, ...}
```

**Confidence.** CONFIRMED (executed; the two verdict dicts differ only in the
`rel_improvement` decimals, and both report `plateau_reached=True` at m=4).

**Disconfirmer.** If the frozen label artifact is guaranteed to hold ≥ 5N training rows, the
cap can never bind. Nothing in the repo guarantees it: `make_labels` sizes the artifact from
`--n-points × n_skt` minus the oracle mask, and `n_price_points` (4096) is set independently
in `pinn_config.yaml`. Show me an assertion tying the two and this is inert.

**Fix (one line).** In `_one`, `assert train_ds.n_rows == min(m * N, ...)` — or, better, raise
(or set `plateau_reached=False` with a loud note) when `subsample_train` returns fewer rows
than requested at any rung below the plateau.

### `run_info_matching.py` — checked and CLEAN

- **Nested ladder.** `subsample_train` takes `perm[:n_keep]` from ONE seed-derived
  permutation, so larger budgets strictly ADD rows. Confirmed in R05: at `n_rows = 5N` the
  rung sizes are exactly `{1: 4096, 2: 8192, 3: 12288, 4: 16384, 5: 20480}` and no rung
  duplicates its predecessor.
- **Gamma scored against `gamma_ref`.** `val_greek_rmse:110` uses
  `val_ds.data["gamma_ref"] if q == "gamma"` — the frozen true consensus, never a noised
  per-arm label. Matches the CLAUDE.md invariant.
- **Capacity control is matched on the axis the contract names.** The contract says "retrain
  saturated baseline at 2x width"; `capacity_control_config(replace(base, n_price_points=pm*N),
  cap_width)` sets `width_mult=2.0` with the SAME seeds and the SAME subsample, and
  `MLP.__init__` turns that into `width 64 -> 128`. Correct axis, correctly held-constant
  elsewhere.
- **Best-validation checkpoints only.** `_one` does `model.load_state_dict(best_state)`
  before scoring — never the last step.

---

## Tier 2 — `train.py` / `train_pinn.py`

### T1 · P2 · CONFIRMED (static) — `--pilot` prints four numbers, two of them the deliberately-reproduced buggy value, and the handoff to the gate is manual

**`train.py:234-247`**
```python
    if args.pilot:
        # BEFORE: reproduces the prior bug exactly (last-step model, arm-label priority) so the
        # magnitude of the correction is visible before rerunning any gate decision on it.
        sigma_pre, rel_pre, src_pre = _pilot_gamma_rmse(model, val_ds, args.device,
                                                         prefer_ref=False)
        model.load_state_dict(best_state)
        sigma, rel, src = _pilot_gamma_rmse(model, val_ds, args.device, prefer_ref=True)
        ...
        print(f"sigma_gamma_pilot BEFORE fix (last-step model, {src_pre}) = "
              f"{sigma_pre:.6g} (relative {rel_pre:.6g})")
        print(f"sigma_gamma_pilot AFTER  fix (best-step model, {src}) = "
              f"{sigma:.6g} (relative {rel:.6g})")
```
consumed by **`gate_headroom.py:468-470`**
```python
    ap.add_argument("--sigma-gamma", type=float, default=None,
                    help="ABSOLUTE pilot-calibrated sigma_gamma; replaces the "
                         "sweep with the single pilot point")
```

**Mechanism.** The correctness fix itself is **present and right**: `sigma_pre` is taken
before `load_state_dict(best_state)` (so it really is the last-step model with the arm's own
label), and `sigma` after (best-validation model, `gamma_ref`) — exactly the fix CLAUDE.md's
calibration example #1 describes. The residual hazard is the *interface*: stdout carries
four numbers on two lines, the first line is the **known-wrong** value, and the gate takes
its input as a hand-typed CLI float. `runlog["sigma_gamma_pilot"]` is written to JSON but
**no code reads it** — the handoff is entirely human.

**Blast radius.** The gate's single pilot point, and therefore the human go/no-go on the
whole project's compute. Direction: `sigma_pre` is computed on an unconverged model, so it
is typically *larger* than `sigma`; feeding it in would make the gate look **more**
favourable than it is.

**Why the tests missed it.** `test_train.py::test_pilot_prints_finite_sigma_gamma` asserts
the printed values are finite and that `runlog` carries `sigma_gamma_pilot` /
`sigma_gamma_pilot_relative` / `sigma_gamma_pilot_source`. It cannot test which number a
human copies, and there is no test asserting the gate is invoked with the runlog value.

**Confidence.** CONFIRMED (static; the print order and the missing machine handoff are
verbatim). The *units* question the audit raises is answered CLEAN: the gate wants the
absolute value and `sigma` is the absolute RMSE, so the correct pairing is
`--sigma-gamma <the AFTER-fix, non-parenthesised number>`.

**Disconfirmer.** A driver script that reads `runlog.json` and shells out to
`gate_headroom.main(["--sigma-gamma", str(runlog["sigma_gamma_pilot"])])`. `grep -rn
sigma_gamma_pilot` finds it written in `train.py` and read nowhere.

**Fix (one line).** Add `--sigma-gamma-from-runlog <runlog.json>` to `gate_headroom`'s CLI
and print the BEFORE line to stderr (or behind a `--show-prefix-bug` flag).

---

### T2 · P2 · CONFIRMED (static) — `last.pt` is labelled `matched_epochs` even when the arm early-stopped

**`train_pinn.py:400`, `:438-446`**
```python
    while step < tcfg.steps and not stopped_early:
        ...
            if step >= tcfg.steps or stopped_early:
                break
        epoch += 1
    ...
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```
**`train_pinn.py:462-467`**
```python
        "checkpoints": {
            "best": {"step": best["step"], "val_total": best["val"], "path": "best.pt"},
            "matched_epochs": {"step": step, "path": "last.pt",
                               "val_total": (val_curve[-1]["val_total"] if val_curve
                                             else best["val"]),
                               "reached_max_steps": step >= tcfg.steps}},
```

**Mechanism.** `train.py` defaults to `early_stop=not args.matched_epochs`, i.e. **early
stopping is on unless the flag is passed**. When an arm early-stops, the loop breaks and
`last_state` is the model at the early-stop step — not at the `steps` budget. It is
nonetheless saved as `last.pt` and filed in the runlog under the key `matched_epochs`.
`pinn_config.yaml:93-95` states the intent plainly: *"last.pt (matched_epochs) is ALSO
recorded for the contract's report_both (matched_epochs, matched_compute)"* — which holds
only under `--matched-epochs`.

**Blast radius.** The contract's `compute_accounting.report_both: [matched_epochs,
matched_compute]` table. If that table is assembled from `last.pt` / `checkpoints.matched_epochs`
across arms produced by default runs, arms that early-stopped at different steps are compared
as though they had a matched budget — precisely the confound the audit's hunt list names.
The `reached_max_steps` flag is the honest signal and it *is* recorded, which caps this at P2.

**Why the tests missed it.** `test_train.py::test_matched_epochs_checkpoint_exists_alongside_best`
asserts both files exist — it does not assert `reached_max_steps` is True, and it does not run
the default (early-stop) path and check the label.

**Confidence.** CONFIRMED (static; control flow is unambiguous).

**Disconfirmer.** If every production run is launched with `--matched-epochs`, the label is
accurate. That is a run-book question, not a code guarantee — and the default is the other way.

**Fix (one line).** Rename the runlog key to `last` and derive `matched_epochs` only when
`reached_max_steps` — or make `train.py` refuse to write `last.pt` under the
`matched_epochs` name when the arm stopped early.

---

### `train.py` / `train_pinn.py` — checked and CLEAN (verified by execution)

Every model-layer invariant the audit asked me to test directly, I tested directly.
Reproduction: `audit/repro/r04_model_invariants.py` → `r04_output.txt`.

- **Loss-scale buffers are bit-identical across arms.** `train_model:377-379` freezes them
  from the **FULL training split** before the first optimizer step (not from a minibatch),
  and `_freeze_loss_scales` takes the gamma scale from `gamma_ref`. Measured across all 15
  arms (ladder rungs, standard_pinn, feedforward, every dose arm, gradient_penalty_only,
  sobolev_sans_pde, lambda_pde_zero, ude):
  ```
    reference (rung3) scales: bc=1, delta=0.348208, gamma=0.000856181, pde=0.012214,
                              price=30.535, vanna=0.00331633, vega=8.34058
    <every arm>              scales == rung3 : True
  ```
  Including the shuffled/bs_gamma/σ=0.5 arms, whose own gamma labels differ by construction.
  The CLAUDE.md `gamma_ref` invariant holds.
- **Price heads are bit-identical across arms at a fixed seed** — including the UDE arm, so
  building the correction net does **not** perturb the price-head init:
  `<every arm> price head == rung3 : True` (15/15).
- **OFF loss terms contribute exactly zero gradient** because they are *absent from the loss
  dict entirely*, not multiplied by zero: `standard_pinn -> ['pde','price']`,
  `rung1_delta -> ['delta','pde','price']`, `sobolev_sans_pde` and `lambda_pde_zero` both
  `-> ['delta','gamma','price','vega']` (no `pde` key, confirming `docs/CONFIG_AUDIT.md` §1's
  claim that the two arms emit an identical loss dict).
- **Compute accounting is correctly differenced.** `_GRAD_CALLS` is a module-level mutable
  global (`SobolevPINN.py:56`), but every reader takes a difference across the run
  (`train_pinn.py:395` / `:453`, `train_arm:61` / `:75`), so cross-arm accumulation in one
  process cannot corrupt the count. Not a finding.
- **λ selection touches train/val only.** `_run_select_lambdas` builds `ArmDataset(..., "train")`
  and `ArmDataset(..., "val")`, and `fit_and_val_score` scores with `_val_greek_score` on the
  val split, using `gamma_ref` for the gamma component (`train.py:119`). No held-out artifact
  is opened. (The `LockedTestSet` guard around it is decorative — see Q5.)
- **Splitting is by PARAMETER POINT, never by (S, K, tau) row** (`ArmDataset:268-271`) — a
  point's rows share its price surface, so a row-wise split would leak. Correct as written.
- **PDE collocation is anchor-excised, and the code refuses to proceed otherwise**:
  `train_model:388-392` raises `"use_pde arm needs pde_anchors — never sample collocation
  points without excising the named eval anchors"`. Matches the CLAUDE.md invariant.
- **Global RNG.** `set_seed` (`train_pinn.py:38-47`) is the only place that touches
  `torch.manual_seed` / `np.random.seed` / `random.seed`, and it is called at the top of
  `train_arm` and `train_model` — controlled entry points. Everything stochastic elsewhere
  uses `np.random.default_rng([seed, _STREAM_*])`. No module-level RNG anywhere in the repo.

---

## Tier 2 — `eval_greeks.py` / `pinn_provider.py`

### `eval_greeks.py` — checked and CLEAN

- **Vega convention matches on BOTH sides.** `oracle.py:28-29` states *"vega everywhere in
  this module = dV/dv0 (sigma-vega = 2*sqrt(v0)*dV/dv0 by the sqrt(v) chain rule)"*, so
  `consensus_vega` is dV/dv0. The prediction side is `autodiff_greeks`' `g[:, i_v]` where
  `i_v` is the `v0` input column and normalization lives inside `MLP.forward` — i.e. raw
  dV/dv0. **Same convention, no missing `2*sqrt(v0)` factor.** This was the highest-risk
  item on the eval hunt list and it is clean.
- **Producer/consumer array layout agrees.** `make_datasets.generate_anchor_grids` builds
  `np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")` and saves `mask_any` / `consensus_*` /
  `uncertainty_*` at that `(nS, nK, nT)` shape; `eval_arm_on_regime` rebuilds the identical
  meshgrid and ravels C-order. Aligned.
- **Regime parameter ordering cannot drift.** The anchor grid stores `params` as a bare
  vector, but both producer and consumer use the *same imported tuple object*:
  `make_datasets.py:68` does `from train_pinn import HESTON_PARAM_NAMES`, and
  `eval_greeks.py:42` does the same. Verified at runtime: `same object: True`,
  `('kappa','theta','xi','rho','v0')`, matching `oracle.HestonParams`' field order. This is
  the exact shape of CLAUDE.md's calibration example #2 and it is correctly guarded.
- **Gamma truth is the consensus, never an arm label** — eval reads `consensus_gamma` from
  the frozen anchor grid; arm-specific corrupted labels exist only in `build_arm_labels` and
  never reach this module.
- **Wing rule implemented as specified.** `eval_arm_on_regime:151-154` computes Gamma's
  `rel_rmse` on body points only and returns NaN on a pure-wing slice; the absolute rmse and
  quantiles include every point. Matches `moneyness_wing_holdout.gamma_metric: absolute_only`.
- **Anchor grids are loaded from the frozen artifact and never touched during training** —
  `run_greek_eval` reads `{regime}_grid.npz` from `anchors_dir`, and no training code path
  opens those files (training samples the hypercube with those regimes *excised*).
- **Checkpoint + dtype.** `build_providers` loads `best.pt`; `PINNProvider` casts to float64
  at load; `eval_arm_on_regime:112-113` takes dtype/device from the loaded model.

### `pinn_provider.py` — checked and CLEAN (verified by execution)

- **Input column assembly cannot transpose.** `_build_x:118-126` builds columns by
  **name lookup** in `cfg.inputs` order and raises `KeyError` on an unknown name — there is
  no positional stacking to get wrong.
- **Chunking is bit-exact.** The audit asked for this to be verified by running both; I ran
  four chunk sizes over 5000 path states:
  ```
    chunk=None  bit-equal to chunk=97: {'price': True, 'delta': True, 'gamma': True, 'vega': True}
    chunk=2048  bit-equal to chunk=97: {...all True}
    chunk=512   bit-equal to chunk=97: {...all True}
          max abs diff: price=0.000e+00, delta=0.000e+00, gamma=0.000e+00, vega=0.000e+00
  ```
  Exactly zero difference at every chunk size, including for Gamma (max |gamma| ≈ 5.9e-05,
  so this is not a scale artifact).
- **`v_floor` / the QE atom at zero.** `evaluate` at `v = 0` returns finite values for all
  five quantities, and `HestonCFProvider` uses the **same** `v_floor = 1e-6`
  (`providers.py:58`) — so the two provider families floor identically and the arms stay
  comparable. Extrapolation below the training range is left un-clamped by design
  (`MLP.forward` is affine with no clamp), which is documented in both modules.
- **The theta_train-frozen convention is correct here and correctly opposite in
  `eval_greeks`** — the two modules each carry an explicit docstring warning about the other
  (CLAUDE.md calibration example #2). Both are right in context.

---

## Tier 3 — `SobolevPINN.py`, `ude.py`, `providers.py`

### CLEAN — verified by execution (`audit/repro/r04_model_invariants.py`)

- **Config flags genuinely disable terms.** `loss()` builds the term dict conditionally, so
  an OFF term is absent — it never enters the autodiff graph, never contributes to `total`,
  and never consumes RNG. The `need` tuple passed to `greeks()` is likewise assembled from
  the flags, so an arm that does not supervise Gamma never runs the double backward.
- **UDE: `g_phi` is exactly zero at init and the residual is bit-identical.**
  ```
    param_counts(ude)  = {'price_head': 13121, 'correction': 321, 'total': 13442}
    param_counts(base) = {'price_head': 13121, 'correction': 0, 'total': 13121}
    max |g_phi(v)| over v in [0, 0.2]  = 0.000e+00
    residual bit-identical (ude vs base) : True
    max abs diff                          : 0.000e+00
  ```
  The price head is 13121 params in **both**, matching `docs/BASELINE_STATUS.md`'s
  single-architecture invariant, and the extra 321 params are isolated in `.correction`.
- **RNG draw order.** `UDESobolevPINN.__init__` calls `super().__init__(cfg)` *first*, so the
  price head consumes its draws before the correction net exists — confirmed by the
  bit-identical price-head state_dict across all 15 arms including `ude`.
- **`_variance_drift` is the only override**, so the UDE change is confined to
  `pde_residual`; `greeks`, `greeks_eval`, `loss` and every scale are inherited unchanged.
- **`providers.HestonCFProvider`** evaluates at the **pathwise** `v0 = max(v, v_floor)` with
  its own (kappa, theta, xi, rho) — structurally symmetric with `PINNProvider`, which pins
  (kappa, theta, xi, rho) at theta_train and uses the pathwise v as `v0`. Same state
  convention, same floor, no asymmetry between the oracle arm and the PINN arms.

---

## Tier 3 — `exhibits.py`

### X1 · P2 · CONFIRMED — Missing values are drawn as a hard `0.0` while the backing CSV correctly records them blank

**`exhibits.py:429-434`** (E2, the mechanism figure's T_ex panel)
```python
        for m in tex_methods:
            row = _by_method(agg, misspec, tex_tc).get(m, {})
            tvals.append(_num(row.get("t_ex_mean")) or 0.0)
            terr.append(_num(row.get("t_ex_seed_std")) or 0.0)
            tcol.append(_method_color(m))
        ax_tex.bar(range(len(tex_methods)), tvals, yerr=terr, capsize=3, color=tcol)
```
**`exhibits.py:611-613`** (E4, fraction-of-gap-closed)
```python
        gvals = [_num(bym.get(m, {}).get("gap_closed_mean")) for m in arms]
        ax_gap.bar(range(len(arms)), [v if v is not None else 0.0 for v in gvals],
                   color=[_method_color(m) for m in arms])
```
**`exhibits.py:623-624`** (E4, cost/directional split) and **`:661-663`** (E4, vanna inset)
```python
        costs = [_num(bym.get(m, {}).get("tc_component_mean")) or 0.0 for m in arms]
        dirs = [_num(bym.get(m, {}).get("directional_component_mean")) or 0.0 for m in arms]
```
```python
            vvals.append(_num(gr.get("reduction_vs_standard_pinn_mean")) or 0.0)
            verr.append(_num(gr.get("reduction_vs_standard_pinn_std")) or 0.0)
```

**Mechanism.** `_num` returns `None` for a blank or non-finite cell (`exhibits.py:139-147`).
`... or 0.0` then substitutes a real, meaningful data value. For **T_ex** this is acute:
`T_ex = 0` means *"this arm trades exactly like the oracle"*, and the contract's mechanism
falsifier is stated in those terms — *"cost channel requires T_ex → 0 for clean arm; unmoved
T_ex kills it"*. A cell with no T_ex is drawn at the exact value that would be read as the
cost channel's confirming evidence. For **gap_closed**, 0.0 reads as "closed none of the
oracle gap" — an affirmative negative result. The error bars degrade the same way, so a
missing seed-std renders as a bar with *zero uncertainty*.

That this is an oversight rather than a convention is visible **inside the same function**:
the 2×2 inset thirty lines later maps missing → `float("nan")` and prints `"n/a"`
(`exhibits.py:444-455`).

**Blast radius.** The rendered E2 (mechanism figure) and E4 (decomposition panel) PNGs —
two of the four contract exhibits. The backing CSVs are **correct** (`_fmt` maps `None` → `""`),
so figure and CSV disagree.

**Why the tests missed it.** `test_exhibits.py`'s fixture `_HEDGE_METHODS` populates
`t_ex`, `gap_closed`, `tc_comp` and `dir_comp` for **every** method, so no exhibit test ever
renders a missing cell. The tests hash the backing CSV (`"we hash the backing CSV, never the
PNG"`), which is exactly the artifact that is right — so even a golden-value test would pass.

**Reproduction.** `audit/repro/r06_exhibits_missing_to_zero.py` → `r06_output.txt`:
```
--- E2 backing CSV, T_ex panel rows ---
   method=rung3                    value='0.2'      err='0.01'
   method=rung1                    value=''         err=''
   method=standard_pinn            value='0.3'      err='0.01'

--- what the FIGURE draws for the same rows ---
   method=rung3                    csv cell=0.2        -> bar height 0.2
   method=rung1                    csv cell=''         -> bar height 0.0
   method=standard_pinn            csv cell=0.3        -> bar height 0.3
```

**Confidence.** CONFIRMED (executed end-to-end through `exhibit_e2`).

**Disconfirmer.** If every agg CSV is guaranteed to carry `t_ex_mean` for every plotted arm,
this never fires. It is not guaranteed: `Hedging_backtest._eval_methods:686` writes
`t_ex = ""` for **every** delta-gamma row, and `:671-672` writes `float("nan")` for every arm
when the oracle provider is absent — both of which `aggregate_over_seeds` propagates as `""`.

**Fix (one line).** Replace `or 0.0` with `if ... is not None else float("nan")` at the five
sites (matplotlib omits NaN bars rather than drawing them at zero), matching the 2×2 inset's
existing handling.

### `exhibits.py` — otherwise checked and CLEAN

- **No recomputation of frozen statistics.** Every panel reads pre-computed `*_mean` /
  `*_seed_std` columns from the agg CSVs. The single exception is E2's 2×2 inset, which
  falls back to `cvar_mean(focal) - cvar_mean(baseline)` when
  `pnl_vs_baseline_cvar_diff_mean` is absent (`exhibits.py:383-388`) — and that fallback is
  explicitly commented (`# fall back to mean diff`) and is the correct quantity, just a
  seed-mean-of-diffs rather than the paired statistic.
- **Missing required inputs raise with the exact expected path** (`_require:132-137`),
  rather than degrading silently.
- **Deterministic backing CSVs**: `_fmt` pins float formatting to `".10g"` and `_write_csv`
  pins the line terminator, so regeneration is bit-stable (which the tests check by hashing).

---

## Cross-cutting

### C1 · P1 · CONFIRMED — 19 contract quantities are re-typed as Python literals across 5 modules; 0 have drifted

This is the consolidated root cause behind **A1**, **G1** and the `eval_greeks` /
`exhibits` threshold literals. Reported once here, with the full inventory, rather than as
five findings.

**Reproduction.** `audit/repro/r08_contract_constants_sweep.py` → `r08_output.txt`.

Which modules read a config file at all:
```
module                    reads contract  reads engine yaml  yaml import
Hedging_backtest.py                 True               True         True
run_hedging.py                      True               True         True
run_info_matching.py                True               True         True
eval_greeks.py                      True              False         True
train.py                            True               True         True
gate_headroom.py                   False              False        False
analyze_results.py                 False               True        False
exhibits.py                        False              False        False
```
(`gate_headroom` receives the resolved contract *dict* via `hb.resolve_config()` in `main`
and dereferences `bm[...]` for the instrument, tiers, seeds and regimes — it simply does not
use it for the threshold. `analyze_results` imports `Hedging_backtest` but never calls
`resolve_config`, so it has no contract in hand at all.)

The inventory, diffed numerically:
```
site                       quantity                               code           contract  status
analyze_results.py:53      cvar level                             0.95               0.95  duplicate (equal)
analyze_results.py:54      bootstrap B                            2000               2000  duplicate (equal)
analyze_results.py:55      global seed                              42                 42  duplicate (equal)
analyze_results.py:50      confirmatory magnitude                  1.0                1.0  duplicate (equal)
analyze_results.py:304     confirm rel thresh                      0.1                0.1  duplicate (equal)
analyze_results.py:301     confirmatory tc                        0.01               0.01  duplicate (equal)
analyze_results.py:556     ood gamma reduction                    0.15               0.15  duplicate (equal)
analyze_results.py:556     ood vega reduction                     0.15               0.15  duplicate (equal)
analyze_results.py:556     price parity tol                        0.1                0.1  duplicate (equal)
analyze_results.py:718     tc tiers                  (0.0, 0.01, 0.02)  (0.0, 0.01, 0.02)  duplicate (equal)
gate_headroom.py:361       gate spread thresh                      0.1                0.1  duplicate (equal)
eval_greeks.py:288         ood gamma reduction                    0.15               0.15  duplicate (equal)
eval_greeks.py:288         ood vega reduction                     0.15               0.15  duplicate (equal)
eval_greeks.py:289         price parity tol                        0.1                0.1  duplicate (equal)
exhibits.py                vanna threshold line                   0.15               0.15  duplicate (equal)
train_pinn.py:188          info-match cap                            5                  5  duplicate (equal)
train_pinn.py:341          feller_min fallback                     0.4                0.4  duplicate (equal)
train_pinn.py:342          excision radius fallback                0.1                0.1  duplicate (equal)
train_pinn.py:288          80/20 split fallback                    0.2                0.2  duplicate (equal)

  duplicated-but-equal : 19
  MISMATCHED           : 0
  no contract value    : 5
```

**Blast radius.** Zero today — **nothing has drifted**. The audit's rule ("duplicates are P1
even when currently equal") is why this is P1 rather than P3: the contract is the one file a
human is *expected* to edit, and 19 shadow copies is 19 chances for a silent divergence
between what the pre-registration says and what the verdict computes.

Five further literals have **no contract counterpart at all** — they are code-local
operationalizations of prose clauses, listed here so they can be written into the memo
before results rather than discovered afterwards:

| site | quantity | value | contract says |
|---|---|---|---|
| `run_info_matching.py:62` | `PLATEAU_TOL` | 0.02 | "grow_until_greek_accuracy_plateaus" (no number) |
| `analyze_results.py:622` | sakuma `rel_tol` | 0.02 | `in_model_hedging: NOT_PASS_FAIL` |
| `analyze_results.py:450` | `spearman_p_max` | 0.05 | "monotone (isotonic + rank correlation)" |
| `eval_greeks.py:45` | wing bounds | (0.75, 1.30) | `moneyness_wing_holdout` (no numeric bounds) |
| `Hedging_backtest.py:49` | QE γ₁, γ₂ | (0.5, 0.5) | not a contract quantity (Andersen central scheme) |

**Fix (one line).** One `contract_thresholds(cfg) -> dict` helper in `Hedging_backtest`,
threaded into `analyze_results.run_analysis`, `gate_headroom.run_gate` and
`eval_greeks.run_greek_eval`; plus the parity test in `audit/test_gaps.md` item 2.

---

### Global state — CLEAN

Full grep across every non-test module:

- **No module-level `np.random` state anywhere.** Every stochastic call site uses
  `np.random.default_rng(...)` locally.
- **`torch.manual_seed` / `np.random.seed` / `random.seed` appear in exactly one place**,
  `train_pinn.set_seed` (lines 38-47), called at the top of `train_arm` and `train_model` —
  controlled entry points. It also pins `torch.use_deterministic_algorithms(True,
  warn_only=True)` and cuDNN determinism.
- **No `torch.set_default_dtype` anywhere.** Dtype is carried per-model
  (`SobolevPINN.__init__` does `self.to(getattr(torch, cfg.dtype))`) and per-provider
  (`PINNProvider` casts to `eval_dtype`), never globally.
- **No `warnings.filterwarnings` and no `np.errstate`.** The only `warnings` use is
  `catch_warnings()` + `simplefilter("ignore")` scoped to two `spearmanr` calls
  (`analyze_results.py:423-425, 431-436`), each with a comment explaining the specific
  nan case being suppressed, and each followed by an explicit `np.isfinite` filter.
- **No mutable default arguments** in any signature I read.
- **`_GRAD_CALLS`** (`SobolevPINN.py:56`) is the only module-level mutable state; every
  reader differences it across a run, so accumulation is harmless (see the `train_pinn`
  CLEAN section).
- **Broad `except Exception`** appears 4× and all 4 are in `analyze_results.py` — see **A4**.
  Zero bare `except:` in the repo.

---

### Determinism — CLEAN

Two Tier-1 entry points, run twice in one process and twice in fresh processes.
Reproduction: `audit/repro/r07_determinism.py` → `r07_output.txt`.

```
=== process 1 ===
engine   rows: 24   in-process repeat identical: True
analysis      : pooled diff=-0.4  in-process repeat identical: True
ENGINE_DIGEST=a96c33519d10e1374f5ac7f5d55587ef4529bc0455ede378166db15ca48159e3
ANALYSIS_DIGEST=2d3a31aff3f932cd08e492f0be63709fbf182124dee40320c46c5e4704caded7
=== process 2 (fresh process) ===
engine   rows: 24   in-process repeat identical: True
analysis      : pooled diff=-0.4  in-process repeat identical: True
ENGINE_DIGEST=a96c33519d10e1374f5ac7f5d55587ef4529bc0455ede378166db15ca48159e3
ANALYSIS_DIGEST=2d3a31aff3f932cd08e492f0be63709fbf182124dee40320c46c5e4704caded7
=== diff ===
(no differences) IDENTICAL ACROSS PROCESSES
```
Bit-equal on the full row set (sha256 over every emitted metric) and on the pooled
bootstrap statistic and per-seed CIs. No hash-seed, dict-order or float-accumulation
sensitivity.

---

## Light pass — oracle layer interfaces (`oracle.py`, `greek_labels.py`, `make_labels.py`, `make_datasets.py`)

Per the audit's instruction, interfaces only; no quadrature re-derived.

**All CLEAN.** The producer/consumer contracts hold on every boundary I checked:

- **`make_labels.build_arm_labels` → `ArmDataset` → `SobolevPINN.loss`.** The batch emits
  `gamma_ref` for **every** arm including `label_source: none`
  (`make_labels.py:348-353`), which is what makes `_freeze_loss_scales` able to take the
  gamma scale from the true consensus. Verified numerically: all 15 arms carry identical
  `loss_scale_*` buffers (R04c). The retained-row set is `mask_any`-filtered *identically*
  across arms (`build_arm_labels:329`, `ArmDataset:268`), and `build_arm_labels` does **not**
  subsample by `n_price_points` — so arms differ only in the `gamma` column, exactly as the
  contract's `corruption: gamma_labels_only` requires.
- **`make_datasets.generate_anchor_grids` → `eval_greeks`.** Keys, shapes and C-order ravel
  all agree; `HESTON_PARAM_NAMES` is a single shared tuple object. Detailed in the
  `eval_greeks` CLEAN section.
- **`oracle` vega convention → `eval_greeks` / `pinn_provider`.** `dV/dv0` on both the label
  and the prediction side. No missing `2*sqrt(v0)`.
- **`make_labels._sha256_file` / `_assert_not_frozen` / `_git_rev` → `Hedging_backtest.bank_write`**
  (line 852) — the path-bank writer reuses the label layer's frozen-path refusal, so a path
  bank cannot be written straight into `data/frozen`.
- **`greek_labels.make_gamma_labels`.** Returns `None` for `source="none"`, and
  `PINNConfig.__post_init__` asserts `supervise_gamma ⇒ label_source != "none"` and
  `gamma_label_noise_sigma > 0 ⇒ label_source == "oracle"` — the two ways to get a
  meaningless dose arm are both blocked at construction. Its bare `default_rng(seed)`
  means the σ ladder shares one underlying normal draw (σ arms get `sigma * z` for a common
  `z`), which is the CRN-analogue and desirable for a dose-response; different training
  seeds still give different realizations because `ArmDataset` passes its own seed through.
- **One asymmetry worth noting, not a finding.** `analyze_results._measured_label_error`
  computes the dose x-axis at a fixed `label_seed = 42`, while the y-axis (`cvar_mean`) is a
  mean over all seeds' *own* label realizations. Since the x statistic is an RMS over
  thousands of points it is essentially seed-independent (≈ σ by construction), so the
  mixing is numerically inert.

**Not re-derived** (out of scope by instruction): the trap-free CF, the COS/FD leg, the MC
leg, the Craig–Sneyd ADI leg, the 4th-leg band routing, and the mask-neutrality statistics.
