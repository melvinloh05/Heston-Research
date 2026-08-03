# FINDINGS_ADDENDUM — noticed during fix batch 1, NOT fixed

Everything here was seen while implementing `audit/FIXLOG.md` and left alone under the
batch's "no refactors" rule. None of it blocks compute. Same tiering as `FINDINGS.md`
(P1 = drift/robustness hazard, P2 = provenance/legibility, P3 = cosmetic).

---

## N1 · P2 — three `threshold_precheck.csv` column NAMES hard-code the contract's numbers

`eval_greeks.py:182, 302-303` writes the columns `gamma_ge_0.15`, `vega_ge_0.15`,
`price_parity_within_0.10`. Fix batch 1 made the *values* in those columns come from the
contract (C1), but the *names* are still string literals. After a contract edit to, say,
`ood_gamma_rmse_reduction_min: 0.20`, the CSV would carry a correctly-computed boolean under
a header that says `0.15` — a header that lies is worse than a number that drifts, because
nothing tests a header.

**Not fixed because** `test_eval_greeks.py:266-267` reads those exact keys, so renaming them
(to e.g. `gamma_ge_min`) would break a pre-existing test — batch rule 4. A one-line comment
now sits at the write site. Worth doing in a batch that is allowed to touch that test.

---

## N2 · P2 — `_MISSPEC_FILTER` / `_INMODEL_FILTER` remain module constants

`analyze_results.py:50-51`. The confirmatory cell SELECTOR (`direction=combined`,
`magnitude=1.0/0.0`) is still typed in Python rather than threaded from the contract like
every threshold now is. It is asserted equal to
`hedging_simulation.confirmatory_cell.perturbation` by
`test_contract_thresholds.py::test_analyze_results_confirmatory_cell_filter_is_the_contract_cell`,
so a divergence fails loudly — but the value itself is still a duplicate.

**Not fixed because** three pre-existing tests pass `ar._MISSPEC_FILTER` into
`paired_ci_from_npz` directly; replacing the constants with a `_filter(th)` helper would
break them. Same story for `eval_greeks.WING_LO/WING_HI` and `run_info_matching.PLATEAU_TOL`
(both now parity-tested, both still constants).

---

## N3 · P2 — `run_analysis`'s `_guard` swallows the new A3 "null" into the same bucket as a missing artifact

`analyze_results.py:1070-1075` (finding **A4**, not in this batch's scope). A3 now emits
`verdict="null"` for an incomplete seed set, and `_guard` emits `verdict="null"` for a
missing artifact, and `mechanism_memo` renders both as `**null**`. The notes distinguish
them ("NOT EVALUATED: 3 of the pre-registered 10 seeds present" vs "not evaluated:
FileNotFoundError: …"), but a reader scanning the verdict column sees one word for three
different states: not-applicable, not-run, and analysis-crashed.

**Suggested with A4:** emit `verdict="error"` for a non-absence exception, and consider a
distinct `verdict="incomplete"` for the A3 shortfall. Both are one-line changes; they belong
together in the batch that does A4.

---

## N4 · P3 — `n_seeds` is blank for four of the seven verdict rows

Introduced by A3. `confirmatory_cell`, `order_attribution` and `sakuma_null_consistency`
populate the new structured column; `dose_response`, `ood_greek_thresholds`,
`mechanism_adjudication` and `goldilocks_bates` leave it `""` because their seed count is
per-arm / per-cell rather than a single number for the row (dose_response already reports
`n_seeds` per arm in `dose_response.csv`; goldilocks reports it per severity cell). Not
wrong, but the column is only half-populated in `threshold_verdicts.csv`.

---

## N5 · P2 — the info-matching cap check runs AFTER training, not before

Fix I1 detects a row-capped rung from the per-seed rows, i.e. once every rung has been
trained. On a full-size run that is hours of GPU spent before the sweep announces its ladder
was unfillable. The check is cheap and fully determined up front —
`min(m * N) > train_ds.n_rows` is knowable from the artifact before the first optimizer
step.

**Not fixed because** the audit's proposed fix, and the shape the batch asked for, is the
post-hoc plateau invalidation; adding a pre-flight raise changes when a legitimate run dies
and deserves its own decision. Recommended for batch 2: a pre-flight `ValueError` naming the
required row count, so a mis-sized artifact fails in seconds rather than hours.

---

## N6 · P3 — `_spearman_seed_bootstrap` still uses a bare seed (Q7)

`analyze_results.py:429` (now ~line 455): `np.random.default_rng(int(seed))` where the
module's own convention two functions above is `np.random.default_rng([seed, _STREAM_*])`.
Numerically inert (different streams either way, different resampling units) and already
filed as **Q7**; re-noted only because fix batch 1 touched every other line in that
function's neighbourhood and deliberately left this one alone — it is a frozen-stream
question, not a cleanup.

---

## N7 · P3 — `train.py --select-lambdas` now costs `len(cpde)` extra fits

Q3's staged search fits the baseline arm once per `lambda_pde` candidate (4 by contract) and
rung3 once per (gamma, vega) combo (9 by contract) = **13 fits**, against the previous joint
grid's 4x3x3 = **36**. So the change is a ~2.8x *saving*, not a cost — recorded here only
because the compute-accounting table will show a different fit count than the pre-amendment
plan implied, and someone will ask.
