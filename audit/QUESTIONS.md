# QUESTIONS — things that look wrong but may be deliberate

Each entry: what I saw, the innocent reading, and the one question I need answered.
None of these are filed as findings, because I can construct a reading under which the
code is right.

---

## Q1 — The realized rebalance `dt` cannot equal the contract's declared `dt`

**What I saw.** `Hedging_backtest.py:941` computes `n_steps = int(round(T_prime * 252)) = 43`
over `[0, 0.17]`, giving `dt = 0.003953488` (252.94 rebalances/year). The contract declares
`rebalancing: {frequency: daily, dt: 0.003968}` and `T_prime: 0.17`.

**Innocent reading.** `0.17 * 252 = 42.84` is not an integer, so no grid can be both exactly
daily and exactly end at T′ = 0.17. Rounding to 43 steps spanning exactly the contract
horizon is the better of the two available choices (the alternative, 42 exact-daily steps,
would end at 0.1667 and silently change the horizon — the quantity the A&T construction is
built around). `fixed_across_arms` holds either way, so no comparison is affected.

**Question.** Do you want the paper to report `dt = 0.003953` (what ran), or should
`T_prime` become `42/252 = 0.1667` so that "daily, dt = 1/252" is literally true? This is a
contract edit, so it is your call, not mine. Filed as finding **H1** at P3 because the code
and the contract genuinely disagree, but the disagreement is forced by the contract itself.

---

## Q2 — `dose_response` adds a `p < 0.05` requirement the contract does not state

**What I saw.** `analyze_results.py:450, 527-529`:
```python
                  spearman_p_max: float = 0.05):
...
    monotone = math.isfinite(rho) and rho > 0.0 and (
        not math.isfinite(p) or p < spearman_p_max)
    verdict = "monotone" if monotone else "flat"
```
The contract says only `dose_response: "monotone (isotonic + rank correlation); flat =
regularization null"`. The `p` here is a seed-bootstrap tail probability
`P(rho <= 0)` (line 440), not a classical p-value.

**Innocent reading.** "Rank correlation" with no significance test would make the verdict
turn on the sign of a statistic computed from as few as 5-7 points, which is not defensible;
some threshold is needed to operationalize it, and 0.05 is the conventional one. The
implementation is also the *conservative* direction — it can only turn `monotone` into
`flat`, i.e. into the pre-registered null.

**Question.** Is `p < 0.05` part of the pre-registration as you intend to write it up? If
yes it should be stated in the memo before results (the contract's `STATE BEFORE RESULTS`
discipline); if no, the verdict should key on `rho > 0` plus the isotonic fit alone. Either
way the choice needs to be on paper before the dose-response runs.

---

## Q3 — λ_pde for the *baseline* arm is selected by optimizing the *treatment* arm

**What I saw.** `train.py:135` fixes the selection base arm:
```python
    base = load_arm(args.pinn_cfg, "rung3_delta_gamma_vega")
```
and `fit_and_val_score` scores each (λ_pde, λ_gamma, λ_vega) combo by
`_val_greek_score` of a **rung3** fit. The single selected `lambda_pde` is then applied to
every arm whose PDE term is live — including `standard_pinn`, via `_apply_lambdas:52-53`:
```python
    if "lambda_pde" in lam and cfg.use_pde and cfg.lambda_pde != 0.0:
        upd["lambda_pde"] = float(lam["lambda_pde"])
```
`standard_pinn` is `{}` in `pinn_config.yaml`, i.e. `use_pde=True, lambda_pde=1.0`, so it is
overridden. For `standard_pinn` the residual is the *only* structural signal (its loss is
price + λ_pde·pde), so its optimal λ_pde has no reason to match rung3's.

**Innocent reading — and I think it is the intended one.** The contract's whole design is
"one model class; every arm is a PINNConfig; identical architecture, identical ansatz". A
*shared* λ_pde keeps the residual weight identical across arms so that the only thing
varying between rung1/rung2/rung3/standard_pinn is the label supervision. Tuning λ_pde
per-arm would introduce a second varying factor and break exactly the control the study is
built on. Under that reading, sharing is correct and the only question is where the shared
value comes from.

**Question.** Given that sharing is right, is rung3 the right arm to source the shared value
from? It is the arm that most benefits from the choice, and the confirmatory contrast is
rung3-vs-standard_pinn. Two alternatives that preserve the control: (a) select λ_pde on
`standard_pinn`'s own validation score (sourcing it from the arm for which the residual is
load-bearing), or (b) run the selection on both and report the confirmatory contrast at
both values as a robustness row. I did not file this as a finding because the "hold λ_pde
fixed across arms" reading is coherent and defensible — but the sourcing choice should be
a stated, pre-registered decision rather than an artifact of which arm `train.py` happens to
default to.

---

## Q4 — `rel_improvement` divides by a signed CVaR

**What I saw.** `Hedging_backtest.py:556-557` (and the identical expression in
`analyze_results._pooled_stratified:236-237`):
```python
            "rel_improvement": (float((cvar_b - cvar_a) / cvar_b)
                                if cvar_b != 0.0 else float("nan"))
```
The denominator is the baseline's CVaR of the loss, taken **signed**. If `cvar_b` were
negative (the worst 5% of outcomes are still profits), a genuine improvement
(`cvar_a < cvar_b`) would produce a *negative* `rel_improvement`, and
`confirmatory_cell`'s `rel >= 0.10` test would read a real improvement as a failure. If
`cvar_b` is small-and-positive the ratio explodes.

**Innocent reading.** For a short call hedged discretely under misspecified dynamics with
transaction costs, `cvar95(loss)` is positive with near-certainty — I measured
`cvar95 = +4.41` at tc=0 and `+12.31` at tc=2% on a toy provider in
`audit/repro/r03_engine_grid_and_crn.py`. The guard `if cvar_b != 0.0` handles the only
truly degenerate case. Using `abs(cvar_b)` would be defensive coding against a state the
contract excludes.

**Question.** Do you want an `abs()` in the denominator (or an assertion that `cvar_b > 0`)
as a cheap tripwire, given that a negative baseline CVaR would silently invert the headline
verdict rather than crash? I did not file it because the audit excludes "inputs the contract
excludes", and a profitable 5% tail is arguably one.

---

## Q5 — The `LockedTestSet` guard around λ selection cannot fire

**What I saw.** `train.py:138`:
```python
    guard = LockedTestSet(args.anchor_grids or "<<held-out anchor grids>>")
```
`--anchor-grids` defaults to `None`, so in the default invocation the guard wraps a
placeholder *string*. `select_lambdas` then checks `test_set.locked` before and after every
combo (`train_pinn.py:524-533`) — but nothing in the selection path can ever call
`.unlock()`, so the check is a tautology. The contract asks for
`lambda_discipline: {assert_test_untouched: true}`.

**Innocent reading.** There is **no actual leakage**: I traced `fit_and_val_score` and it
touches only `ArmDataset(..., "train")` and `ArmDataset(..., "val")`, and scores with
`_val_greek_score` on the val split. The guard is a structural placeholder for a future
version where the anchor grids are genuinely passed in, and a tautological guard is harmless.

**Question.** Should production λ-selection runs be required to pass `--anchor-grids <dir>`
so the guard wraps the real artifact? As written, the assertion the contract asks for is
satisfied by construction rather than by evidence — which is fine until someone adds a code
path that reads the anchors during selection, at which point the guard still will not fire
(it only trips on `.data` access, and a new path would more likely `np.load` the file
directly).

---

## Q6 — Anchor-grid `params` is stored as an unnamed vector

**What I saw.** `make_datasets.py:443` writes `"params": np.array([getattr(p, k) for k in
HESTON_PARAM_NAMES])` — a bare 5-vector with no names — and `eval_greeks.py:118` reads it
back with `zip(HESTON_PARAM_NAMES, ...)`. The label artifact, by contrast, stores an explicit
`param_names` key (`make_labels.py:187`) and `build_arm_labels` reads it back by name.

**Innocent reading.** Both sides import the *same tuple object* from `train_pinn`
(verified at runtime: `same object: True`), so they cannot diverge while that import holds.
This is genuinely safe today.

**Question.** Worth adding `param_names` to the anchor-grid npz for symmetry with the label
artifact? The failure mode it would guard against — someone reordering
`HESTON_PARAM_NAMES`, or a consumer defining its own tuple — is precisely CLAUDE.md's
calibration example #2, and the grids are frozen artifacts that will outlive the import.
Not filed as a finding because nothing is wrong right now.

---

## Q7 — `_spearman_seed_bootstrap` uses a bare seed rather than the stream convention

**What I saw.** `analyze_results.py:429`: `rng = np.random.default_rng(int(seed))`, where
every other RNG in the module and the engine uses
`np.random.default_rng([seed, _STREAM_*])` (`_STREAM_POOLED = 7` is defined ten lines above).

**Innocent reading.** `default_rng(42)` and `default_rng([42, 7])` are different streams, so
there is no collision with the pooled bootstrap, and the two bootstraps resample different
things (seeds vs paths) in different functions. Numerically inert.

**Question.** Is this an oversight worth normalizing to `[seed, _STREAM_SPEARMAN]` for
consistency with the documented convention, or deliberate? CLAUDE.md maintains an explicit
list of the *frozen* bare-arithmetic seed derivations (`make_datasets`, `make_labels`) and
this one is not on it, which suggests oversight rather than intent. Zero numerical
consequence either way.
