# Correctness audit — v1 pipeline

You are auditing a research codebase for **silent correctness bugs that would poison
published results**. The entire test suite currently passes. That is exactly why this
audit exists: I am looking for the bugs the tests cannot see, and for tests that pass
because they are vacuous.

This is a pre-registered causal study. A wrong number here does not crash — it produces
a plausible verdict that is false, and I defend it in front of reviewers. Treat every
number that reaches `analyze_results.py` as load-bearing.

---

## Hard constraints

1. **Read-only on source.** Do not edit, refactor, reformat, or "fix" any `.py`, `.yaml`,
   or `.md` file in the repo. Not even an obvious one-character bug. I need to see the
   diff-free state and decide myself.
2. **You may write only inside `audit/`.** Create it. Everything you produce goes there.
3. **You may execute code** — run scripts, run pytest, write throwaway reproduction
   scripts under `audit/repro/`, import modules and poke at them. Execution is
   encouraged; it is the difference between a finding and a guess.
4. **No git operations.** No commit, no checkout, no stash, no branch.
5. **No network.**

---

## What counts as a finding

Only bugs that change a number, bias a comparison, or invalidate a claim.

**In scope:**
- Wrong math, wrong sign, wrong convention, wrong units
- Leakage of held-out data into selection or training
- Broken common-random-numbers or seed handling that breaks paired comparisons
- Any asymmetry between arms other than the one factor an arm is supposed to vary
- Metric definitions that don't match `heston_benchmark_v6.yaml`
- Aggregation errors (mean-of-ratios vs ratio-of-means, bootstrap over the wrong unit,
  CI over the wrong axis, pooling that ignores seed structure)
- Silent failure: bare `except`, `.get(k, default)` where a missing key should be fatal,
  NaN→0 coercion, clipping that masks divergence, suppressed warnings
- Shape/broadcasting bugs — especially `(n,)` vs `(n,1)` silently becoming `(n,n)`
- State or order dependence: global RNG, mutable default args, module-level state,
  in-place tensor mutation, `torch.set_default_dtype` side effects
- Loading the wrong artifact (checkpoint, label array, config) downstream

**Explicitly out of scope. Do not report these:**
- Style, naming, formatting, type hints, docstrings, dead code, performance
- "Consider adding a test for X" as a standalone item
- Anything that would only matter under inputs the contract excludes

---

## Evidence standard

This is the part I care about most. I would rather receive four real findings than
twenty that sound impressive.

Every finding must carry:

- **`file:line`** and a **verbatim quote** of the offending code. If you cannot quote
  it, you do not have a finding. Do not paraphrase code from memory — re-read the file.
- **Mechanism**: the specific chain from this line to a wrong output. Not "this could
  cause issues" — name the quantity that comes out wrong.
- **Blast radius**: which reported number moves, and in which direction. If you cannot
  name the number, downgrade it to a QUESTION.
- **Why the test suite missed it**: name the specific existing test that covers this
  code path and explain why it still passes. If no test covers the path, say so
  explicitly — that is itself the answer.
- **Reproduction**: for anything you rate P0 or P1, a script in `audit/repro/` that you
  **actually ran**, which prints the wrong value next to the right one. Paste the real
  terminal output. No reproduction, no P0/P1 — demote it to P2 and label it
  `INFERRED (static only)`.
- **Confidence**: `CONFIRMED` (I executed it and saw the wrong number) /
  `LIKELY` (strong static reading, no execution) / `UNCERTAIN`.
- **Disconfirmer**: one sentence — what would I find that proves you wrong?
- **The fix in one line.** Describe it. Do not apply it.

**Severity:**
- **P0** — poisons a headline number or flips a pre-registered verdict
- **P1** — biases an arm-vs-arm comparison (the comparison is the whole paper)
- **P2** — affects a secondary or diagnostic number
- **P3** — real but numerically inert

---

## On not making things up

I am not scoring you on finding count. A module you read carefully and found nothing
wrong in is a **useful result** — report it as `CLEAN` with a one-line note on what you
checked, and move on. Padding the report with speculative findings makes it *less*
valuable to me, because I then have to spend my own time disproving your inventions
instead of fixing real bugs.

Specific rules:

- If code looks wrong but you can construct a reading under which it is deliberate,
  file it under **QUESTIONS**, not FINDINGS, and state that reading. Several things in
  this repo are counterintuitive on purpose.
- Never invent a line number, a function name, a test name, or terminal output. If you
  did not run it, do not present output as if you did.
- Do not report the same root cause as three separate findings.
- If you are running low on context, stop and say so in the report. Do not compress by
  guessing at the modules you didn't reach.

**Two calibration examples** — real bugs already found and fixed in this repo, so you
know the shape of what I'm after:

1. `train.py --pilot` computed the gamma RMSE from the *last-step* model rather than the
   best-validation checkpoint, and preferred the arm's own (deliberately corrupted)
   `gamma` label over the frozen `gamma_ref` consensus label. Every test passed. The
   number it produced feeds the headroom gate, so the gate decision was being made on a
   contaminated input.
2. Two modules write the Heston parameter columns of the network input in opposite ways
   — `pinn_provider.py` freezes them at `theta_train`, `eval_greeks.py` writes the
   regime's own parameters. Both are correct in context. Swapping them would produce
   confident, plausible, wrong Greeks with no error raised anywhere.

That is the level: quiet, plausible, numerically fatal.

---

## Scope and priority

The oracle layer (`oracle.py`, `greek_labels.py`, `make_labels.py`, `make_datasets.py`)
has already been audited by hand. Give it a **light pass only** — check its *interfaces*
with downstream consumers (are the arrays the consumers read the ones the producers
wrote? do label keys mean the same thing on both sides?) and do not re-derive the
quadrature.

Spend your effort in this order:

### Tier 1 — deepest scrutiny
`analyze_results.py`, `Hedging_backtest.py`, `gate_headroom.py`

### Tier 2 — thorough
`run_hedging.py`, `eval_greeks.py`, `run_info_matching.py`, `pinn_provider.py`,
`train_pinn.py`, `train.py`

### Tier 3 — targeted
`SobolevPINN.py`, `ude.py`, `providers.py`, `exhibits.py`

### Cross-cutting — do these regardless
Test-suite vacuity, config↔code consistency, global state (below).

---

## Module-specific hunt lists

Use these as starting points, not limits. If you find a category I haven't listed,
report it.

**`Hedging_backtest.py`**
- Is the inception premium marked at the oracle's `theta_train` price for *every* arm,
  including the oracle itself? An arm marking its own price is a free lunch.
- Are common random numbers genuinely common across arms — same paths, same order, same
  rebalance times? Trace the RNG stream construction and check no arm consumes a
  different number of draws.
- QE scheme: the `psi_c` branch switch, the martingale correction, whether the
  exponential branch's atom at zero is handled where downstream code assumes `v > 0`.
- Transaction costs: applied on every rebalance *and* on final liquidation? Per-share or
  per-notional, and does that match the contract? Applied to the absolute change in
  position, not the signed one?
- Turnover and `T_ex`: does the definition in code match the contract's? Is `T_ex`
  computed against the oracle provider named in `hedging_config.yaml`, not a hardcoded
  key?
- `tau` bookkeeping at the terminal step — off-by-one between the last rebalance and the
  liability mark.
- Discounting: applied once, consistently, at the same convention as the premium mark.
- No-trade band: applied symmetrically, and selected on validation rather than on the
  cell being reported.

**`gate_headroom.py`**
- Do the clean and noise-corrupted legs share paths (CRN)? If not, the spread is
  contaminated by Monte Carlo noise and the gate is measuring the wrong thing.
- Units of `sigma_gamma_pilot` on ingestion: absolute RMSE or relative? `train.py` emits
  both. Confirm which one is consumed and that it matches what the noise model expects.
- Is the threshold read from `heston_benchmark_v6.yaml` or hardcoded?
- Comparison direction and strictness (`<` vs `<=`), one-sided vs two-sided.
- Which checkpoint the pilot model is loaded from.

**`analyze_results.py`**
- Every threshold and every falsifier: read from the contract YAML, or duplicated in
  Python? Any duplicated constant is a drift hazard — diff them numerically and report
  any mismatch as P0.
- Verdict direction: confirm the inequality sense for each threshold against the memo.
  An inverted sign here produces a confident wrong verdict silently.
- Bootstrap: what is the resampling unit? Paths within a seed are not independent
  replicates of the arm; seeds are. Resampling the wrong unit produces CIs that are far
  too narrow.
- Pooling across seeds: mean of per-seed ratios, or ratio of pooled means? These differ,
  and only one matches the pre-registration.
- The mechanism 2×2 assignment: check the cell logic exhaustively, including ties and
  the NULL band.
- Missing cells: does a failed or absent run silently become a `NaN` that gets dropped
  from a mean, quietly changing the denominator?
- Is the confirmatory cell's result kept structurally separate from the exploratory
  sweep, or can an exploratory cell contaminate the headline?

**`eval_greeks.py`**
- Vega convention: `dV/dv0` vs sigma-vega `2*sqrt(v0)*dV/dv0`. Confirm which is compared
  against which, on both the prediction and the label side.
- RMSE normalization: per-regime denominator or global? Mixing them makes regimes
  incomparable.
- Which gamma label is used as truth — must be `gamma_ref`, never the arm's `gamma`.
- Anchor grids: loaded from the frozen artifact, and never touched during training or
  lambda selection.
- Checkpoint selection and eval dtype.

**`run_hedging.py`, `run_info_matching.py`**
- Seed derivation per cell: check for collisions between cells, and that two different
  cells cannot accidentally receive the same path bank when they shouldn't.
- Within a cell, do all arms get identical paths?
- Band/hyperparameter selection: validation only. Trace it and confirm nothing from the
  reported cell feeds back into a selection decision.
- Info-matching: is the capacity control actually matched on the axis the memo claims?

**`pinn_provider.py`**
- Input column assembly order vs `cfg.inputs` — a silent transposition of two parameter
  columns is invisible and fatal.
- Does `chunk` change results at all? Chunked and unchunked autodiff must agree
  bit-for-bit; verify by running both.
- `v_floor` application and the un-clamped extrapolation below the training range.

**`train_pinn.py`, `train.py`**
- Lambda selection touching anything other than train/val.
- Early-stop vs matched-epochs: if some arms early-stop and others run full budget, the
  comparison is confounded by training budget. Check how this is decided per arm.
- Loss-scale buffers: computed when, from what, and are they identical across arms that
  are supposed to be identical? A scale fitted on arm-specific data makes losses
  incomparable.
- Do OFF loss terms contribute exactly zero gradient, or do they still participate in
  scale normalization / total-loss denominators?
- Compute accounting: does the logged step count match what actually ran?

**`SobolevPINN.py`, `ude.py`**
- Do the config flags genuinely disable terms, or merely multiply by zero while still
  affecting autodiff graph construction, scaling, or RNG consumption?
- RNG draw order: does constructing the UDE correction net perturb the price-head init?
  The claim is that price heads are bit-identical across arms — test it directly by
  building two arms with the same seed and diffing `state_dict` tensors.
- Is `g_phi` actually zero at init, and is the UDE residual bit-identical to the base
  residual at step 0? Run it.

**`exhibits.py`**
- Any recomputation of a statistic that should be read from the frozen CSV.
- Silent row filtering, dropna, or clipping that changes what a figure shows.
- Axis/label/legend mismatches against the quantity actually plotted.

---

## Cross-cutting checks

**Config↔code consistency.** Extract every numeric literal in the Python that
corresponds to a contract quantity (thresholds, tolerances, path counts, horizons, cost
tiers, Feller bound, seeds, hypercube edges) and diff it against
`heston_benchmark_v6.yaml`, `pinn_config.yaml`, `hedging_config.yaml`. Report every
duplicate and every mismatch. Duplicates are P1 even when currently equal.

**Global state.** Grep for and assess: `np.random.` at module level, `torch.manual_seed`
outside a controlled entry point, `torch.set_default_dtype`, mutable default arguments,
module-level caches, in-place ops (`_`-suffixed torch methods) on tensors that outlive
the call, `warnings.filterwarnings`, `np.errstate`.

**Determinism.** Pick two Tier-1 entry points and run each twice in the same process and
in two fresh processes. Report any difference beyond bit-equality.

**Test-suite vacuity.** This is high-value — all tests passing is only reassuring if the
tests can fail. For each `test_*.py`, look for:
- Tests with no assertion, or asserting only `is not None` / shape / finiteness
- Tolerances loose enough that the bug class the test targets would slip through
- Tests that compare a function against a reimplementation of the same logic, so a
  shared misconception passes both
- Degenerate inputs that hide bugs: `n=1`, square arrays, `K == S`, `rho = 0`,
  `tau` identical across points, symmetric parameters
- Mocks or monkeypatches that replace the unit under test
- `skip`/`xfail` markers
- Golden-value tests where the golden value was generated by the code under test

Produce `audit/test_gaps.md` listing, per Tier-1 module, which of its *behaviors* have
no falsifying test — and for the top 5, a one-line description of the test that would
catch it.

---

## Procedure

1. Read `CLAUDE.md`, `heston_benchmark_v6.yaml`, `README.md`, `HANDOFF.md` first, plus
   `claim_memo_v6.pdf` and `roadmap_v6.pdf` if present. The contract is the spec you are
   auditing against; where code and contract disagree, the contract is right by
   definition.
2. Run `pytest -q` once and record the baseline.
3. Work module by module in the tier order above. **Append each module's section to
   `audit/FINDINGS.md` as you finish it**, before starting the next one. Do not hold the
   whole report until the end — if you run out of context or time, I want the completed
   modules on disk.
4. Maintain `audit/progress.md` with a one-line status per module (`pending` /
   `in progress` / `done — N findings` / `done — clean`) so I can see where you stopped.
5. When all tiers are done, write `audit/SUMMARY.md`.

## Output

**`audit/FINDINGS.md`** — findings in severity order, each in the format specified above.

**`audit/QUESTIONS.md`** — things that look wrong but may be deliberate. One paragraph
each: what you saw, the innocent reading, and the specific question I should answer.

**`audit/test_gaps.md`** — as described.

**`audit/SUMMARY.md`** — at most one page:
- Counts by severity
- The three findings I should fix before spending any compute, and why those three
- Which modules you audited, which you skipped, and which you only skimmed
- Where you were unable to verify something and what you'd need to verify it
- An honest statement of your own confidence in this audit's coverage

**`audit/repro/`** — every script you ran, with its real output captured alongside.

Begin with `audit/progress.md` populated with the full module list, then start Tier 1.