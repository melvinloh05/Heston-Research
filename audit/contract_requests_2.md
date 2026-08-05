# contract_requests_2.md — three dataset-sizing inputs the contract does not declare

**Status: OPEN — a human decision, requested before the freeze.** Nothing here is acted on by
fix batch 4; `heston_benchmark_v6.yaml` is untouched. Companion to
[`contract_requests.md`](contract_requests.md), which is closed (its one request became AM2-1).

**Why now.** [`dataset_sizing.md`](dataset_sizing.md) recommends `--n-param-points 512` and
every row count in it is a product of five factors. Three of those factors are **code defaults
with no contract clause**, listed there as assumptions 2, 4 and 5. The recommendation is only
as durable as those three defaults: each is a bare literal in a function signature that any
future session may change without tripping a single test, and each moves the sizing answer.

The freeze makes this urgent in one specific way: `n_param_points` will be baked into a frozen
artifact, and the argument that 512 is the right number is *conditional* on these three. If one
of them changes after the freeze, the frozen artifact is not wrong — but the memo justifying
its size is, and there is nothing in the repo that would say so.

---

## Summary

| # | input | value | declared where | contract clause today | if it changes |
|---|---|---|---|---|---|
| 1 | `val_param_frac` | `0.20` | [make_datasets.py:149](../make_datasets.py#L149) + CLI + a **second copy** at [train_pinn.py:288](../train_pinn.py#L288) | none — `splits:` covers eval holdouts only | every row count moves; at 0.25 the repo default 448 goes from PASS to FAIL |
| 2 | `n_skt` | `64` | [make_datasets.py:148](../make_datasets.py#L148) + CLI; **`make_labels` defaults to 16** | none | rows scale linearly; the measured mask rates do not transfer |
| 3 | `mc_subset_frac` | `0.10` | [make_datasets.py:149](../make_datasets.py#L149) (train/val) and [:338](../make_datasets.py#L338) (anchors) | `oracle.three_way_validation` — which this value **partially contradicts** | small sizing effect, but it sets what fraction of the artifact is genuinely 3-way validated |

Item 3 is the one I would look at first, and not for sizing reasons — see below.

---

## 1. `val_param_frac = 0.20`

**Where it lives.** Three places, and they are not one source:

- [make_datasets.py:149](../make_datasets.py#L149) — `generate_train_val(..., val_param_frac: float = 0.20)`; this is the value that determines the `split` array actually written into the artifact.
- [make_datasets.py:586](../make_datasets.py#L586) — `--val-param-frac` CLI default, same value.
- [train_pinn.py:288](../train_pinn.py#L288) — `ArmDataset._resolve_split`'s fallback for an artifact with no `split` key: a **bare `0.20` literal**, not a parameter, not read from anywhere.

**What the contract says today.** Nothing. `splits:` ([contract:270-276](../heston_benchmark_v6.yaml#L270-L276)) declares `training`, the eval anchors, the two OOD holdouts, and the tau/moneyness sanity checks. The train/validation partition of the *training* hypercube is not among them — even though λ selection is validation-only and `LockedTestSet` enforcement is a contract invariant, so the size of that validation split is load-bearing for the study's headline discipline.

**What breaks if it changes.** The whole sizing chain, and the recommendation flips:

```
train rows = n_param_points × (1 − val_param_frac) × n_skt × keep       (need ≥ 5N = 20480)

                 vf=0.20        vf=0.25        vf=0.30
   n = 448       +2.4%          −4.0% FAIL     −10.4% FAIL
   n = 512      +15.9%          +8.7%           +1.4%
   n = 544      +23.6%         +15.9%           +8.2%
```

So a move to 0.25 — an unremarkable choice a future session could make in one keystroke —
turns the repo's own default of 448 into a budget failure, and eats more than half of 512's
margin. 512 survives to 0.30 only barely.

**The separate hazard: the two copies can drift.** `make_datasets` writes `split` at *its*
value; `ArmDataset`'s fallback is hardcoded at 0.20. Change the CLI flag or the `generate_train_val`
default and the two disagree — silently for a `make_datasets` artifact (the written `split`
wins), and *wrongly* for a bare `make_labels` artifact (no `split` key → the fallback fires at
0.20 regardless). One contract key read by both would close this.

**Requested shape.** A `splits.train_val_param_frac: 0.20` (or under `training_parameterization`),
with the note that the split is BY PARAMETER POINT — `make_datasets.SPLIT_RULE` already states
the rule, it is only the fraction that is undeclared.

---

## 2. `n_skt = 64`

**Where it lives.** [make_datasets.py:148](../make_datasets.py#L148) and the `--n-skt` CLI
default. Note the **entry points disagree**: [make_labels.py:75](../make_labels.py#L75)
defaults `n_skt=16`. `dataset_sizing.md` §0.3 flags this; every number in the memo assumes 64.

**What the contract says today.** Nothing. The contract's `grid: {S.n: 41, K.n: 33, tau.n: 16}`
governs the *anchor* grids only. The hypercube artifact's rows-per-parameter-point is `n_skt`
uniform `(S, K, tau)` triples drawn once and shared across every parameter point — a code
decision end to end.

**What breaks if it changes.** Two things, one obvious and one not:

- *Obvious:* rows scale linearly, so the budget does. At `n = 512`: `n_skt=64` → 23 737 rows (PASS); `n_skt=32` → 11 869 (FAIL); `n_skt=16` → 5 934 (FAIL by 3.5×). Running the `make_labels` entry point at *its* default silently produces an artifact under a quarter of the required size — and `make_labels` runs no budget check, which is the I1 failure mode.
- *Not obvious, and the reason this is a contract question rather than a lint:* the per-category mask rates in `dataset_sizing.md` §3 (0.0685 / 0.0749 / 0.5211) were **measured at `n_skt = 64`**, on the specific shared `(S, K, tau)` draw that 64 produces. A different `n_skt` draws a different triple set, so those rates — and therefore the composed retention 0.9107 that every candidate size rests on — would have to be re-measured, not rescaled. The sizing memo would need re-running, not editing.

**Requested shape.** Declare it (e.g. `training_parameterization.sampling.n_skt: 64`), and let
both entry points read it, so `make_labels`' 16 stops being a second answer to the same question.

---

## 3. `mc_subset_frac = 0.10` — flagging this one beyond its sizing role

**Where it lives.** [make_datasets.py:149](../make_datasets.py#L149) for the train/val
artifact and [make_datasets.py:338](../make_datasets.py#L338) for the anchor grids. These are
the same number applied to **different axes**: in `generate_train_val` it is a fraction of
*parameter points* ([:196](../make_datasets.py#L196), `ceil(mc_subset_frac * n_param_points)`);
in `generate_anchor_grids` it is a fraction of *grid rows per tau slice*
([:369](../make_datasets.py#L369)). Two defaults that look like one setting.

**Sizing effect: small, and honestly reported.** The MC category masks only 0.0064 harder than
the plain one, so at 10 % weight it adds ~0.0006 to the composed mask rate —
`dataset_sizing.md` assumption 4 has this right, and no candidate size changes over any
plausible value.

**The reason it belongs in the contract anyway.** The contract declares
`oracle.three_way_validation` with `leg3: monte_carlo_pathwise_or_likelihood_ratio`
([contract:285-289](../heston_benchmark_v6.yaml#L285-L289)), and CLAUDE.md states the oracle
invariant as "3-way cross-validation (CF-analytic / FD-on-COS / MC), agreement tol_rel = 1e-3".
At `mc_subset_frac = 0.10`, **~90 % of the frozen label artifact is validated by two legs, not
three.** The MC leg runs on a 10 % subset of parameter points; everywhere else `cross_validate`
sees CF and FD only.

I am not asserting this is wrong. There is an obvious defensible reading — MC at 200 000 paths
costs ~1 s/point against CF+FD's 0.5 s, the two cheap legs are the ones that disagree most
informatively, and the fourth-leg ADI band clause already targets the regime where CF-diff is
least trustworthy. But it is a **deviation between a contract invariant and what the code
does**, decided by an undeclared default, and it is the kind of thing that should be settled
before an artifact is frozen rather than discovered in review afterwards. Either:

- the contract means "3-way on a declared subset" and should say so, with the fraction declared; or
- it means 3-way everywhere and `mc_subset_frac` is a cost concession that needs a stated rationale and a number.

**Requested shape.** A key under `oracle` — e.g. `three_way_validation.mc_coverage_frac: 0.10`
with a one-line rationale — plus, separately, a distinct name for the anchor-grid quantity so
the two axes stop sharing a default.

---

## What fix batch 4 did and did not do

- **Did not** touch the contract, and did not change any of the three values.
- **Did** close the fourth item `dataset_sizing.md` raised in the same breath — its §1 "fragility
  worth a separate ticket" and assumption 6, `make_datasets.N_INFO = 4096` as a bare literal.
  That one was unambiguously a code defect with no contract question attached (the value was
  already declared, in `pinn_config.yaml shared.n_price_points`; nothing read it), so ITEM 2
  fixed it and locked it with a parity test and a mutation test. `dataset_sizing.md` is
  annotated accordingly.
- The three above are different: each needs a value chosen by a human, not a pointer rewired.
