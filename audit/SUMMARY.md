# SUMMARY — v1 pipeline correctness audit

Baseline `python -m pytest -q`: **189 passed, 19 warnings, 45.48s**.
Repo state: **no `data/`, no `results/`** — nothing has been run and nothing is frozen, so
every finding below is fixable at zero compute cost.
No source file was modified.

## Counts by severity

| | count | IDs |
|---|---|---|
| **P0** | **0** | — |
| **P1** | 5 | A1, A2, A3, G1, C1 |
| **P2** | 6 | A4, G2, R1, I1, T1, T2, X1 *(7 rows; X1 and T2 both P2)* |
| **P3** | 2 | A5, H1 |
| **QUESTIONS** | 7 | Q1–Q7 |

Corrected count: **0 P0 · 5 P1 · 7 P2 · 2 P3**, plus 7 questions.
No finding is a live numerical mismatch: the one class that could have been (duplicated
contract constants) I diffed exhaustively and **0 of 19 have drifted**.

## The three to fix before spending any compute

**1. `A2` — `_mechanism_reading` is sign-blind (`analyze_results.py:662-664`).**
This is the only finding that can produce a *confident, wrong, affirmative claim*. An arm
that is significantly **worse** at 0% TC returns `reading = "channel_i"` — the robustness
channel — identically to an arm that is significantly better; and one that degrades faster
as costs rise while trading less returns `"channel_ii"`, the cost channel. That string is
copied verbatim into `threshold_verdicts.csv` and into the mechanism memo's headline line,
and E2 is built on it. The pre-registered null ("second-order supervision doesn't improve OOD
delta hedging") would be reported as a mechanism *finding*. Two comparison operators fix it.

**2. `G1` — the gate threshold is hardcoded (`gate_headroom.py:361`).**
Fix this one *first in wall-clock order*, because the gate runs before all training and its
output is the human go/no-go on the entire compute budget. The value is currently right
(0.10 == the contract's 10%), so this is insurance, not a repair — but it is the cheapest
possible insurance on the most expensive decision in the project.

**3. `I1` — the info-matching plateau can be produced by the row cap
(`run_info_matching.py:71-85`).**
If the frozen label artifact holds fewer than `5N = 20480` training rows, the top rungs train
on bit-identical data, the curve is flat by construction, and `plateau_reached=True` is
reported — indistinguishable from a real plateau. The reported paragraph would then assert
the contract's claim ("plateau bounds what THIS architecture extracts from prices") when the
plateau actually bounds the artifact size. This must be fixed *before* the label artifact is
frozen, because the artifact size is the thing that decides whether it fires.

Honourable mention: **A3** (confirmatory verdict on a partial seed set) is the cheapest fix
of the five P1s and directly protects the headline number.

## Coverage

**Audited in full, line by line:** `analyze_results.py`, `Hedging_backtest.py`,
`gate_headroom.py` (Tier 1); `run_hedging.py`, `eval_greeks.py`, `pinn_provider.py`,
`train.py` (Tier 2); `SobolevPINN.py`, `ude.py`, `providers.py` (Tier 3).

**Audited by targeted reading of the load-bearing paths:** `train_pinn.py` (data plumbing,
`train_model`, scale freeze, λ selection, compute accounting), `run_info_matching.py`
(subsample, plateau rule, capacity control, sweep loop), `exhibits.py` (E2/E4 data paths,
`_num`/`_fmt`/`_write_csv`, missing-value handling).

**Light pass, interfaces only (as instructed):** `oracle.py`, `greek_labels.py`,
`make_labels.py`, `make_datasets.py` — producer/consumer keys, shapes, array order, param
naming, vega convention. Quadrature not re-derived.

**Skimmed only:** `exhibits.py` E1/E3 rendering internals and the matplotlib styling layer;
`infra/modal_app.py` and `infra/digest.py` (not on the audit's scope list).

**Not audited:** the four oracle legs' numerics; the ADI band routing; mask-neutrality
statistics; `test_modal_app.py` / `test_digest.py`.

## Verified by execution (not inferred)

Eight reproduction scripts, all actually run, output captured in `audit/repro/*_output.txt`:

- R01/R08 — contract-constant sweep: **19 duplicates, 0 mismatches, 5 with no contract value**
- R02 — confirmatory `pass` on 3 seeds; `_mechanism_reading` sign-blindness (both directions)
- R03 — realized `dt = 0.003953` vs contract `0.003968`; CRN bit-identity; Bates λ=0 recovery;
  tc-invariance of positions; PnL decomposition balance at all three tiers
- R04 — **all model-layer invariants hold bit-exactly**: price heads identical across 15 arms
  including UDE, `g_phi ≡ 0` at init with a bit-identical residual, loss scales identical
  across all 15 arms, OFF terms absent from the loss dict, and PINNProvider chunked ==
  unchunked at **exactly zero** difference for every chunk size
- R05 — row-cap plateau indistinguishable from an information plateau
- R06 — E2 draws a missing T_ex as a bar at exactly 0.0 while its own CSV says blank
- R07 — both Tier-1 entry points bit-identical in-process **and across fresh processes**

## Where I could not verify, and what I would need

1. **`G2` — how much the delta clip actually bites.** Quantifying the clipped fraction at
   each `sigma_rel` needs a full reference state cloud through `HestonCFProvider` at
   production sizes. Cheap in wall-clock but it is a real gate run; I did not launch one
   because the autonomy dial reserves compute spend for you. One line added to the gate's
   summary (`mean(clip binds)` per arm) settles it permanently.
2. **`I1` — whether the row cap will actually bind.** Depends on the size of the frozen label
   artifact, which does not exist yet. I demonstrated the mechanism on a synthetic dataset;
   whether it fires is decided when you choose `--n-points` for `make_labels`.
3. **Anything requiring a trained checkpoint.** No `best.pt` exists, so every provider-level
   check ran against a freshly-initialized network. That is sufficient for the structural
   properties I claimed (bit-identity, chunking, column assembly, `v = 0` finiteness) and
   insufficient for anything about *fitted* Greek quality.
4. **The gate's `field` vs `iid` T_ex ordering at production scale.** CLAUDE.md's declared
   deviation #1 rests on it; the test asserts it only at smoke size.

## Honest statement of confidence

**High** on Tier 1. I read `analyze_results.py` and `Hedging_backtest.py` in full and
executed against both; the engine in particular is the strongest module in the repo — the
premium convention, CRN, the QE scheme, the tc accounting, the exact PnL decomposition, the
T_ex definition and the two-way `gap_closed` aggregation are all correct as written, and I
tried hard to break them. `gate_headroom.py` I read in full but could only execute against
partially.

**Medium-high** on Tier 2/3. `train_pinn.py`'s invariants I verified by execution rather than
by reading (which I consider stronger for this class of property), but I read its training
loop rather than driving a full fit. `exhibits.py` I audited on the data paths, not the
rendering.

**The main thing I am least sure of** is what I could not see: this audit is entirely
pre-compute. The bug classes that survive an audit like this are the ones that only appear
against real fitted models and real frozen artifacts — a checkpoint loaded from the wrong
arm directory, a mask that turns out to be non-neutral, a Greek that is fine at
initialization and pathological after 20k steps. I found no evidence of any of those, but
absence of evidence here is weak: none of those artifacts exist to inspect.

**One structural observation.** The repo's own defensive machinery is unusually good —
`_assert_contract_targets`, `_assert_decomposition`, the `_BankLoader` sha/DGP/shape checks,
the "refusing silent expiry settlement" guards, the `LockedTestSet`, the loud
missing-checkpoint error. Almost every finding I have is in the *last mile*: the layer that
turns correct numbers into reported verdicts and figures (`analyze_results`, `gate_headroom`'s
decision row, `exhibits`' bars). That is where I would concentrate any further review, and
it is why 5 of the 7 P2s and all 3 of my priority fixes live there rather than in the physics
or the engine.
