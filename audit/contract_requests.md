# contract_requests.md — keys the code needs that `heston_benchmark_v6.yaml` does not declare

## EMPTY — nothing is outstanding.

Written during fix batch 1 (branch `fix/audit-batch-1`) with ONE open request. That request
was granted by contract amendment #2 and consumed by fix batch 3, so this file now records a
closed loop rather than a queue.

| request | asked for | declared by | consumed by |
|---|---|---|---|
| 1. `confirmatory_rel_threshold` — the 10% relative CVaR95 improvement of the headline verdict | `acceptance_thresholds.confirmatory_cell_rel_min: 0.10` (the key name this file proposed) | amendment #2, AM2-1, commit `6786c0a` | fix batch 3 ITEM 6 — `Hedging_backtest.contract_thresholds` reads it; the `0.10` literal and its `TODO(C1)` marker are deleted; parity- and consumption-tested in `test_contract_thresholds.py` |

Batch 1's item 2 ("nothing else was missing") still holds: every other threshold the verdict,
gate and Greek-eval layers act on is declared in the contract and read from it. The count of
re-typed contract literals in the codebase is **0**, and
`test_contract_thresholds.py::test_contract_thresholds_match_the_yaml` is the guard that keeps
it there the next time a human edits the YAML.

**If a future batch needs a key the contract does not declare**, add it back here in the
shape batch 1 used — needed by / where the literal now lives / what the contract says today /
requested amendment / why it matters — and leave exactly one `TODO(C1)`-marked literal at the
single site that needs it, so there is one place to edit when the key lands.
