# Audit progress — COMPLETE

Baseline: `python -m pytest -q` → **189 passed, 19 warnings, 45.48s** (2026-07-28).
Repo state: no `data/` or `results/` directory exists — **nothing frozen, nothing run**.
Read-only: **no source file was modified**. Everything written lives under `audit/`.

| Module | Tier | Status |
|---|---|---|
| analyze_results.py | 1 | done — 5 findings (A1 P1, A2 P1, A3 P1, A4 P2, A5 P3) |
| Hedging_backtest.py | 1 | done — 1 finding (H1 P3); engine core CLEAN |
| gate_headroom.py | 1 | done — 2 findings (G1 P1, G2 P2) |
| run_hedging.py | 2 | done — 1 finding (R1 P2); band selection + CRN CLEAN |
| eval_greeks.py | 2 | done — clean (vega convention + grid layout verified) |
| run_info_matching.py | 2 | done — 1 finding (I1 P2) |
| pinn_provider.py | 2 | done — clean (chunking bit-exact, verified) |
| train_pinn.py | 2 | done — 1 finding (T2 P2); all invariants verified by execution |
| train.py | 2 | done — 1 finding (T1 P2) |
| SobolevPINN.py | 3 | done — clean (verified by execution) |
| ude.py | 3 | done — clean (g_phi≡0, residual bit-identical, verified) |
| providers.py | 3 | done — clean |
| exhibits.py | 3 | done — 1 finding (X1 P2) |
| oracle.py / greek_labels.py / make_labels.py / make_datasets.py | light | done — clean (interfaces only, per instruction) |
| Cross-cutting: config↔code constants | X | done — 1 finding (C1 P1): 19 duplicates, **0 mismatches** |
| Cross-cutting: global state | X | done — clean |
| Cross-cutting: determinism | X | done — clean (bit-equal in-process and cross-process) |
| Cross-cutting: test-suite vacuity | X | done — see `test_gaps.md`; no skips, no mocks of the unit under test, no assertion-free tests |

## Deliverables

- `audit/FINDINGS.md` — 14 findings in severity order (0 P0 · 5 P1 · 7 P2 · 2 P3), each with
  verbatim quote, mechanism, blast radius, why the tests missed it, reproduction,
  confidence, disconfirmer and a one-line fix. Plus per-module CLEAN sections.
- `audit/QUESTIONS.md` — 7 items that look wrong but have a coherent deliberate reading.
- `audit/test_gaps.md` — per-Tier-1-module behaviours with no falsifying test + the top 5
  tests to write.
- `audit/SUMMARY.md` — counts, the three to fix first, coverage, limits, confidence.
- `audit/repro/` — 8 scripts, all actually run, each with its captured output:

| script | what it proves | output |
|---|---|---|
| `r01_analyze_constants_vs_contract.py` | analyze_results opens no contract; 11 literals diffed | `r01_output.txt` |
| `r02_confirmatory_seed_count_and_direction.py` | `pass` on 3 seeds; mechanism sign-blindness | `r02_output.txt`, `r02b_output.txt` |
| `r03_engine_grid_and_crn.py` | realized dt vs contract dt; CRN; tc-invariance; decomposition | `r03_output.txt` |
| `r04_model_invariants.py` | price heads / loss scales / UDE residual / chunking all bit-exact | `r04_output.txt` |
| `r05_info_matching_cap_plateau.py` | row-cap plateau == information plateau | `r05_output.txt` |
| `r06_exhibits_missing_to_zero.py` | E2 draws missing T_ex at 0.0; CSV says blank | `r06_output.txt` |
| `r07_determinism.py` | Tier-1 entry points bit-identical in and across processes | `r07_output.txt` (+ `r07_run1/2.txt`) |
| `r08_contract_constants_sweep.py` | repo-wide constant inventory: 19 dup, 0 mismatch, 5 no-contract | `r08_output.txt` |
