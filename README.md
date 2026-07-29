# Sobolev-PINN Greeks for Heston — v6

**One causal question.** Does explicitly supervising **Gamma** (with Vega) inside a trained,
PDE-retaining, parametric Heston PINN improve **delta-only** hedging PnL under **misspecified**
dynamics — and *through which channel*: transaction-cost/turnover, or model-uncertainty
robustness?

The claim is attributed on a **supervision ladder** (price → +Δ → +Δ+Γ → +Δ+Γ+ν; the claim lives
on the **rung1→rung2 gap**) and a **Gamma-label-noise dose-response**. The residual×supervision
factorial is completed by a Sobolev-sans-PDE arm (ω_PDE = 0). The mechanism is **adjudicated,
not assumed**: pre-registered 2×2, TC sweep {0, 1, 2%}, excess-turnover statistic `T_ex`.

---

## Status at a glance — 2026-07-29

| | |
|---|---|
| Code | **Complete** — 13.5k lines Python, one commit (`0e5071d`, 2026-07-28) |
| Tests | **189 passed**, 0 failed, ~46 s (`python -m pytest -q`) |
| Oracle | Quick self-test: **4/4 trust gates pass** |
| `data/`, `results/` | **Do not exist.** Nothing has been run. Nothing is frozen. |
| Correctness audit | Done, read-only — **0 P0 · 5 P1 · 7 P2 · 2 P3**; **none applied yet** |
| Blocking next step | Human approval to spend compute (see [Approval gates](#approval-gates)) |

There are **no study results yet**. Every number that exists today is either a self-test, a
scratchpad pilot, or an audit output — see [`docs/STATUS_2026-07-29.md`](docs/STATUS_2026-07-29.md)
for the adviser-facing summary.

---

## Single source of truth

**`heston_benchmark_v6.yaml`** is the benchmark contract. It is **read-only**: when code and the
contract disagree, *the contract wins and the code changes*. Editing it is a human decision.

Two derived configs implement it: `pinn_config.yaml` (19 arms, one architecture, training
schedule) and `hedging_config.yaml` (instrument, horizon, misspecification, costs, risk).

---

## Pipeline — run in this order

Each stage consumes the previous stage's artifact. Stages marked **GATE** must pass before the
next one is worth running.

| # | Stage | Command | Produces |
|---|---|---|---|
| 1 | **Oracle certification** — GATE | `python oracle.py` (quick) · `python oracle.py --full` (pre-freeze) | 4-leg trust gates pass/fail |
| 2 | Label artifact | `python make_labels.py --n-points N --out-dir data/labels_dev` | `labels.npz` + 4th-leg band routing + mask-neutrality report |
| 3 | Datasets + anchor grids | `python make_datasets.py --out-dir data/ds_dev` | train/val rows, held-out anchor grids |
| 4 | Pilot fit | `python train.py --arm rung3_delta_gamma_vega --pilot --data ... --out ...` | `sigma_gamma` for the gate |
| 5 | **Oracle-headroom gate** — GATE | `python gate_headroom.py --sigma-gamma <pilot> --out-dir results/gate` | Max detectable effect; **go / RETUNE** on the whole compute budget |
| 6 | λ selection (validation only) | `python train.py --select-lambdas --data ... --out lambdas_selected.yaml` | `lambdas_selected.yaml` (frozen once written) |
| 7 | Training grid | `python train.py --arm <arm> --seed <s> --data ... --out results/grid/<arm>/s<s>` · GPU: `python infra/modal_app.py --launch` | `best.pt` / `last.pt` + runlog per arm×seed |
| 8 | Info-matching (A10) | `python run_info_matching.py --data ... --out-dir results/info_matching` | Saturation curve + `info_matched_baseline` checkpoints |
| 9 | **OOD Greek RMSE** — PRIMARY | `python eval_greeks.py --ckpt-root results/grid --arms ... --seeds ... --anchors-dir ... --out-dir results/greeks` | `ood_param_greeks_agg.csv` |
| 10 | **Hedging** — PRIMARY | `python run_hedging.py confirmatory --ckpt-root results/grid --out-dir results/hedging` then `... full_sweep ...` | Per-cell PnL, CVaR95, `T_ex`, resumable ledger |
| 11 | Verdicts | `python analyze_results.py --confirmatory-dir ... --full-dir ... --out-dir results/analysis` | `threshold_verdicts.csv`, mechanism reading |
| 12 | Exhibits | `python exhibits.py --results-root results --out-dir results/exhibits` | E1–E4 figures + bit-stable CSVs |

Support: `infra/modal_app.py` (L40S dispatch — **DRY-RUN by default**; full grid = 80 runs ≈ 33
GPU-h ≈ $65 at list price) and `infra/digest.py` (nightly markdown digest, 3 anomaly classes).

---

## Repo map

**Oracle layer** — ground truth
- `oracle.py` — 4 Greek legs: trap-free CF (Albrecher 2007), COS+FD, MC, Craig–Sneyd ADI; cross-validation at `tol_rel = 1e-3`; self-test
- `greek_labels.py` — Gamma-label sources for the dose-response arms
- `make_labels.py` — per-parameter-point labels, 4th-leg band routing, mask-neutrality report
- `make_datasets.py` — chunked resumable dataset builder + anchor grids

**Model layer** — one class, every arm is a config
- `SobolevPINN.py` — the model; loss terms toggled by config flags
- `ude.py` — UDE arm (learned residual on the variance drift)
- `train_pinn.py` — training loop, compute accounting, λ selection, samplers
- `train.py` — per-arm CLI entry point

**Hedging layer**
- `Hedging_backtest.py` — model-agnostic delta-only engine, QE paths, exact PnL decomposition
- `providers.py` — oracle provider (θ_train hedger) + pathwise trust check
- `pinn_provider.py` — trained checkpoint → `GreekProvider`
- `run_hedging.py` — confirmatory cell, full sweep, band selection
- `run_info_matching.py` — A10 saturation curve

**Decision layer**
- `gate_headroom.py` — oracle-headroom gate (runs before all training)
- `eval_greeks.py` — OOD-parameter Greek RMSE
- `analyze_results.py` — threshold verdicts + mechanism adjudication
- `exhibits.py` — E1–E4, pure functions of frozen CSVs

`test_*.py` — one per module (16 files, 189 tests). `docs/` — status memo, config audit, baseline.
`audit/` — read-only v1 correctness audit (findings, questions, test gaps, 8 reproduction scripts).

---

## Non-negotiable invariants

Violating any of these **invalidates results**.

- **Strategy is delta-only** hedging of the underlying. Delta-gamma is secondary, evaluation-only.
- **Primary metrics are held-out only**: OOD-param Greek RMSE + tails (`near_feller`,
  `strong_neg_corr`; Δ, Γ, ν, vanna) and misspecified delta-only hedging **CVaR95** (loss = −PnL)
  at TC ∈ {0, 0.01, 0.02}.
- **PnL convention**: self-financing; initial premium = θ_train **oracle** price for *all* arms;
  terminal liability mark = **true-DGP** price at T′; T′ = 0.17 on the (S₀=100, K=100, τ₀=0.25)
  call; daily rebalancing (dt = 0.003968), fixed across arms. The engine **raises** rather than
  silently settling at expiry.
- **CRN**: all arms hedge the *same* paths within a cell. Path banks are frozen artifacts.
- **Training sampling**: Latin hypercube over κ∈[1,4], θ∈[0.02,0.12], ξ∈[0.20,0.60],
  ρ∈[−0.80,−0.20], v₀∈[0.01,0.12]; **reject** Feller 2κθ/ξ² < 0.40; **excise** 10%-relative-radius
  balls around every named evaluation anchor. Named regimes are **eval anchors, never training data**.
- **Oracle**: 3-way cross-validation at `tol_rel = 1e-3`; the **4th ADI leg is required** for
  `near_feller`, `feller_violating_volvol`, and any sampled point with Feller ratio ∈ [0.40, 0.60].
  Disagreeing points are masked; the mask is declared pre-training and must pass three neutrality
  checks.
- **One model class**; identical architecture and identical ansatz/terminal treatment across arms.
  λ selection on **validation only** (`LockedTestSet` enforces).
- **Seeds**: 5 default, 10 on the confirmatory cell = (combined perturbation, 1% TC, baseline
  regime, rung3 vs `standard_pinn`).
- **Loss scale normalization**: the **Gamma scale is frozen from the true consensus** (batch key
  `gamma_ref`) so every dose-response arm and the gradient-penalty arm share *one* normalization.
  Never remove `gamma_ref` from batches; never let per-arm labels set the gamma scale.
- The smoothing control is a **pure no-trade band** (no EMA). `t_ex` uses the sum-incl-endpoints
  definition.
- Every run logs: config hash, seed, wall clock, param count, derivative-eval count, peak memory.

### Frozen code — do not "fix"

`make_datasets.py` (`mc_seed = seed + 104729*ridx + 7919*j`) and `make_labels.py`
(`seed = seed + 7919*(mc_seed_offset + i)`) fold indices into a scalar seed by bare arithmetic,
violating the `np.random.default_rng([seed, _STREAM_*])` convention used everywhere else. They are
**frozen** — changing them changes the frozen labels and datasets.

Three further **declared design deviations** (anisotropic gate corruption field; gate decision
requiring CI-excludes-0 in *every* seed; the P13 pooled-diff CVaR test asserting a small residual
rather than ~0) are documented with rationales in [`CLAUDE.md`](CLAUDE.md). They are deliberate.

---

## Pre-registered thresholds

Stated **before** results. Failures are reported as honest nulls.

| Claim | Threshold |
|---|---|
| **Confirmatory pass** | Misspec delta CVaR95 improvement **≥10% relative** AND paired-bootstrap 95% CI excludes 0, at the confirmatory cell |
| **Order attribution** | rung2 beats rung1 at the cell, CI excludes 0 |
| **Dose-response** | Monotone (isotonic fit + rank correlation); flat = regularization null |
| **OOD Greek RMSE** | Γ and ν reduction **≥15%** at price parity within 10% |
| **Mechanism** | Gap at 0% TC ⇒ robustness channel (i); zero-at-0% widening with TC and `T_ex`→0 ⇒ cost channel (ii); **`T_ex` unmoved kills (ii)** regardless of PnL. Both readings publishable. |
| **In-model × κ=0** | Must reproduce the Sakuma null — consistency check, *not* pass/fail |

Headline scale-free number: **fraction of the baseline-to-oracle gap closed**.

---

## Approval gates

Free to do: write/edit code and tests, run pytest and local smoke runs, make plots/tables,
triage logs.

**Propose and wait for human approval before:** spending money (GPU dispatch) · freezing any
artifact (dataset generation into `data/frozen`, path banks, git tags) · deleting `data/` or
`results/` · editing any YAML config · adding dependencies · choosing or changing the no-trade
band width · promoting anything into `data/frozen` · editing `lambdas_selected.yaml` once written.

---

## Conventions

- Tests: `python -m pytest -q` — all must pass before any commit.
- Style: numpy-style docstrings, type hints, no new dependencies without approval.
- Environment: Python 3.12.7, torch 2.5.1, numpy 1.26.4, scipy 1.11.4, PyYAML 6.0.1 (macOS arm64).
