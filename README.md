# Sobolev-PINN Greeks for Heston — v6

**One causal question.** Does explicitly supervising **Gamma** (with Vega) inside a trained,
PDE-retaining, parametric Heston PINN improve **delta-only** hedging PnL under **misspecified**
dynamics — and *through which channel*: transaction-cost/turnover, or model-uncertainty
robustness?

The claim is attributed on a **supervision ladder** (price → +Δ → +Δ+Γ → +Δ+Γ+ν; the claim lives
on the **rung1→rung2 gap**) and a **Gamma-label-noise dose-response**. The residual×supervision
factorial is completed by a Sobolev-sans-PDE arm (ω_PDE = 0). The mechanism is **adjudicated,
not assumed**: pre-registered 2×2, TC sweep {0, 1, 2%}, excess-turnover statistic `T_ex`.

The study is **pre-registered**. Thresholds, cells, seed counts, the mechanism decision rule and
the verdict vocabulary were fixed in `heston_benchmark_v6.yaml` before any result existed. Every
registered criterion is reported, including the ones that failed.

---

## Status at a glance — 2026-08-18

| | |
|---|---|
| Code | Complete |
| Tests | **275 passed**, 0 failed (`python -m pytest -q`) |
| Oracle | 4-leg certification gates pass |
| Runs | **Complete** — 85 training runs, 100 hedging sweep cells, 10 seeds on the confirmatory cell |
| Correctness audit | Done — 0 P0 · 5 P1 · 7 P2; batches 1–3 applied |
| Registered outcome | 4 of 8 criteria **fail**, 1 flat, 1 flag, 1 no-decisive-regime |

Full scorecard with contract provenance: [`results/analysis/registration_ledger.md`](results/analysis/registration_ledger.md).

---

## Results

The registered program returns a comprehensive null.

| Registered criterion | Verdict |
|---|---|
| Confirmatory cell (rung3 vs standard PINN, 1% TC) | **fail** — +0.02%, CI [−0.024, +0.024] |
| Order attribution (rung2 beats rung1) | **fail** — significant *reversal* at 1% TC |
| Gamma-label dose-response | **flat** — monotonicity not demonstrated |
| OOD Greek RMSE at price parity | **fail** — on the parity clause only |
| Sakuma null (in-model × zero cost) | **flag** |
| Mechanism adjudication | `channel_i` |
| Bates severity sweep | `no_decisive_regime` |

Three findings underneath it are robust:

- **Out-of-distribution Greek accuracy.** Gamma supervision cuts OOD Gamma RMSE by 86–89% and Vega
  by 93%. All three falsification controls behave as required: shuffled Gamma labels are *worse*
  than no Gamma, the wrong model's Gamma is worse than no Gamma, and a label-free smoothness
  penalty is catastrophic. The gain is specifically the information in true labels, not curvature
  regularisation.
- **Hedging, below a transaction-cost frontier.** 31.5% CVaR95 reduction at zero cost, significant
  across the whole calibrated cost range, crossing zero near 1% and reversing above it. Mean PnL is
  identical across arms, so this is pure tail reduction rather than return.
- **The gain is first-order.** Of that 31.5%, delta supervision earns 91.9%, gamma 6.5%, vega 1.7%
  — each increment individually significant below the frontier. Gamma labels make the network much
  better at Gamma; a delta-only hedge can only partly spend that.

Two of the nulls trace to one design error: the registered 1% cost tier is roughly 200× the rate
the comparison paper calibrates for the underlying leg this strategy trades, and it sits within
0.001pp of where the effect crosses zero. That tier also renders `headline_scale_free` undefined
(the oracle is worse than the baseline there) and drives the `no_decisive_regime` reading of the
Bates sweep.

---

## Single source of truth

**`heston_benchmark_v6.yaml`** is the benchmark contract. It is **read-only**: when code and the
contract disagree, *the contract wins and the code changes*. Editing it is a human decision.
Amendments are dated and recorded rather than applied silently.

Two derived configs implement it: `pinn_config.yaml` (arms, one architecture, training schedule)
and `hedging_config.yaml` (instrument, horizon, misspecification, costs, risk).

`make_registration_ledger.py` resolves declared values **live** from the contract by dotted key
path, so the contract and any table built from it cannot drift; a vanished key renders
`MISSING KEY` rather than dropping the row.

---

## Pipeline — run in this order

Each stage consumes the previous stage's artifact. Stages marked **GATE** must pass before the
next one is worth running.

| # | Stage | Command | Produces |
|---|---|---|---|
| 1 | **Oracle certification** — GATE | `python oracle.py` (quick) · `python oracle.py --full` (pre-freeze) | 4-leg trust gates pass/fail |
| 2 | Label artifact | `python make_labels.py --n-points N --out-dir data/labels_dev` | `labels.npz` + 4th-leg band routing + mask-neutrality report |
| 3 | Datasets + anchor grids | `python make_datasets.py --out-dir data/ds_dev` | train/val rows, held-out anchor grids |
| 4 | Pilot fit | `python train.py --arm <arm> --pilot --data ... --out ...` | `sigma_gamma` for the gate |
| 5 | **Oracle-headroom gate** — GATE | `python gate_headroom.py --sigma-gamma-from-runlog <run>/runlog.json --out-dir results/gate` | Max detectable effect; go / RETUNE on the whole compute budget |
| 6 | λ selection (validation only) | `python train.py --select-lambdas --data ... --out lambdas_selected.yaml` | `lambdas_selected.yaml` (frozen once written) |
| 7 | Training grid | `python train.py --arm <arm> --seed <s> --data ... --out results/grid/<arm>/s<s>` · GPU: `python infra/modal_app.py --launch` | `best.pt` / `last.pt` + runlog per arm×seed |
| 8 | Info-matching (A10) | `python run_info_matching.py --data ... --out-dir results/info_matching` | Saturation curve + `info_matched_baseline` checkpoints |
| 9 | **OOD Greek RMSE** — PRIMARY | `python eval_greeks.py --ckpt-root results/grid --arms ... --seeds ... --anchors-dir ... --out-dir results/greeks` | `ood_param_greeks_agg.csv` |
| 10 | **Hedging** — PRIMARY | `python run_hedging.py confirmatory --ckpt-root results/grid --out-dir results/hedging` then `... full_sweep ...` | Per-cell PnL, CVaR95, `T_ex`, resumable ledger |
| 11 | Verdicts | `python analyze_results.py --confirmatory-dir ... --full-dir ... --greek-agg-csv ... --labels-npz ... --out-dir results/analysis` | `threshold_verdicts.csv`, `dose_response.csv`, mechanism memo |
| 12 | Registration ledger | `python make_registration_ledger.py --verdicts ... --out-dir results/analysis` | `registration_ledger.{md,csv}` |
| 13 | Exhibits | `python exhibits.py --results-root results --out-dir results/exhibits` | E1–E4 figures + bit-stable CSVs |

Step 11 needs `--labels-npz`; without it the dose-response silently returns `null`
(NOT EVALUATED) rather than a verdict.

The headroom gate must be calibrated on the **baseline's** gamma error, not the best arm's — the
arm choice flips the verdict, and the best arm's sigma measures the residual floor after the fix
rather than the ceiling on the claim.

Support: `infra/modal_app.py` (L40S dispatch — **DRY-RUN by default**; full grid = 85 runs ≈ $69
at list price) and `infra/digest.py` (nightly markdown digest, 3 anomaly classes).

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
- `make_registration_ledger.py` — pre-registration ledger, provenance wired to contract keys
- `exhibits.py` — E1–E4, pure functions of frozen CSVs

`test_*.py` — one per module (275 tests). `docs/` — status memos, config audit, baseline.
`audit/` — read-only v1 correctness audit: brief, findings, questions, test gaps, and 16
reproduction scripts with their recorded outputs.

---

## Non-negotiable invariants

Violating any of these **invalidates results**.

- **Strategy is delta-only** hedging of the underlying. Delta-gamma is secondary, evaluation-only.
- **Primary metrics are held-out only**: OOD-param Greek RMSE + tails (`near_feller`,
  `strong_neg_corr`; Δ, Γ, ν, vanna) and misspecified delta-only hedging **CVaR95** (loss = −PnL)
  at TC ∈ {0, 0.01, 0.02}.
- **PnL convention**: self-financing; initial premium = θ_train **oracle** price for *all* arms;
  terminal liability mark = **true-DGP** price at T′; T′ = 0.17 on the (S₀=100, K=100, τ₀=0.25)
  call. Rebalancing is **43 steps at `dt_realized` = 0.003953488** (252.94/yr), fixed across arms —
  `dt` is derived, not declared: T′ = 0.17 does not divide evenly into 1/252 steps (contract
  amendment Q1). The engine **raises** rather than silently settling at expiry.
- **CRN**: all arms hedge the *same* paths within a cell. Path banks are frozen artifacts; when
  `paths_dir` is null the engine resimulates and CRN still holds, because paths key off the cell
  seed rather than the arm.
- **Training sampling**: Latin hypercube over κ∈[1,4], θ∈[0.02,0.12], ξ∈[0.20,0.60],
  ρ∈[−0.80,−0.20], v₀∈[0.01,0.12]; **reject** Feller 2κθ/ξ² < 0.40; **excise** 10%-relative-radius
  balls around every named evaluation anchor. Named regimes are **eval anchors, never training
  data** — measured on the frozen artifact, the nearest training point to any anchor sits 2.3–3.7×
  outside its ball, with zero points inside. This includes `baseline`, so the in-model hedging cell
  is free of *misspecification* but its parameter point is still **held out**.
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
| **Dose-response** | Spearman ρ > 0 (isotonic fit for shape) AND one-sided seed-bootstrap tail probability P(ρ≤0) < 0.05. A non-monotone curve is reported as **"monotonicity not demonstrated"**, *not* "is flat" (contract amendment Q2) |
| **OOD Greek RMSE** | Γ and ν reduction **≥15%** at price parity within 10% |
| **Mechanism** | Gap at 0% TC ⇒ robustness channel (i); zero-at-0% widening with TC and `T_ex`→0 ⇒ cost channel (ii); **`T_ex` unmoved kills (ii)** regardless of PnL. Both readings publishable. |
| **In-model × κ=0** | Must reproduce the Sakuma null — consistency check, *not* pass/fail |

Headline scale-free number: **fraction of the baseline-to-oracle gap closed**. Note this is
undefined at the registered 1% tier, where the oracle underperforms the baseline; at zero cost it
reads 90.8% / 97.2% / 98.9% along the ladder.

**Verdict vocabulary** (contract amendment AM2-2): `null` = NOT EVALUATED, the artifact is
legitimately absent and no claim is made. `error` = evaluation was attempted and FAILED — a defect,
never a study outcome. The two are never merged in any table, memo or figure.

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
- Training runs on **CPU**. At the contract's batch size (1024) CPU beats Apple MPS: a
  second-order autodiff graph on a small MLP is many tiny kernels, so dispatch latency dominates
  and MPS only wins above roughly batch 4000. The 85-run grid takes ~5 h on an M4 Pro with two
  workers.
