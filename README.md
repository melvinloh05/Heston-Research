# Sobolev-PINN Greeks for Heston

Does explicit supervision of **Gamma** (with Vega) in a trained, PDE-retaining, parametric
Heston PINN improve **delta-only** hedging PnL under misspecified dynamics — and through which
channel?

The study is **pre-registered**. Thresholds, cells, seed counts, the mechanism decision rule and
the verdict vocabulary were all fixed in `heston_benchmark_v6.yaml` before any result existed.
Every registered criterion is reported, including the ones that failed.

**Status: all runs complete.** 85 training runs, 100 hedging sweep cells, 10 seeds on the
confirmatory cell. See `results/analysis/registration_ledger.md` for the full scorecard.

---

## Headline results

| Registered criterion | Verdict |
|---|---|
| Confirmatory cell (rung3 vs standard PINN, 1% TC) | **fail** — +0.02%, CI [−0.024, +0.024] |
| Order attribution (rung2 beats rung1) | **fail** — significant reversal at 1% TC |
| Gamma-label dose-response | **flat** — monotonicity not demonstrated |
| OOD Greek RMSE at price parity | **fail** — on the parity clause only |
| Sakuma null (in-model × zero cost) | **flag** |
| Mechanism adjudication | `channel_i` |
| Bates severity sweep | `no_decisive_regime` |

The registered program returns a comprehensive null. Three findings underneath it are robust:

- **Out-of-distribution Greek accuracy.** Gamma supervision cuts OOD Gamma RMSE by 86–89% and
  Vega by 93%. All three falsification controls behave as required: shuffled Gamma labels are
  *worse* than no Gamma, the wrong model's Gamma is worse than no Gamma, and a label-free
  smoothness penalty is catastrophic. The gain is specifically the information in true labels.
- **Hedging, below a transaction-cost frontier.** 31.5% CVaR95 reduction at zero cost, holding
  across the whole calibrated cost range and reversing only above ~0.6%. Mean PnL is identical
  across arms, so this is pure tail reduction rather than return.
- **The gain is first-order.** Of the 31.5%, delta supervision earns 91.9%, gamma 6.5%, vega
  1.7% — each increment individually significant. Gamma labels make the network much better at
  Gamma; a delta-only hedge can only partly spend that.

Two of the nulls trace to a nameable design error: the registered 1% cost tier is ~200x the rate
the comparison paper calibrates for the underlying leg this strategy actually trades, and it sits
within 0.001pp of where the effect crosses zero.

---

## Single source of truth

`heston_benchmark_v6.yaml` is the contract and is **read-only**. When code and contract disagree,
the contract wins and the code changes. Amendments are dated and recorded rather than applied
silently. `make_registration_ledger.py` reads declared values live from it, by dotted key path, so
the paper and the contract cannot drift.

## Non-negotiable invariants

- Delta-only hedging of the underlying; delta-gamma is evaluation-only.
- Primary metrics are **held-out only**: OOD-parameter Greek RMSE, and misspecified delta-only
  hedging CVaR95 at TC in {0, 1%, 2%}.
- Common random numbers: all arms hedge the same paths within a cell.
- Named regimes are **evaluation anchors, never training data** — a 10%-relative-radius ball is
  excised around every one of them. Measured on the frozen artifact, the nearest training point
  to any anchor sits 2.3–3.7x outside its ball.
- Oracle: 3-way cross-validation (analytic CF / FD-on-COS / Monte Carlo) at 1e-3 relative
  tolerance, with a 4th Craig-Sneyd ADI leg required on Feller-stressed points. Disagreements are
  masked under a rule declared before training.
- One model class. Every arm is a flag setting of one config, with identical architecture,
  sampling, compute and terminal-condition treatment.
- Lambda selection on validation only.

## Pipeline

Run in order; each stage consumes the previous stage's artifacts.

```
python oracle.py --full                      # certify the 4-leg Greek oracle
python make_datasets.py                      # parameter sampling (Feller + anchor excision)
python make_labels.py                        # frozen label artifact + mask-neutrality report
python train.py --select-lambdas             # lambda selection, validation only
python train.py --arm <arm> --seed <n> --pilot   # pilot fit, supplies sigma_gamma
python gate_headroom.py --sigma-gamma-from-runlog <run>/runlog.json   # power check, BEFORE training
python train.py --arm <arm> --seed <n>       # the 85-run grid
python eval_greeks.py                        # OOD Greek metrics on the frozen anchors
python run_hedging.py confirmatory           # 10-seed confirmatory cell
python run_hedging.py full_sweep             # perturbations, Bates severity, Merton
python run_info_matching.py                  # price-budget saturation control
python analyze_results.py                    # verdicts in the contract's vocabulary
python make_registration_ledger.py           # the pre-registration ledger
python exhibits.py                           # figures and tables, pure from CSVs
```

The headroom gate runs **before** all training (`oracle_headroom_gate.runs_before`). Calibrate it
on the *baseline's* gamma error, not the best arm's — the arm choice flips the verdict.

## Repo map

| Path | Contents |
|---|---|
| `heston_benchmark_v6.yaml` | the contract — read-only |
| `pinn_config.yaml`, `hedging_config.yaml` | arm flags; engine settings |
| `oracle.py` | 4-leg Heston Greek oracle, cross-validation, certification gates |
| `make_datasets.py`, `make_labels.py`, `greek_labels.py` | sampling, frozen labels, dose-response label sources |
| `SobolevPINN.py`, `train_pinn.py`, `train.py` | one arm class, training, lambda discipline |
| `ude.py` | learned drift-correction arm (universal differential equation) |
| `providers.py`, `pinn_provider.py` | oracle and PINN hedgers behind one interface |
| `Hedging_backtest.py` | model-agnostic delta-only engine, QE paths, exact PnL decomposition |
| `run_hedging.py`, `run_info_matching.py` | confirmatory / full-sweep runners; saturation sweep |
| `gate_headroom.py` | pre-training power check |
| `eval_greeks.py`, `analyze_results.py` | OOD Greek metrics; registered verdicts |
| `make_registration_ledger.py`, `exhibits.py` | ledger table; figures and tables |
| `infra/` | Modal dispatch (optional) and nightly digest |
| `audit/` | independent correctness audit, findings and reproduction scripts |
| `docs/` | status reports and configuration audits |

## Tests

```
python -m pytest -q        # 275 tests; all must pass before any commit
python oracle.py           # quick certification
python oracle.py --full    # pre-freeze certification
```

## Reproducing

Training runs on CPU. At the contract's batch size (1024) CPU is faster than Apple MPS —
a second-order autodiff graph on a small MLP is many tiny kernels, so dispatch latency dominates;
MPS only wins above roughly batch 4000. The full 85-run grid takes about 5 hours on an M4 Pro
with two workers. `infra/modal_app.py` dispatches the same grid to GPUs if wall clock matters.

Path banks are common-random-number artifacts. When `paths_dir` is null the engine resimulates;
CRN still holds, because paths key off the cell seed rather than the arm.
