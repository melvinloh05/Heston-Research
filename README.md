# When the Simulator Is Wrong, Do Better Derivatives Improve Decisions?

[![CPU checks](https://github.com/melvinloh05/Heston-Research/actions/workflows/tests.yml/badge.svg)](https://github.com/melvinloh05/Heston-Research/actions/workflows/tests.yml)

Code and selected result tables for the paper by **Melvin Loh and Prathamesh Dinesh Joshi**, accepted at the **Sim2Science workshop, NeurIPS 2026**.

**[Paper](paper/paper.pdf)** · [Paper source](paper/) · [Saved results](results/README.md) · [Reproduction guide](docs/REPRODUCIBILITY.md)

## What the study asks

Does learning a simulator's derivatives more accurately improve the decisions made using them when the simulator is wrong? We study this in Heston option pricing. A neural surrogate predicts an option value, and a fixed hedging rule uses its price sensitivity to choose an asset holding.

Adding curvature labels improves an unsupervised variance derivative by **40% and 26%** in two held-out coefficient regimes. These accuracy gains do not establish a consistent hedging benefit from derivative labels. Value-only training largely reproduces the zero-cost gains against the residual baseline, and the preregistered confirmatory hedging test fails.

The practical lesson is to evaluate derivative accuracy alongside the decisions that use it. The training baseline, the region where errors are measured, and the derivatives consumed by the decision rule all affect how the results should be interpreted. The evidence covers one PDE and simulated paths. Transfer to other scientific systems and real-market performance remain open questions.

## Quick start

Run these commands from a terminal with Python 3.11 available. A GPU, cloud account and pretrained checkpoint are unnecessary for the examples and tests.

```sh
git clone https://github.com/melvinloh05/Heston-Research.git
cd Heston-Research
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m examples.reference_demo
python -m examples.paper_results
python -m pytest -q
```

On Windows, activate the environment with `.venv\Scripts\activate`. On Linux, you can install the CPU-only PyTorch wheel before installing the requirements to avoid downloading CUDA libraries.

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

The numerical example evaluates a Heston call at three asset prices and compares the Fourier and finite-difference-on-COS references. The results example reads saved tables and prints the cross-target accuracy comparison and the residual-weight hedging comparison. Neither command trains a model or writes experiment files.

The dependency versions are pinned in `requirements.txt` and `requirements-dev.txt`. GitHub Actions runs the implementation tests, numerical self-tests, both examples and command-line help on Python 3.11 with CPU PyTorch. Local checks also cover the existing Python 3.9.6 macOS environment.

## From the paper to the code

| Component | Entry point | Role |
| --- | --- | --- |
| Surrogate | [`SobolevPINN.py`](SobolevPINN.py) | Parametric neural model, input derivatives and loss terms |
| Training | [`train.py`](train.py), [`train_pinn.py`](train_pinn.py) | Arm selection, validation, checkpoints and run accounting |
| Numerical references | [`oracle.py`](oracle.py), [`greek_labels.py`](greek_labels.py) | Reference solvers, derivative labels and uncertainty checks |
| Data generation | [`make_datasets.py`](make_datasets.py), [`make_labels.py`](make_labels.py) | Training labels and held-out evaluation grids |
| Derivative evaluation | [`eval_greeks.py`](eval_greeks.py) | Accuracy across held-out coefficients and evaluation regions |
| Decision evaluation | [`run_hedging.py`](run_hedging.py), [`Hedging_backtest.py`](Hedging_backtest.py), [`providers.py`](providers.py) | Common-path hedging, costs and tail-loss measurements |
| Registered analysis | [`analyze_results.py`](analyze_results.py), [`make_registration_ledger.py`](make_registration_ledger.py) | Thresholds, comparisons and verdicts |
| Experiment settings | [`heston_benchmark_v6.yaml`](heston_benchmark_v6.yaml) | Registered seeds, splits, thresholds and amendments |

The main training configurations form a nested sequence.

| Training arm | Value labels | PDE residual | Derivative labels |
| --- | --- | --- | --- |
| `feedforward` | Yes | No | None |
| `standard_pinn` | Yes | Yes | None |
| `rung1_delta` | Yes | Yes | Delta |
| `rung2_delta_gamma` | Yes | Yes | Delta and gamma |
| `rung3_delta_gamma_vega` | Yes | Yes | Delta, gamma and variance sensitivity |

Result tables abbreviate the three supervised arms as `rung1`, `rung2` and `rung3`. In this code, `vega` means sensitivity to **variance**, rather than volatility. The registered hedge uses delta alone. The nested training design measures incremental effects in this ordering and cannot isolate every label interaction.

## Reproduction and available artifacts

The checkout includes selected per-seed accuracy and hedging tables, run metadata, implementation tests, registered settings, and the accepted camera-ready paper with its source. The [results guide](results/README.md) maps the published tables to paper comparisons and records their checksums.

Full reproduction also requires the original label arrays, checkpoints and path arrays, which are not bundled here. The saved tables allow readers to inspect reported comparisons without those large artifacts. Recomputing pathwise bootstrap intervals or retraining the models requires the underlying inputs. The [reproduction guide](docs/REPRODUCIBILITY.md) explains each stage and its prerequisites.

```sh
python make_datasets.py --help
python train.py --help
python eval_greeks.py --help
python run_hedging.py --help
```

`audit/` and `archive/` preserve development history. Older prose under those directories and `results/` may contain preliminary interpretations. Use the current paper for scientific conclusions and qualifications.

## Citation

Melvin Loh and Prathamesh Dinesh Joshi. *When the Simulator Is Wrong, Do Better Derivatives Improve Decisions?* Sim2Science workshop, NeurIPS 2026. Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## Questions and changes

For a code issue, include the command, Python version, error output and relevant configuration. See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and the experiment conventions that changes need to preserve.

## License

A software license has not yet been specified. The NeurIPS style file retains its original notices.
