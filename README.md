# When the Simulator Is Wrong, Do Better Derivatives Improve Decisions?

Research code for the paper accepted at **Sim2Science: ML with Imperfect Scientific Models**, a NeurIPS 2026 workshop.

**[Read the paper](paper/paper.pdf)** · [LaTeX source](paper/main.tex) · [Reproduction guide](docs/REPRODUCIBILITY.md)

We study whether training a neural surrogate on derivatives, as well as values, improves derivative accuracy and downstream decisions. The testbed is Heston option pricing: the network predicts an option value, and a hedging rule uses its derivative to choose how much of the underlying asset to hold.

## Findings

- Derivative labels improve accuracy at held-out Heston coefficients. Adding curvature labels also reduces error in an unsupervised variance derivative by **40% and 26%** in the two evaluated regimes.
- Better derivatives do **not** establish a consistent hedging benefit specific to derivative labels under the tested model departures and transaction costs. Value-only training largely reproduces the zero-cost gains against the residual baseline.
- The preregistered confirmatory hedging test fails. The paper reports failed tests, reference uncertainty and limitations alongside the positive accuracy results.

The study uses one PDE, a nested sequence of label sets and simulated paths. It does not establish a general ranking of derivative targets or validate performance in real markets. In the code, `vega` denotes the derivative with respect to **variance**, rather than volatility.

## Setup

The dependency pins reflect the existing CPU environment. Local checks use Python 3.9.6 on macOS; a clean installation on other platforms has not been verified. Use Python 3.9–3.12 for these pins.

```sh
git clone https://github.com/melvinloh05/Heston-Research.git
cd Heston-Research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests exercise the implementation using small synthetic inputs; they do not reproduce the full paper experiments. The optional cloud runner is not needed for local use.

## Using the code

Start with the command-line help:

```sh
python make_datasets.py --help
python train.py --help
python eval_greeks.py --help
python run_hedging.py --help
```

The [reproduction guide](docs/REPRODUCIBILITY.md) explains the inputs and stage order. Large datasets, checkpoints and path arrays are **not included in this Git checkout**. The files under `results/` are historical metadata and notes, not a complete downloadable experiment bundle. Full reproduction requires the original artifacts or new generation and training runs.

## Repository map

| Path | Purpose |
| --- | --- |
| `paper/` | Camera-ready PDF, LaTeX source, official style and completed checklist |
| `heston_benchmark_v6.yaml` | Registered benchmark settings and amendments |
| `SobolevPINN.py`, `train_pinn.py`, `train.py` | Surrogate model and training |
| `oracle.py`, `greek_labels.py` | Numerical references and derivative labels |
| `make_labels.py`, `make_datasets.py` | Label and evaluation-grid generation |
| `Hedging_backtest.py`, `run_hedging.py`, `providers.py` | Hedging simulation and providers |
| `eval_greeks.py`, `analyze_results.py` | Derivative evaluation and outcome analysis |
| `test_*.py` | Implementation tests |
| `results/`, `audit/`, `docs/` | Historical records and methodological documentation |
| `archive/` | Superseded manuscripts and working notes |

Use the current paper for scientific conclusions. Older notes and result summaries retain historical wording and may contain claims superseded by the paper. Scientific settings and experimental source were not changed for this presentation cleanup.

## Citation

Melvin Loh and Prathamesh Dinesh Joshi. *When the Simulator Is Wrong, Do Better Derivatives Improve Decisions?* Sim2Science workshop, NeurIPS 2026. See [CITATION.cff](CITATION.cff).

## License

A software license has not yet been specified. Public availability alone does not grant a software reuse license. The NeurIPS style file retains its original notices.
