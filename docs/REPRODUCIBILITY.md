# Running the research code

This repository contains experimental source and historical run metadata. The paper's large label arrays, trained checkpoints, simulated paths and numerical result tables are not included in Git. A source checkout alone cannot reproduce the published numbers without those inputs or new generation and training runs.

## Local checks

Install the dependencies as described in the root README, then run:

```sh
python -m pytest -q
python train.py --help
python eval_greeks.py --help
python run_hedging.py --help
```

The existing CPU environment passed **284 tests** on 30 October 2026. The documented command-line entry points and `python oracle.py` numerical self-tests also passed. These checks do not reproduce the full study. `python oracle.py --full` adds more expensive checks. Passing these checks is not a convergence certificate for every evaluation point.

## Experimental stages

Run commands from the repository root. Settings, seeds, held-out anchors and thresholds are in `heston_benchmark_v6.yaml`; the model, decision task and selected weights are in `pinn_config.yaml`, `hedging_config.yaml` and `lambdas_selected.yaml`. Preserve them when comparing with the paper. Use new output directories for new experiments.

1. **Labels and held-out grids:** `make_datasets.py train_val` generates the training/validation artifact; `make_datasets.py anchors` creates the evaluation grids. Inspect each subcommand's `--help` before running. Numerical label generation can be expensive.
2. **Training:** `train.py` consumes the label artifact. The main sequence is `standard_pinn`, `rung1_delta`, `rung2_delta_gamma`, and `rung3_delta_gamma_vega`; `feedforward` is the value-only comparison. Use the stored selected weights for the reported configuration. `--select-lambdas` performs a new search.
3. **Derivative evaluation:** `eval_greeks.py` consumes checkpoints and held-out anchor grids.
4. **Hedging evaluation:** `run_hedging.py confirmatory` and `run_hedging.py full_sweep` consume checkpoints and write decision-task outputs.
5. **Analysis:** `analyze_results.py` and `make_registration_ledger.py` build registered comparisons from the required inputs. Auxiliary `reanalysis_*.py` scripts cover exploratory analyses; inspect their input paths before running.

Once a training artifact is available, an example for one arm and seed is:

```sh
python train.py \
  --arm rung2_delta_gamma \
  --seed 42 \
  --data /absolute/path/to/train_val_labels.npz \
  --pinn-cfg pinn_config.yaml \
  --lambdas lambdas_selected.yaml \
  --matched-epochs \
  --device cpu \
  --out outputs/grid/rung2_delta_gamma/s42
```

Replace the label path with an actual generated artifact. One run does not reproduce the five/ten-seed comparisons. Cloud execution is optional and not required for local use.

The optional `robustness_grid.sh` and `lambda_pde_curve.sh` helpers require Zsh and expect the original label artifact at `data/frozen/v6-labels-20260812/train_val/train_val_labels.npz`. They run relative to their own checkout and stop before launching experiments if that artifact is absent. They are full experiment launchers, not installation checks.

## Interpreting the code

- An **arm** is a training configuration. Each rung adds one derivative label; this does not isolate all target interactions.
- `delta`, `gamma`, and `vega` mean price sensitivity, price curvature, and variance sensitivity. The last is not the usual volatility derivative.
- The primary hedge uses delta to trade the underlying asset. Gamma and variance derivatives are training targets, not additional inputs to that fixed rule.
- Transaction costs `0`, `0.01`, and `0.02` mean 0%, 1%, and 2%. CVaR95 measures tail loss; lower is better. Confidence intervals for loss differences have loss units.
- Solver disagreement is a reference-resolution diagnostic, not a proven error bound. Held-out Heston coefficients and processes outside the Heston model class are distinct evaluations.
- Historical prose in `results/`, `audit/`, and older notes can differ from the final interpretation. Use the camera-ready paper for conclusions and qualifications.

## Paper build

The `paper/` folder is self-contained. Compile `main.tex` with a LaTeX installation or upload that folder to Overleaf. With Tectonic installed:

```sh
cd paper
tectonic main.tex
```

`paper.pdf` is the checked camera-ready PDF. Its main text ends on page 5; acknowledgments begin on page 6. References, the completed checklist and appendices follow.
