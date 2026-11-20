# Saved result tables

These CSV files are copied unchanged from the study's saved evaluation outputs. They allow inspection of selected paper comparisons without training models or downloading checkpoints. The file sizes and SHA-256 hashes are recorded in [`published_tables.json`](published_tables.json).

| Table | Contents | Paper comparison |
| --- | --- | --- |
| [`eval_greeks_full/ood_param_greeks.csv`](eval_greeks_full/ood_param_greeks.csv) | Per-seed derivative errors on the full held-out grids | Derivative accuracy, supervision ladder and cross-target transfer |
| [`eval_greeks_hedgeslice/ood_param_greeks.csv`](eval_greeks_hedgeslice/ood_param_greeks.csv) | Full-grid and restricted-region errors | Evaluation-region ranking reversal in Section 3.6 and Appendix D |
| [`hedging/confirmatory/headline_delta_only_per_seed.csv`](hedging/confirmatory/headline_delta_only_per_seed.csv) | Registered coefficient-shift evaluation across costs | Confirmatory hedging and selected-residual-weight comparisons |
| [`hedging_robustness/confirmatory/headline_delta_only_per_seed.csv`](hedging_robustness/confirmatory/headline_delta_only_per_seed.csv) | Residual-free sensitivity comparison across costs | Effect of residual weighting, discussed further in the expanded manuscript |
| [`hedging_sweep/full_sweep/headline_delta_only_per_seed.csv`](hedging_sweep/full_sweep/headline_delta_only_per_seed.csv) | Coefficient shifts, Bates and Merton departures across costs | Model departures and transaction-cost sensitivity |

From the repository root, run the following to read two comparisons directly from these tables.

```sh
python -m examples.paper_results
```

## Reading the columns

- `regime` names the held-out Heston coefficient setting. `near_feller` and `strong_neg_corr` are the paper's interpolated and extrapolated anchors.
- `slice` distinguishes the full grid from the restricted hedging region.
- `arm` or `method` identifies the model. Accuracy tables use `rung1`, `rung2`, `rung3`, and `sans_pde` for the longer training-arm names. `vega` is variance sensitivity.
- `seed` identifies the training seed. Accuracy comparisons use five seeds. Confirmatory comparisons use ten. The broader departure sweep uses five, as recorded in the paper.
- `rmse` and `rel_rmse` measure derivative or value error. `oracle_unc_rms` records a solver-disagreement diagnostic, not a proven error bound.
- `tc` is the proportional transaction-cost rate. `0.01` means 1%.
- `cvar` is the mean of the worst 5% of simulated losses. Lower is better.
- `in_model`, `sweep`, `direction`, `magnitude`, `lambda_j` and `sigma_j` distinguish evaluation cells. Match all applicable cell fields before comparing methods.
- `pnl_vs_baseline_ci_lo` and `pnl_vs_baseline_ci_hi` are recorded pathwise comparison intervals. They are in loss units. Averaging interval endpoints across seeds does not produce a confidence interval for the aggregate comparison.

The per-seed files contain both registered and exploratory cells. Use the [paper](../paper/paper.pdf) and [registered settings](../heston_benchmark_v6.yaml) to identify the relevant comparison. The saved summaries do not contain individual path outcomes, so they cannot recreate the paired-bootstrap confirmatory interval.

Other subdirectories retain historical run metadata and notes. The original label arrays, checkpoint weights, path banks and remaining output tables are not included in this checkout. The [reproduction guide](../docs/REPRODUCIBILITY.md) describes the prerequisites for recomputing them.
