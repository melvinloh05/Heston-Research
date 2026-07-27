oracle.py              4 Greek legs: trap-free CF, COS+FD, MC, Craig–Sneyd ADI
greek_labels.py        Gamma-label sources for the dose-response arms
make_labels.py         per-parameter-point label generation
make_datasets.py       chunked resumable dataset builder + anchor grids

SobolevPINN.py         the model — one class, every arm is a config
ude.py                 UDE arm (learned residual on the variance drift)
train_pinn.py          training loop, compute accounting, λ selection
train.py               per-arm CLI entry point

Hedging_backtest.py    the hedging engine
providers.py           oracle provider + pathwise trust check
pinn_provider.py       trained checkpoint → GreekProvider
run_hedging.py         confirmatory cell, full sweep, band selection
run_info_matching.py   A10 saturation curve

gate_headroom.py       the oracle-headroom gate
eval_greeks.py         OOD-parameter Greek RMSE
analyze_results.py     threshold verdicts + mechanism adjudication
exhibits.py            E1–E4, pure functions of frozen CSVs

test_*.py              one per module
*.yaml                 contract, PINN config, hedging config