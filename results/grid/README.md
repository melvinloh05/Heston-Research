# `results/grid/` — trained checkpoint artifacts

One directory per arm, one subdirectory per seed: `<arm>/s<seed>/{best.pt, last.pt,
runlog.json, loss_curves.csv}`. `best.pt` is the best-validation model; `last.pt` is the
matched-epochs endpoint (the contract asks for both). Every fit ran `train.py
--matched-epochs` at 20,000 steps on CPU; `runlog.json` carries the wall clock, parameter
count, derivative-eval count and peak memory the contract requires.

## Reading a checkpoint's `cfg` — one trap

`best.pt["cfg"]` is the arm's `PINNConfig` as a dict. **Do not quote its `lambda_pde`
field directly.** `SobolevPINN.loss` gates the residual on
`cfg.use_pde and cfg.lambda_pde != 0.0`, so on an arm with `use_pde=False` the stored
`lambda_pde` was never applied and is simply whatever `pinn_config.yaml`'s default was.

Concretely, in this directory:

| arm | `use_pde` | stored `lambda_pde` | weight the residual ACTUALLY carried |
|---|---|---|---|
| `standard_pinn`, `rung1_delta`, `rung2_delta_gamma`, `rung3_delta_gamma_vega` | `True` | `0.01` | **0.01** (the selected value) |
| `info_matched_baseline` | `True` | `1.0` | 1.0 |
| `gradient_penalty_only` | `True` | `0.01` | 0.01 |
| `sobolev_sans_pde`, `feedforward` | `False` | `1.0` | **0.0 — the term is off** |

Read literally, the last row says the two PDE-free arms carried the heaviest residual in
the study, which is the exact opposite of the truth. `train._apply_lambdas` deliberately
declines to override `lambda_pde` on a residual-free arm (so the Sakuma / DML-no-PDE arms
keep their pinned configuration), which is why the default survives into the artifact.

Use `SobolevPINN.effective_lambda_pde(cfg)` — it returns 0.0 whenever the term is off and
the applied weight otherwise. It accepts either a `PINNConfig` or the checkpoint's `cfg`
dict. Guarded by `test_sobolev_pinn.py::test_effective_lambda_pde_*`.

These checkpoints are **registered artifacts and are not rewritten** to tidy the field.

## Related checkpoint roots

| root | what it is |
|---|---|
| `results/grid/` | the registered production grid, `lambda_pde = 0.01` |
| `results/grid_robustness/` | the contract's `lambda_selection.robustness_row`: the same two confirmatory arms at the rung3-sourced `lambda_pde = 0.0`. **Bit-identical weights** to `results/grid/feedforward` and `results/grid/sobolev_sans_pde` respectively (verified in `docs/CODE_AUDIT_2026-08-20.md` §1), because at `lambda_pde = 0` those configurations coincide. |
| `results/lambda_sweep/grid/<lam_tag>/` | `docs/CODE_AUDIT_2026-08-20.md` action 2: a **sensitivity analysis** over the interval below the registered value that the registered candidate grid never explored. Not a re-selection; the registered `lambda_pde` remains 0.01 and no registered verdict reads these. |
