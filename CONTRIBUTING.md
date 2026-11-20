# Working on this code

Run commands from the repository root using the environment described in the [README](README.md).

```sh
python -m pytest -q
python oracle.py
python -m examples.reference_demo
python -m examples.paper_results
```

The tests cover implementation behavior using small inputs. They do not reproduce the full experiments. Cloud execution and the original frozen artifacts are unnecessary for these checks.

## Research conventions

The registered experiment is defined by `heston_benchmark_v6.yaml`. Preserve its settings and amendments when reproducing the paper. Put a new experiment's outputs in a separate directory and record any departures from the published configuration.

Changes to training, references or evaluation should preserve the held-out anchor exclusions, validation-only parameter selection, shared simulation paths within comparison cells, initial-premium convention, and fixed hedging horizon. Keep the derivative-label normalization and seed streams consistent across arms. The primary strategy trades the underlying using delta. Gamma and variance sensitivity are training targets.

Report unavailable evaluations separately from failed evaluations. The paper's negative registered results and numerical-reference qualifications are part of the research record. A new analysis should identify its inputs and distinguish exploratory comparisons from registered tests.

For a bug fix, describe the affected command or behavior and include a regression test when existing tests would miss the failure. Check the relevant tests and full suite before submitting a change. For an issue, include the Python version, command, configuration and traceback. Include only the small input needed to reproduce the problem.
