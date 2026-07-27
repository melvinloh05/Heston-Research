"""infra/ — dispatch + reporting scaffolding for the v6 grid.

modal_app  Modal GPU dispatch for the training grid (DRY-RUN by default).
digest     nightly one-page markdown digest over results/grid + results/hedging.

No secrets live in code: credentials come from the environment / .env only.
"""
