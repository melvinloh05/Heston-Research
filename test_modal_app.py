"""Tests for infra/modal_app.py — the grid table, cost model, and the
DRY-RUN-without-modal contract (the lazy import must not fire on a dry run).

modal is not a project dependency; these tests force it un-importable
(sys.modules['modal'] = None) and assert the dry run still produces its plan +
cost estimate, and that only --launch reaches the import.
"""
from __future__ import annotations

import sys

import pytest

from infra import modal_app as ma


# ---------------------------------------------------------------------------
# grid / seed table
# ---------------------------------------------------------------------------

def test_high_seed_arms_get_10_others_5():
    for arm in ma.HIGH_SEED_ARMS:
        assert ma.seeds_for(arm) == list(range(42, 52))
    # sigma_010 is a genuine default-seed arm (dose-response, full-sweep only);
    # the supervision-ladder arms are all confirmatory-cell arms and get 10.
    assert ma.seeds_for("sigma_010") == list(range(42, 47))


def test_build_grid_counts():
    grid = ma.build_grid()
    n_high = len(ma.HIGH_SEED_ARMS)
    n_other = len(ma.GRID_ARMS) - n_high
    assert len(grid) == n_high * ma.N_SEEDS_HIGH + n_other * ma.N_SEEDS_DEFAULT
    # every entry is (arm, seed) with the right per-arm seed count
    by_arm: dict[str, int] = {}
    for arm, seed in grid:
        by_arm[arm] = by_arm.get(arm, 0) + 1
    for arm in ma.GRID_ARMS:
        want = ma.N_SEEDS_HIGH if arm in ma.HIGH_SEED_ARMS else ma.N_SEEDS_DEFAULT
        assert by_arm[arm] == want


def test_build_grid_subset():
    grid = ma.build_grid(("sigma_010",))
    assert grid == [("sigma_010", s) for s in range(42, 47)]
    assert ma.build_grid(("rung1_delta",)) == [("rung1_delta", s)
                                               for s in range(42, 52)]


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

def test_estimate_cost_is_gpu_hours_times_rate():
    grid = ma.build_grid(("sigma_010",))              # 5 runs
    cost = ma.estimate_cost(grid, minutes_per_run=30.0, usd_per_hr=2.0)
    assert cost["n_runs"] == 5
    assert cost["gpu_hours"] == pytest.approx(5 * 30.0 / 60.0)      # 2.5
    assert cost["usd"] == pytest.approx(2.5 * 2.0)                  # 5.0
    assert cost["gpu_type"] == ma.GPU_TYPE


def test_format_plan_mentions_runs_and_cost():
    grid = ma.build_grid(("rung1_delta",))
    plan = ma.format_plan(grid, ma.estimate_cost(grid))
    assert "DRY RUN" in plan and "rung1_delta" in plan
    assert "GPU-hr" in plan and "$" in plan


# ---------------------------------------------------------------------------
# dry run must not import modal
# ---------------------------------------------------------------------------

@pytest.fixture
def modal_unimportable(monkeypatch):
    """Force `import modal` to raise ImportError (modal is not installed here,
    but pin it so the test states the contract explicitly)."""
    monkeypatch.setitem(sys.modules, "modal", None)
    yield


def test_dry_run_without_modal(modal_unimportable, capsys):
    grid = ma.build_grid(("rung1_delta", "rung2_delta_gamma"))
    res = ma.dispatch_grid(grid, launch=False)
    assert res["launched"] is False
    assert res["grid"] == grid
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "GPU-hr" in out
    assert "dry run only" in out


def test_main_dry_run_without_modal(modal_unimportable):
    res = ma.main(["--arms", "sigma_010", "--minutes", "10", "--rate", "1.0"])
    assert res["launched"] is False
    assert res["cost"]["usd"] == pytest.approx(5 * 10.0 / 60.0 * 1.0)


def test_launch_is_the_only_path_that_imports_modal(modal_unimportable):
    # with modal pinned un-importable, launch=True must fail AT the import,
    # proving the dry run above never touched it. Point the input refs at files
    # that DO exist so the launch preflight passes and the import is reached.
    with pytest.raises((ImportError, TypeError, AttributeError)):
        ma.dispatch_grid(ma.build_grid(("rung1_delta",)), launch=True,
                         data_ref="pinn_config.yaml",
                         lambdas_ref="pinn_config.yaml")


def test_launch_preflight_refuses_missing_inputs(modal_unimportable):
    """A missing --lambdas must stop the launch, not silently train on the
    pinn_config defaults across the whole grid."""
    grid = ma.build_grid(("rung1_delta",))
    with pytest.raises(FileNotFoundError, match="refusing to launch"):
        ma.dispatch_grid(grid, launch=True, data_ref="pinn_config.yaml",
                         lambdas_ref="definitely_not_here.yaml")
    with pytest.raises(FileNotFoundError, match="refusing to launch"):
        ma.dispatch_grid(grid, launch=True, data_ref="no_such_data.npz",
                         lambdas_ref="pinn_config.yaml")


def test_dry_run_does_not_preflight(modal_unimportable):
    """Costing a grid before the artifacts are frozen stays legal."""
    res = ma.dispatch_grid(ma.build_grid(("rung1_delta",)), launch=False,
                           data_ref="not_frozen_yet.npz",
                           lambdas_ref="not_selected_yet.yaml")
    assert res["launched"] is False


# ---------------------------------------------------------------------------
# dispatch table vs. what the hedging runners actually load (drift guard)
#
# These are the checks that would have caught the two post-GPU-spend crashes:
# rung1_delta / rung2_delta_gamma dispatched at 5 seeds while run_confirmatory
# hedges them at 10, and sigma_000 hedged but never dispatched (nor aliased).
# ---------------------------------------------------------------------------

def _dispatched() -> dict[str, set]:
    """{checkpoint dir: set(seeds)} that a full default dispatch leaves on disk."""
    out: dict[str, set] = {}
    for arm, seed in ma.build_grid():
        out.setdefault(arm, set()).add(seed)
    return out


def test_grid_covers_the_hedging_runners():
    """Every (arm dir, seed) the runners load must be produced by the dispatch.

    build_providers resolves an engine arm name to a checkpoint DIR through
    pinn_provider._ARM_DIR (identity when unlisted) and raises on a missing
    best.pt, so any gap here is a crash after the GPU bill, not a warning.
    """
    import Hedging_backtest as hb
    from pinn_provider import _ARM_DIR
    from run_hedging import ALL_ARMS, _CONFIRMATORY_ARMS

    have = _dispatched()
    cfg = hb.resolve_config()
    meta = cfg["benchmark"]["meta"]
    conf_seeds = {int(meta["global_seed"]) + i
                  for i in range(int(meta["seeds_confirmatory_cell"]))}
    full_seeds = {int(s) for s in cfg["derived"]["seeds"]}

    missing: list[str] = []
    for arms, seeds, label in ((_CONFIRMATORY_ARMS, conf_seeds, "confirmatory"),
                               (ALL_ARMS, full_seeds, "full_sweep")):
        for name in arms:
            d = _ARM_DIR.get(name, name)
            gap = sorted(seeds - have.get(d, set()))
            if gap:
                missing.append(f"{label}: arm {name!r} -> dir {d!r} missing seeds {gap}")
    assert not missing, "dispatch grid does not cover the runners:\n  " + \
                        "\n  ".join(missing)


def test_confirmatory_ladder_arms_are_all_high_seed():
    """The rung1/rung2 order-attribution arms are confirmatory-cell arms, so they
    must sit in HIGH_SEED_ARMS — not the 5-seed default."""
    from pinn_provider import _ARM_DIR
    from run_hedging import _CONFIRMATORY_ARMS

    for name in _CONFIRMATORY_ARMS:
        assert _ARM_DIR.get(name, name) in ma.HIGH_SEED_ARMS, name


# ---------------------------------------------------------------------------
# artifact pull-back (no modal involved)
# ---------------------------------------------------------------------------

def test_write_artifacts_roundtrip(tmp_path):
    result = {"arm": "rung3_delta_gamma_vega", "seed": 42, "status": "ok",
              "best_pt": b"\x00\x01weights", "runlog_json": '{"arm": "rung3"}'}
    dest = ma._write_artifacts(str(tmp_path), result)
    assert (tmp_path / "rung3_delta_gamma_vega" / "s42" / "best.pt").read_bytes() \
        == b"\x00\x01weights"
    assert (tmp_path / "rung3_delta_gamma_vega" / "s42" / "runlog.json").read_text() \
        == '{"arm": "rung3"}'
    assert dest.endswith("rung3_delta_gamma_vega/s42")


def test_safe_get_converts_error_to_failed():
    class Boom:
        def get(self):
            raise RuntimeError("remote blew up")
    res = ma._safe_get(Boom(), ("rung1_delta", 42))
    assert res["status"] == "failed" and res["arm"] == "rung1_delta"
    assert "remote blew up" in res["stderr_tail"]
