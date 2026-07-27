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
    assert ma.seeds_for("rung1_delta") == list(range(42, 47))


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
    grid = ma.build_grid(("rung1_delta",))
    assert grid == [("rung1_delta", s) for s in range(42, 47)]


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

def test_estimate_cost_is_gpu_hours_times_rate():
    grid = ma.build_grid(("rung1_delta",))            # 5 runs
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
    res = ma.main(["--arms", "rung1_delta", "--minutes", "10", "--rate", "1.0"])
    assert res["launched"] is False
    assert res["cost"]["usd"] == pytest.approx(5 * 10.0 / 60.0 * 1.0)


def test_launch_is_the_only_path_that_imports_modal(modal_unimportable):
    # with modal pinned un-importable, launch=True must fail AT the import,
    # proving the dry run above never touched it.
    with pytest.raises((ImportError, TypeError, AttributeError)):
        ma.dispatch_grid(ma.build_grid(("rung1_delta",)), launch=True)


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
