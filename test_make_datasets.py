"""Smoke tests for make_datasets.py (tiny sizes; freezing stays a human step).

Orchestration only — label ACCURACY is the oracle selftest's job. Covered here:
chunked resume without recomputation (part mtimes) + bit-for-bit merged/fresh
equality, the by-parameter-point split rule, the 5N budget check under a
monkeypatched (high) mask rate, anchor-grid leg routing (ADI exactly on the two
fourth-leg regimes, MC subset size + tau-slice stratification as declared), the
F7 uncertainty table, the holdout index masks, and the frozen-path refusal on
both entry points.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import make_datasets as md
import oracle
from make_datasets import generate_anchor_grids, generate_train_val, info_budget
from make_labels import LABEL_QUANTITIES

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
SEED = 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
# smoke budget floors: the production 5N/1N floors are exercised by the
# dedicated budget-check test; the fixture only needs to complete
TV_KW = dict(n_param_points=6, n_skt=4, mc_subset_frac=0.1, val_param_frac=0.2,
             chunk_size=2, leg_kwargs=SMOKE_LEGS, min_train_rows=1, min_val_rows=0)

GRID_OVR = {"S": {"n": 4}, "K": {"n": 3}, "tau": {"n": 3}}
ANCHOR_LEGS = {"mc": {"steps_per_year": 64, "chunk_pairs": 2000},
               "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
ADI_REGIMES = {"near_feller", "feller_violating_volvol"}


@pytest.fixture(scope="module")
def tv(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("tv_run1")
    res = generate_train_val(CONTRACT, CFG, str(d), SEED, **TV_KW)
    return {"dir": d, **res}


@pytest.fixture(scope="module")
def anchors(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("anchors_run1")
    res = generate_anchor_grids(CONTRACT, str(d), SEED, mc_subset_frac=0.25,
                                mc_paths=2000, leg_kwargs=ANCHOR_LEGS,
                                grid_override=GRID_OVR)
    return {"dir": d, **res}


def _part_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_chunked_resume_skips_parts_and_merges_bitforbit(tv, tmp_path):
    files = _part_files(tv["dir"] / "parts")
    assert sum(p.name == "labels.npz" for p in files) == 3       # ceil(6 / 2) parts
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    # kill-and-rerun: delete the merged file, keep parts
    (tv["dir"] / "train_val_labels.npz").unlink()
    res2 = generate_train_val(CONTRACT, CFG, str(tv["dir"]), SEED, **TV_KW)
    assert res2["manifest"]["runtime"]["parts_reused"] == [True, True, True]
    for p, t in mtimes.items():
        assert p.stat().st_mtime_ns == t, f"part recomputed on resume: {p}"
    # merged output of the resumed run == a fresh full run, bit for bit
    fresh = generate_train_val(CONTRACT, CFG, str(tmp_path / "fresh"), SEED, **TV_KW)
    a, b = np.load(res2["npz_path"]), np.load(fresh["npz_path"])
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert a[k].tobytes() == b[k].tobytes(), k
    assert res2["manifest"]["merged_sha256"] == fresh["manifest"]["merged_sha256"]


def test_split_is_by_parameter_point(tv):
    d = np.load(tv["npz_path"])
    n, n_skt = TV_KW["n_param_points"], TV_KW["n_skt"]
    split = d["split"]
    # one entry per PARAMETER POINT (aligned with axis 0 of the row arrays),
    # so every parameter index lands in exactly one split by construction
    assert split.shape == (n,)
    assert np.isin(split, (0, 1)).all()
    assert d["mask_any"].shape == (n, n_skt)
    n_val = int((split == 1).sum())
    assert abs(n_val - TV_KW["val_param_frac"] * n) <= 1         # within 1 point
    # deterministic recompute of the declared rule (rng [seed, 21])
    val_idx = np.random.default_rng([SEED, 21]).choice(n, n_val, replace=False)
    expect = np.zeros(n, dtype=np.int8)
    expect[val_idx] = 1
    assert np.array_equal(split, expect)
    assert "BY PARAMETER POINT" in tv["manifest"]["split_rule"]
    assert tv["manifest"]["n_train_points"] + tv["manifest"]["n_val_points"] == n


def test_budget_check_raises_with_exact_n_needed(tmp_path, monkeypatch):
    real = oracle.cross_validate
    pattern = np.array([True, True, True, False])                # mask rate 0.75

    def high_mask(legs, tol_rel, mc_se=None, **kw):
        cons, unc, mask = real(legs, tol_rel, mc_se, **kw)
        for g in LABEL_QUANTITIES:
            mask[g] = pattern.copy()
        return cons, unc, mask

    monkeypatch.setattr("make_labels.cross_validate", high_mask)
    out = tmp_path / "budget"
    # keep = 0.25: retained TRAIN rows = 3 pts * 4 rows * 0.25 = 3 < 8;
    # needed n = ceil(max(8 / (0.75 * 4 * 0.25), 1 / (0.25 * 4 * 0.25))) = 11
    with pytest.raises(ValueError, match=r"n_param_points >= 11"):
        generate_train_val(CONTRACT, CFG, str(out), SEED, n_param_points=4,
                           n_skt=4, chunk_size=2, mc_subset_frac=0.1,
                           val_param_frac=0.25, leg_kwargs=SMOKE_LEGS,
                           min_train_rows=8, min_val_rows=1)
    assert not (out / "train_val_labels.npz").exists(), \
        "failed budget must not emit a merged artifact"


def test_info_budget_is_the_configs_n_not_a_literal():
    """ITEM 2 (C1-class parity): the 5N/N floors the budget check acts on ARE
    `shared.n_price_points`, navigated independently out of pinn_config.yaml —
    not a module constant re-typed alongside it (the old `N_INFO = 4096`)."""
    raw = yaml.safe_load(Path(CFG).read_text())
    n = int(raw["shared"]["n_price_points"])
    assert info_budget(raw) == (5 * n, n)
    assert not hasattr(md, "N_INFO"), "the re-typed literal is back"
    # no numeric 4096 anywhere in the module's CODE (prose may still name it as
    # the value that used to be hardcoded)
    tree = ast.parse(Path(md.__file__).read_text())
    assert not [k for k in ast.walk(tree)
                if isinstance(k, ast.Constant) and k.value == 4096]
    # and a config with no N raises rather than falling back
    with pytest.raises(KeyError):
        info_budget({"shared": {}})


def test_budget_check_tracks_a_mutated_n_price_points(tmp_path):
    """ITEM 2 mutation: change n_price_points in a COPY of the config and the
    budget check must move with it. Under the old constant both halves below
    validate against 4096 (5N = 20480) and the first would fail too."""
    raw = yaml.safe_load(Path(CFG).read_text())

    def cfg_with_n(n: int) -> str:
        raw["shared"]["n_price_points"] = n
        p = tmp_path / f"pinn_n{n}.yaml"
        p.write_text(yaml.safe_dump(raw))
        return str(p)

    # N=2 -> floors (10, 2). 6 points x 4 rows, 20% val = 20 train / 4 val rows
    # retained at the smoke mask rate: PASSES only because 5N followed the config.
    out = tmp_path / "n_small"
    res = generate_train_val(CONTRACT, cfg_with_n(2), str(out), SEED,
                             n_param_points=6, n_skt=4, chunk_size=2,
                             mc_subset_frac=0.1, val_param_frac=0.2,
                             leg_kwargs=SMOKE_LEGS)
    budget = res["manifest"]["budget"]
    assert (budget["min_train_rows"], budget["min_val_rows"]) == (10, 2)
    assert budget["n_price_points"] == 2, "the manifest must record the N used"
    assert json.loads(Path(res["manifest_path"]).read_text())["budget"] == budget

    # N=8 -> floors (40, 8); 3 train points x 4 rows = 12 < 40, so it must RAISE
    # quoting the MUTATED 5N, not 20480 and not 8.
    with pytest.raises(ValueError, match=r"< 40 \(5N\)"):
        generate_train_val(CONTRACT, cfg_with_n(8), str(tmp_path / "n_big"), SEED,
                           n_param_points=4, n_skt=4, chunk_size=2,
                           mc_subset_frac=0.1, val_param_frac=0.25,
                           leg_kwargs=SMOKE_LEGS)


def test_anchor_adi_and_mc_routing_as_declared(anchors):
    man = anchors["manifest"]
    assert set(man["adi_regimes"]) == ADI_REGIMES
    assert len(man["regime_order"]) == 5
    nS, nK, nT = GRID_OVR["S"]["n"], GRID_OVR["K"]["n"], GRID_OVR["tau"]["n"]
    n_pick = int(np.ceil(0.25 * nS * nK))                        # 3 rows per tau slice
    for ridx, name in enumerate(man["regime_order"]):
        has_adi = "adi" in man["regimes"][name]["legs"]
        assert has_adi == (name in ADI_REGIMES), name
        assert (anchors["dir"] / "parts" / name / "adi.npz").exists() == has_adi
        # near_feller_mc_multiplier=4 honored exactly on the two fourth-leg regimes
        assert man["regimes"][name]["mc_n_paths_effective"] == \
            2000 * (4 if has_adi else 1)
        d = np.load(anchors["regime_npz"][name])
        for g in LABEL_QUANTITIES:
            for kind in ("consensus", "uncertainty", "mask"):
                assert d[f"{kind}_{g}"].shape == (nS, nK, nT)
        # MC subset: size and tau-slice stratification exactly as declared
        mc_mask = d["mc_mask"]
        rng = np.random.default_rng([SEED, 3, ridx])
        for j in range(nT):
            rows = np.sort(rng.choice(nS * nK, n_pick, replace=False))
            got = np.flatnonzero(mc_mask[:, :, j].ravel())
            assert got.size == n_pick
            assert np.array_equal(got, rows), (name, j)
    assert "stratified by tau slice" in man["mc_stratification"]


def test_anchor_uncertainty_table_covers_all_regimes_and_greeks(anchors):
    with open(anchors["table_path"]) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 25                                       # 5 regimes x 5 greeks
    seen = {(r["regime"], r["greek"]) for r in rows}
    assert seen == {(reg, g) for reg in anchors["manifest"]["regime_order"]
                    for g in LABEL_QUANTITIES}
    for r in rows:
        assert 0.0 <= float(r["mask_rate"]) <= 1.0
        assert float(r["uncertainty_p95"]) >= 0.0
        assert np.isfinite(float(r["uncertainty_mean"]))


def test_anchor_holdout_masks_are_declared_views(anchors):
    nS, nK, nT = GRID_OVR["S"]["n"], GRID_OVR["K"]["n"], GRID_OVR["tau"]["n"]
    d = np.load(anchors["holdout_path"])
    tau_m, wing_m = d["tau_maturity_holdout"], d["moneyness_wing_holdout"]
    assert tau_m.shape == wing_m.shape == (nS, nK, nT)
    assert tau_m[:, :, -1].all() and not tau_m[:, :, :-1].any()  # top tau slice
    S_ax = np.linspace(50.0, 150.0, nS)
    K_ax = np.linspace(60.0, 140.0, nK)
    ratio = S_ax[:, None] / K_ax[None, :]
    expect = np.broadcast_to(((ratio < 0.75) | (ratio > 1.30))[:, :, None],
                             (nS, nK, nT))
    assert np.array_equal(wing_m, expect)
    assert wing_m.any() and not wing_m.all()                     # non-vacuous
    # views on the grid, not partitions: overlap is allowed and present
    assert (tau_m & wing_m).sum() == wing_m[:, :, -1].sum()
    man = anchors["manifest"]["holdouts"]
    assert man["tau_maturity_holdout"]["n_points"] == int(tau_m.sum())
    assert man["moneyness_wing_holdout"]["n_points"] == int(wing_m.sum())
    assert np.array_equal(d["tau_flat_idx"], np.flatnonzero(tau_m.ravel()))
    assert np.array_equal(d["wing_flat_idx"], np.flatnonzero(wing_m.ravel()))
    assert np.array_equal(man["moneyness_wing_holdout"]["flat_idx"],
                          d["wing_flat_idx"])


def test_anchor_grids_resume_skips_parts(anchors):
    files = _part_files(anchors["dir"] / "parts")
    assert sum(p.name == "adi.npz" for p in files) == 2
    mtimes = {p: p.stat().st_mtime_ns for p in files}
    res2 = generate_anchor_grids(CONTRACT, str(anchors["dir"]), SEED,
                                 mc_subset_frac=0.25, mc_paths=2000,
                                 leg_kwargs=ANCHOR_LEGS, grid_override=GRID_OVR)
    for p, t in mtimes.items():
        assert p.stat().st_mtime_ns == t, f"part recomputed on resume: {p}"
    assert res2["manifest"]["runtime"]["part_seconds"] == {}     # nothing recomputed
    a = np.load(anchors["regime_npz"]["baseline"])
    b = np.load(res2["regime_npz"]["baseline"])
    for g in LABEL_QUANTITIES:
        assert np.array_equal(a[f"consensus_{g}"], b[f"consensus_{g}"],
                              equal_nan=True)


def test_frozen_path_refusal_on_both_entry_points(tmp_path):
    for bad in (tmp_path / "data" / "frozen" / "v1", tmp_path / "Frozen_anchors"):
        with pytest.raises(AssertionError, match="human"):
            generate_train_val(CONTRACT, CFG, str(bad), SEED, **TV_KW)
        with pytest.raises(AssertionError, match="human"):
            generate_anchor_grids(CONTRACT, str(bad), SEED,
                                  grid_override=GRID_OVR, leg_kwargs=ANCHOR_LEGS)
        assert not bad.exists(), "refusal must precede any write"
