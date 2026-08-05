"""Contract-constant parity for the three dataset-sizing/validation inputs declared
by amendment 4 (audit/contract_requests_2.md): training_parameterization.sampling.
{val_param_frac, n_skt} and oracle.three_way_validation.mc_coverage_frac.

Before amendment 4 these were bare Python literals with no contract clause — two of
them (val_param_frac, n_skt) had a SECOND, independently-editable copy that could
silently drift from the first. C1 style: doctor a copy of the contract and confirm
the resolved value follows it, so a re-typed literal (which would not move) cannot
pass.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from make_datasets import generate_anchor_grids, generate_train_val
from make_labels import generate_labels
from SobolevPINN import load_arm
from train_pinn import ArmDataset

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
SEED = 0
SMOKE_LEGS = {"mc": {"n_paths": 2000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}
GRID_OVR = {"S": {"n": 3}, "K": {"n": 3}, "tau": {"n": 2}}

BM = yaml.safe_load(open(CONTRACT))
_SAMP = BM["training_parameterization"]["sampling"]
CONTRACT_N_SKT = int(_SAMP["n_skt"])
CONTRACT_VAL_FRAC = float(_SAMP["val_param_frac"])
CONTRACT_MC_COVERAGE = float(BM["oracle"]["three_way_validation"]["mc_coverage_frac"])


def _doctored_contract(tmp_path, **overrides) -> str:
    """A copy of the contract with sampling/oracle values overridden."""
    doc = yaml.safe_load(open(CONTRACT))
    for k, v in overrides.items():
        if k in ("val_param_frac", "n_skt"):
            doc["training_parameterization"]["sampling"][k] = v
        elif k == "mc_coverage_frac":
            doc["oracle"]["three_way_validation"]["mc_coverage_frac"] = v
        else:
            raise KeyError(k)
    path = tmp_path / "doctored_contract.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(path)


# ---------------------------------------------------------------------------
# make_datasets.generate_train_val
# ---------------------------------------------------------------------------

def test_generate_train_val_defaults_are_the_contract(tmp_path):
    res = generate_train_val(CONTRACT, CFG, str(tmp_path / "tv"), SEED,
                             n_param_points=3, chunk_size=3, leg_kwargs=SMOKE_LEGS,
                             min_train_rows=0, min_val_rows=0)
    man = res["manifest"]
    assert man["n_skt"] == CONTRACT_N_SKT
    assert man["val_param_frac"] == CONTRACT_VAL_FRAC
    assert man["mc_subset_frac"] == CONTRACT_MC_COVERAGE
    assert len(man["mc_subset"]) == 3          # coverage 1.0 -> every point runs MC


def test_generate_train_val_reads_a_doctored_contract_not_a_literal(tmp_path):
    doctored = _doctored_contract(tmp_path, val_param_frac=0.5, n_skt=8,
                                  mc_coverage_frac=0.5)
    res = generate_train_val(doctored, CFG, str(tmp_path / "tv_doc"), SEED,
                             n_param_points=4, chunk_size=4, leg_kwargs=SMOKE_LEGS,
                             min_train_rows=0, min_val_rows=0)
    man = res["manifest"]
    assert man["n_skt"] == 8
    assert man["val_param_frac"] == 0.5
    assert man["mc_subset_frac"] == 0.5
    assert man["n_val_points"] == 2
    assert len(man["mc_subset"]) == 2          # ceil(0.5 * 4)


def test_generate_train_val_explicit_kwargs_still_override(tmp_path):
    res = generate_train_val(CONTRACT, CFG, str(tmp_path / "tv_ovr"), SEED,
                             n_param_points=3, n_skt=4, mc_subset_frac=0.1,
                             val_param_frac=0.25, chunk_size=3, leg_kwargs=SMOKE_LEGS,
                             min_train_rows=0, min_val_rows=0)
    man = res["manifest"]
    assert (man["n_skt"], man["val_param_frac"], man["mc_subset_frac"]) == (4, 0.25, 0.1)


# ---------------------------------------------------------------------------
# make_datasets.generate_anchor_grids
# ---------------------------------------------------------------------------

def test_generate_anchor_grids_mc_subset_frac_default_is_the_contract(tmp_path):
    res = generate_anchor_grids(CONTRACT, str(tmp_path / "anchors"), SEED,
                                mc_paths=1000, leg_kwargs=SMOKE_LEGS,
                                grid_override=GRID_OVR)
    assert res["manifest"]["mc_subset_frac"] == CONTRACT_MC_COVERAGE


def test_generate_anchor_grids_reads_a_doctored_contract_not_a_literal(tmp_path):
    doctored = _doctored_contract(tmp_path, mc_coverage_frac=0.5)
    res = generate_anchor_grids(doctored, str(tmp_path / "anchors_doc"), SEED,
                                mc_paths=1000, leg_kwargs=SMOKE_LEGS,
                                grid_override=GRID_OVR)
    assert res["manifest"]["mc_subset_frac"] == 0.5


# ---------------------------------------------------------------------------
# make_labels.generate_labels (the second, previously-disagreeing n_skt default)
# ---------------------------------------------------------------------------

def test_generate_labels_n_skt_and_mc_frac_default_are_the_contract(tmp_path):
    res = generate_labels(CONTRACT, CFG, 2, SEED, str(tmp_path / "labels"),
                          leg_kwargs=SMOKE_LEGS)
    man = res["manifest"]
    assert man["n_skt"] == CONTRACT_N_SKT
    assert man["mc_subset_frac"] == CONTRACT_MC_COVERAGE
    assert len(man["mc_subset"]) == 2          # coverage 1.0 -> every point runs MC


def test_generate_labels_reads_a_doctored_contract_not_a_literal(tmp_path):
    doctored = _doctored_contract(tmp_path, n_skt=8, mc_coverage_frac=0.5)
    res = generate_labels(doctored, CFG, 2, SEED, str(tmp_path / "labels_doc"),
                          leg_kwargs=SMOKE_LEGS)
    man = res["manifest"]
    assert man["n_skt"] == 8
    assert man["mc_subset_frac"] == 0.5
    assert len(man["mc_subset"]) == 1          # ceil(0.5 * 2)


# ---------------------------------------------------------------------------
# train_pinn.ArmDataset._resolve_split fallback (the two independent copies)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def npz_no_split(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("armds_labels")
    res = generate_labels(CONTRACT, CFG, 20, SEED, str(out),
                          n_skt=4, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)
    return res["npz_path"]


def test_resolve_split_fallback_fraction_is_the_contract(npz_no_split):
    d = np.load(npz_no_split, allow_pickle=False)
    n_points = d["mask_any"].shape[0]
    split = ArmDataset._resolve_split(d, n_points, seed=0, warn=False)
    n_val = int((split == 1).sum())
    assert n_val == min(max(round(CONTRACT_VAL_FRAC * n_points), 1), n_points - 1)


def test_resolve_split_fallback_reads_a_doctored_contract_not_a_literal(npz_no_split,
                                                                        tmp_path):
    doctored = _doctored_contract(tmp_path, val_param_frac=0.5)
    d = np.load(npz_no_split, allow_pickle=False)
    n_points = d["mask_any"].shape[0]
    split = ArmDataset._resolve_split(d, n_points, seed=0, warn=False,
                                      contract_path=doctored)
    n_val = int((split == 1).sum())
    assert n_val == round(0.5 * n_points)
    assert n_val != round(CONTRACT_VAL_FRAC * n_points)


def test_arm_dataset_end_to_end_uses_contract_path_kwarg(npz_no_split, tmp_path):
    """The public constructor plumbs contract_path through to the fallback split,
    not just the staticmethod in isolation."""
    doctored = _doctored_contract(tmp_path, val_param_frac=0.5)
    cfg = load_arm(CFG, "rung1_delta")
    tr = ArmDataset(npz_no_split, cfg, "train", seed=0, warn=False,
                    contract_path=doctored)
    va = ArmDataset(npz_no_split, cfg, "val", seed=0, warn=False,
                    contract_path=doctored)
    d = np.load(npz_no_split, allow_pickle=False)
    n_points, n_skt = d["mask_any"].shape
    keep = ~d["mask_any"].ravel()
    retained_point = np.flatnonzero(keep) // n_skt
    split = ArmDataset._resolve_split(d, n_points, seed=0, warn=False,
                                      contract_path=doctored)
    expect_val_rows = int((split[retained_point] == 1).sum())
    assert va.n_rows == expect_val_rows
    assert tr.n_rows + va.n_rows == int(keep.sum())
