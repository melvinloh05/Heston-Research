"""Smoke tests for make_labels.py (tiny sizes; freezing stays a human step).

Seed 0 is chosen so the 8 hypercube samples contain EXACTLY ONE point with
Feller ratio in [0.40, 0.60] (verified below, so the ADI-band assertion can
never pass vacuously). Leg kwargs are shrunk for speed; label ACCURACY is the
oracle selftest's job, not this file's — these tests check orchestration:
leg routing, mask/manifest structure, determinism, and the frozen-path refusal.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from make_labels import (LABEL_QUANTITIES, build_arm_labels, generate_labels,
                         mask_neutrality_report)
from SobolevPINN import load_arm

CONTRACT = "heston_benchmark_v6.yaml"
CFG = "pinn_config.yaml"
N_POINTS, N_SKT, SEED = 8, 4, 0
SMOKE_LEGS = {"mc": {"n_paths": 4000, "steps_per_year": 64, "chunk_pairs": 2000},
              "adi": {"nx": 101, "nv": 41, "steps_per_year": 200}}


def _generate(out_dir) -> dict:
    return generate_labels(CONTRACT, CFG, N_POINTS, SEED, str(out_dir),
                           n_skt=N_SKT, mc_subset_frac=0.1, leg_kwargs=SMOKE_LEGS)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> dict:
    return _generate(tmp_path_factory.mktemp("labels_run1"))


def test_adi_leg_present_exactly_for_feller_band_points(artifact):
    d = np.load(artifact["npz_path"])
    feller = d["feller_ratio"]
    band = (feller >= 0.40) & (feller <= 0.60)
    assert band.sum() == 1, "seed 0 must yield exactly one band point (non-vacuous test)"
    legs = artifact["manifest"]["legs_per_point"]
    for i in range(N_POINTS):
        assert ("adi" in legs[i]) == bool(band[i]), (i, feller[i], legs[i])
        assert {"cf", "fd"} <= set(legs[i])
    assert np.array_equal(d["adi_points"], band)
    # MC on the declared subset only
    mc = set(artifact["manifest"]["mc_subset"])
    assert len(mc) == 1   # ceil(0.1 * 8)
    for i in range(N_POINTS):
        assert ("mc" in legs[i]) == (i in mc)


def test_npz_mask_and_consensus_keys_exist(artifact):
    d = np.load(artifact["npz_path"])
    for g in LABEL_QUANTITIES:
        for kind in ("consensus", "uncertainty", "mask"):
            key = f"{kind}_{g}"
            assert key in d, key
            assert d[key].shape == (N_POINTS, N_SKT)
        assert d[f"mask_{g}"].dtype == bool
    assert d["mask_any"].shape == (N_POINTS, N_SKT) and d["mask_any"].dtype == bool
    expected_any = np.any(np.stack([d[f"mask_{g}"] for g in LABEL_QUANTITIES]), 0)
    assert np.array_equal(d["mask_any"], expected_any)
    assert d["params"].shape == (N_POINTS, 5)
    for k in ("S", "K", "tau"):
        assert d[k].shape == (N_SKT,)
    for g, rate in artifact["manifest"]["mask_rate"].items():
        assert 0.0 <= rate <= 1.0, (g, rate)


def test_manifest_hash_stable_across_runs_same_seed(artifact, tmp_path):
    res2 = _generate(tmp_path / "labels_run2")
    m1, m2 = artifact["manifest"], res2["manifest"]
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    # the hash covers the payload: leg routing and mask rates must match too
    assert m1["legs_per_point"] == m2["legs_per_point"]
    assert m1["mask_rate"] == m2["mask_rate"]
    # and the on-disk JSON round-trips the same deterministic fields
    disk = json.loads(open(res2["manifest_path"]).read())
    assert disk["manifest_sha256"] == m1["manifest_sha256"]
    d1, d2 = np.load(artifact["npz_path"]), np.load(res2["npz_path"])
    for g in LABEL_QUANTITIES:
        assert np.array_equal(d1[f"consensus_{g}"], d2[f"consensus_{g}"], equal_nan=True)


def test_refuses_to_write_into_frozen_paths(tmp_path):
    for bad in (tmp_path / "data" / "frozen" / "v1", tmp_path / "Frozen_labels"):
        with pytest.raises(AssertionError, match="human"):
            generate_labels(CONTRACT, CFG, N_POINTS, SEED, str(bad),
                            n_skt=N_SKT, leg_kwargs=SMOKE_LEGS)
        assert not bad.exists(), "refusal must precede any write"


def test_mask_neutrality_report_emits_three_checks(artifact, tmp_path):
    out = tmp_path / "neutrality.md"
    report, stats = mask_neutrality_report(artifact["npz_path"], out_path=str(out))
    assert out.read_text() == report
    for header in ("## (a) Mask rate by Feller-ratio band",
                   "## (b) Retained-point marginals",
                   "## (c) Mask rate by |gamma| decile"):
        assert header in report
    assert "pass" not in report.lower() or "auto-passes" in report.lower()
    assert set(stats["ks_masked_vs_retained"]) == {"kappa", "theta", "xi", "rho", "v0"}
    assert len(stats["feller_bands"]) == 5 and len(stats["gamma_deciles"]) == 10
    assert stats["overall_mask_rate"] == pytest.approx(
        artifact["manifest"]["mask_rate"]["any"])


def test_build_arm_labels_sigma_variants_differ_only_in_gamma(artifact):
    b0 = build_arm_labels(artifact["npz_path"], load_arm(CFG, "sigma_000"), seed=7)
    b25 = build_arm_labels(artifact["npz_path"], load_arm(CFG, "sigma_025"), seed=7)
    assert set(b0) == set(b25)
    n = b0["x"].shape[0]
    assert n > 0, "smoke artifact retained no points — legs disagree everywhere"
    for key in b0:
        if key == "gamma":
            continue
        assert torch.equal(b0[key], b25[key]), f"{key} must be true consensus in every arm"
    assert not torch.equal(b0["gamma"], b25["gamma"])
    # sigma_000 gamma IS the retained consensus; delta/vega always true consensus
    d = np.load(artifact["npz_path"])
    keep = ~d["mask_any"].ravel()
    assert np.allclose(b0["gamma"].numpy(),
                       d["consensus_gamma"].ravel()[keep].astype(np.float32))
    assert np.allclose(b0["delta"].numpy(),
                       d["consensus_delta"].ravel()[keep].astype(np.float32))
    # x columns follow cfg.inputs = (S, K, tau, kappa, theta, xi, rho, v0)
    assert b0["x"].shape[1] == 8
    # v6: BC supervision dropped — no x_bc/price_bc emitted for any arm
    assert "x_bc" not in b0 and "price_bc" not in b0


def test_build_arm_labels_emits_shared_gamma_ref(artifact):
    """gamma_ref = true consensus, identical across every arm (incl. the
    gradient-penalty arm, which has no gamma key), so loss scales stay
    arm-invariant."""
    d = np.load(artifact["npz_path"])
    keep = ~d["mask_any"].ravel()
    ref = d["consensus_gamma"].ravel()[keep].astype(np.float32)
    b0 = build_arm_labels(artifact["npz_path"], load_arm(CFG, "sigma_000"), seed=7)
    b50 = build_arm_labels(artifact["npz_path"], load_arm(CFG, "sigma_050"), seed=7)
    bgp = build_arm_labels(artifact["npz_path"],
                           load_arm(CFG, "gradient_penalty_only"), seed=7)
    for b in (b0, b50, bgp):
        assert "gamma_ref" in b
        assert np.allclose(b["gamma_ref"].numpy(), ref)
    assert torch.equal(b0["gamma_ref"], b50["gamma_ref"])
    assert "gamma" not in bgp and "gamma_ref" in bgp