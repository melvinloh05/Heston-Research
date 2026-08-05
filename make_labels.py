"""Hypercube training-label generation: the code path for the contract's
oracle.fourth_leg band clause ("any sampled point with feller_ratio in
[0.40, 0.60]"), which oracle_greeks (regime-name-driven) cannot serve.

Orchestration ONLY: legs and cross_validate come from oracle.py unchanged.
Per sampled parameter point the legs are CF + FD, MC on a declared seeded
subset (full MC per point is expensive), and ADI added for EVERY point whose
Feller ratio falls in [0.40, 0.60] — the band clause made executable.

FREEZING IS A HUMAN STEP: generate_labels refuses (asserts) any out_dir whose
resolved path contains 'frozen'. This module is developed and tested on tiny
smoke sizes; promoting an artifact into data/frozen is done by a human after
oracle certification and the mask-neutrality review.

Artifact layout (out_dir):
    labels.npz     consensus/uncertainty/mask per contract quantity
                   (price, delta, gamma, vega, vanna), shapes (n_points, n_skt),
                   plus params, (S, K, tau), feller_ratio, mc subset, tolerance.
    manifest.json  config hashes of both YAMLs, seed, per-point leg lists, tol,
                   mask rate per Greek, git rev if available. Deterministic
                   fields are hashed into manifest_sha256; wall clock lives
                   under "runtime", excluded from the hash.

Mask convention: mask_any = OR over the five contract label quantities only
(oracle.quantities); cross_validate's internal "any" also covers theta/d_xi/
d_rho, which are not label quantities and are not stored here.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from greek_labels import make_gamma_labels
from oracle import (HestonParams, cross_validate, heston_greeks_adi, heston_greeks_cf,
                    heston_greeks_fd, heston_greeks_mc)
from SobolevPINN import PINNConfig
from train_pinn import HESTON_PARAM_NAMES, anchors_from_contract, sample_hypercube_params

LABEL_QUANTITIES = ("price", "delta", "gamma", "vega", "vanna")   # contract oracle.quantities
FELLER_BAND = (0.40, 0.60)                                        # contract oracle.fourth_leg


def _sha256_file(path: str) -> str:
    """Hex sha256 of a file's raw bytes (config-hash convention for the manifest)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_rev() -> str | None:
    """Current git HEAD if the working directory is a repo; None otherwise."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def _assert_not_frozen(out_dir: str) -> Path:
    """Refuse any output path containing 'frozen' — artifact freezing is a human step."""
    resolved = Path(out_dir).resolve()
    assert "frozen" not in str(resolved).lower(), (
        f"out_dir {resolved} contains 'frozen': make_labels never writes frozen "
        "artifacts — a human promotes a reviewed artifact into data/frozen")
    return resolved


def generate_labels(contract_path: str, pinn_cfg_path: str, n_points: int, seed: int,
                    out_dir: str, *, n_skt: int | None = None,
                    mc_subset_frac: float | None = None,
                    leg_kwargs: dict[str, dict] | None = None,
                    params: np.ndarray | None = None,
                    skt: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                    mc_subset: np.ndarray | None = None,
                    mc_seed_offset: int = 0) -> dict:
    """Sample hypercube parameter points and build the cross-validated label artifact.

    Parameters
    ----------
    contract_path, pinn_cfg_path : str
        The benchmark contract (ranges, grid, tolerance — authoritative) and the
        engine supplement (feller_min, excision radius; values mirror the contract).
    n_points : int
        Hypercube parameter samples (Feller-rejected, anchor-excised when the
        contract declares regimes).
    seed : int
        Single seed; parameter sampling, (S, K, tau) sampling, the MC subset draw
        and per-point MC leg seeds all derive from it deterministically.
    out_dir : str
        Destination directory; must NOT contain 'frozen' (asserted).
    n_skt : int | None
        (S, K, tau) triples drawn uniformly inside the contract grid ranges,
        SHARED across parameter points (rectangular arrays, CRN-style comparisons).
        None (default) reads training_parameterization.sampling.n_skt from the
        contract; explicit values (e.g. smoke tests) override.
    mc_subset_frac : float | None
        Fraction of parameter points (ceil, at least 1) that additionally run the
        MC leg; declared in the manifest. CF + FD run everywhere. None (default)
        reads oracle.three_way_validation.mc_coverage_frac from the contract;
        explicit values (e.g. smoke tests) override.
    leg_kwargs : dict[str, dict] | None
        Per-leg keyword overrides, keys in {"cf", "fd", "mc", "adi"} — smoke tests
        pass tiny grids/path counts here; production uses oracle.py defaults.
    params, skt, mc_subset, mc_seed_offset :
        Chunked-orchestration overrides (make_datasets.generate_train_val): a
        pre-sampled parameter block of shape (n_points, 5) that SKIPS the
        internal sampler (the caller samples one global hypercube and passes
        slices, so chunking never changes the sample), a shared (S, K, tau)
        grid, explicit LOCAL MC-subset indices (may be empty; overrides the
        mc_subset_frac draw), and the chunk's global point offset so per-point
        MC leg seeds stay unique across chunks. Defaults (None/None/None/0)
        reproduce the original single-call behavior bit-for-bit.

    Returns
    -------
    dict with "npz_path", "manifest_path", "manifest".
    """
    out = _assert_not_frozen(out_dir)
    lk = {name: dict((leg_kwargs or {}).get(name, {})) for name in ("cf", "fd", "mc", "adi")}
    contract = yaml.safe_load(open(contract_path))
    pinn_raw = yaml.safe_load(open(pinn_cfg_path))
    hyc = pinn_raw.get("hypercube_sampling", {})

    params_given = params is not None
    samp = contract["training_parameterization"]["sampling"]
    n_skt = int(samp["n_skt"]) if n_skt is None else n_skt
    mc_subset_frac = (float(contract["oracle"]["three_way_validation"]["mc_coverage_frac"])
                      if mc_subset_frac is None else mc_subset_frac)
    ranges = {k: tuple(map(float, samp["ranges"][k])) for k in HESTON_PARAM_NAMES}
    if params_given:
        params = np.ascontiguousarray(np.asarray(params, dtype=float))
        assert params.shape == (n_points, len(HESTON_PARAM_NAMES)), params.shape
    else:
        anchors = anchors_from_contract(contract_path) if "regimes" in contract else None
        params = sample_hypercube_params(
            ranges, n_points, seed,
            feller_min=float(hyc.get("feller_min", 0.40)),
            method=samp.get("method", "latin_hypercube"),
            anchors=anchors,
            excise_rel_radius=float(hyc.get("excision", {}).get("rel_radius", 0.10)))
    feller = 2.0 * params[:, 0] * params[:, 1] / params[:, 2] ** 2

    gc = contract["grid"]
    r, q = float(gc["r"]), float(gc["q"])
    if skt is None:
        rng_skt = np.random.default_rng([seed, 1])
        S = rng_skt.uniform(gc["S"]["min"], gc["S"]["max"], n_skt)
        K = rng_skt.uniform(gc["K"]["min"], gc["K"]["max"], n_skt)
        tau = rng_skt.uniform(gc["tau"]["min"], gc["tau"]["max"], n_skt)
    else:
        S, K, tau = (np.asarray(a, dtype=float) for a in skt)
        assert S.shape == K.shape == tau.shape == (n_skt,), "skt must be 3 x (n_skt,)"

    if mc_subset is None:
        n_mc = max(1, int(np.ceil(mc_subset_frac * n_points)))
        mc_subset = np.sort(np.random.default_rng([seed, 2]).choice(n_points, n_mc,
                                                                    replace=False))
    else:
        mc_subset = np.sort(np.asarray(mc_subset, dtype=np.int64))
    mc_set = set(int(i) for i in mc_subset)
    tol = float(contract["oracle"]["three_way_validation"]["agreement_tol_rel"])

    t0 = time.perf_counter()
    cons = {g: np.empty((n_points, n_skt)) for g in LABEL_QUANTITIES}
    unc = {g: np.empty((n_points, n_skt)) for g in LABEL_QUANTITIES}
    msk = {g: np.zeros((n_points, n_skt), dtype=bool) for g in LABEL_QUANTITIES}
    legs_per_point: list[list[str]] = []
    for i in range(n_points):
        p = HestonParams(*(float(v) for v in params[i]))
        legs = {"cf": heston_greeks_cf(S, K, tau, p, r, q, **lk["cf"]),
                "fd": heston_greeks_fd(S, K, tau, p, r, q, **lk["fd"])}
        mc_se = None
        if i in mc_set:
            legs["mc"], mc_se = heston_greeks_mc(S, K, tau, p, r, q,
                                                 seed=seed + 7919 * (mc_seed_offset + i),
                                                 **lk["mc"])
        if FELLER_BAND[0] <= p.feller_ratio <= FELLER_BAND[1]:
            legs["adi"] = heston_greeks_adi(S, K, tau, p, r, q, **lk["adi"])
        consensus, uncertainty, mask = cross_validate(legs, tol, mc_se)
        for g in LABEL_QUANTITIES:
            cons[g][i] = getattr(consensus, g)
            unc[g][i] = getattr(uncertainty, g)
            msk[g][i] = mask[g]
        legs_per_point.append(sorted(legs))
    mask_any = np.any(np.stack([msk[g] for g in LABEL_QUANTITIES]), axis=0)

    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "labels.npz"
    arrays = {"params": params, "param_names": np.array(HESTON_PARAM_NAMES),
              "S": S, "K": K, "tau": tau, "r": np.float64(r), "q": np.float64(q),
              "feller_ratio": feller, "mc_subset": mc_subset,
              "adi_points": (feller >= FELLER_BAND[0]) & (feller <= FELLER_BAND[1]),
              "mask_any": mask_any, "seed": np.int64(seed), "tol_rel": np.float64(tol)}
    for g in LABEL_QUANTITIES:
        arrays[f"consensus_{g}"] = cons[g]
        arrays[f"uncertainty_{g}"] = unc[g]
        arrays[f"mask_{g}"] = msk[g]
    np.savez(npz_path, **arrays)

    manifest = {
        "contract_path": str(contract_path), "contract_sha256": _sha256_file(contract_path),
        "pinn_cfg_path": str(pinn_cfg_path), "pinn_cfg_sha256": _sha256_file(pinn_cfg_path),
        "seed": seed, "n_points": n_points, "n_skt": n_skt,
        "mc_subset_frac": mc_subset_frac, "mc_subset": [int(i) for i in mc_subset],
        "tol_rel": tol, "feller_band": list(FELLER_BAND),
        "leg_kwargs": lk, "legs_per_point": legs_per_point,
        # chunked-orchestration provenance (all defaults for a plain single call)
        "params_override_sha256": (hashlib.sha256(params.tobytes()).hexdigest()
                                   if params_given else None),
        "skt_override": skt is not None,
        "mc_seed_offset": int(mc_seed_offset),
        "mask_rate": {g: float(msk[g].mean()) for g in LABEL_QUANTITIES}
                     | {"any": float(mask_any.mean())},
        "git_rev": _git_rev(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["runtime"] = {"wall_clock_s": time.perf_counter() - t0}   # excluded from hash
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"npz_path": str(npz_path), "manifest_path": str(manifest_path),
            "manifest": manifest}


# ----------------------------------------------------------------------------
# Mask neutrality (contract oracle.masking.neutrality_checks) — statistics only,
# a HUMAN judges; no auto-pass/fail anywhere in this section.
# ----------------------------------------------------------------------------

_FELLER_BINS = (0.40, 0.60, 1.00, 2.00, 4.00, np.inf)


def mask_neutrality_report(labels_npz: str, out_path: str | None = None) -> tuple[str, dict]:
    """The three declared neutrality checks as reported statistics (markdown).

    (a) mask rate per Feller-ratio band — masking concentrated where a baseline
        PINN would fail (near the boundary) would bias arm comparisons;
    (b) KS distance, masked vs retained marginal, per parameter dimension —
        distributional shift of the retained set;
    (c) mask rate per |gamma| decile — preferential retention of
        smooth-agrees-with-smooth points would show up as rate rising with |gamma|.

    Saves the markdown next to the npz (or to `out_path`) and returns
    (markdown, stats). Deliberately NO thresholds: report, human judges.
    """
    from scipy.stats import ks_2samp

    d = np.load(labels_npz, allow_pickle=False)
    params, feller = d["params"], d["feller_ratio"]
    names = [str(n) for n in d["param_names"]]
    mask = d["mask_any"]                                   # (n_points, n_skt)
    n_points, n_skt = mask.shape
    flat_mask = mask.ravel()
    stats: dict = {"n_points": int(n_points), "n_skt": int(n_skt),
                   "overall_mask_rate": float(flat_mask.mean()),
                   "mask_rate_per_greek": {g: float(d[f"mask_{g}"].mean())
                                           for g in LABEL_QUANTITIES}}

    # (a) mask rate vs feller band (per parameter point, averaged over its skt grid)
    point_rate = mask.mean(axis=1)
    bands = []
    for lo, hi in zip(_FELLER_BINS[:-1], _FELLER_BINS[1:]):
        sel = (feller >= lo) & (feller < hi)
        bands.append({"band": f"[{lo:.2f}, {hi if np.isfinite(hi) else 'inf'})",
                      "n_points": int(sel.sum()),
                      "mask_rate": float(point_rate[sel].mean()) if sel.any() else float("nan")})
    stats["feller_bands"] = bands

    # (b) masked vs retained marginals per parameter dimension (flattened points)
    par_flat = np.repeat(params, n_skt, axis=0)
    ks = {}
    for j, name in enumerate(names):
        a, b = par_flat[flat_mask, j], par_flat[~flat_mask, j]
        ks[name] = float(ks_2samp(a, b).statistic) if a.size and b.size else float("nan")
    stats["ks_masked_vs_retained"] = ks

    # (c) mask rate by |gamma| decile of the consensus
    ag = np.abs(d["consensus_gamma"]).ravel()
    order = np.argsort(ag)
    deciles = []
    for k, chunk in enumerate(np.array_split(order, 10)):
        deciles.append({"decile": k + 1,
                        "abs_gamma_max": float(ag[chunk].max()) if chunk.size else float("nan"),
                        "mask_rate": float(flat_mask[chunk].mean()) if chunk.size else float("nan")})
    stats["gamma_deciles"] = deciles

    lines = ["# Mask neutrality report", "",
             f"Artifact: `{labels_npz}` — {n_points} parameter points x {n_skt} (S, K, tau).",
             f"Overall mask rate {stats['overall_mask_rate']:.3f}; per Greek: "
             + ", ".join(f"{g} {v:.3f}" for g, v in stats["mask_rate_per_greek"].items()) + ".",
             "", "Statistics only — the contract's neutrality checks are judged by a",
             "human; nothing here auto-passes or auto-fails.", "",
             "## (a) Mask rate by Feller-ratio band",
             "", "| band | n_points | mask rate |", "|---|---|---|"]
    lines += [f"| {b['band']} | {b['n_points']} | {b['mask_rate']:.3f} |" for b in bands]
    lines += ["", "Flag for review if masking concentrates in the low-Feller bands",
              "(where a baseline PINN would also fail).", "",
              "## (b) Retained-point marginals (KS distance, masked vs retained)",
              "", "| dimension | KS statistic |", "|---|---|"]
    lines += [f"| {name} | {ks[name]:.3f} |" for name in names]
    lines += ["", "## (c) Mask rate by |gamma| decile",
              "", "| decile | max abs(gamma) | mask rate |", "|---|---|---|"]
    lines += [f"| {dd['decile']} | {dd['abs_gamma_max']:.4g} | {dd['mask_rate']:.3f} |"
              for dd in deciles]
    lines += ["", "Rate rising with |gamma| would indicate preferential retention of",
              "smooth-agrees-with-smooth points (check c)."]
    report = "\n".join(lines) + "\n"

    dest = Path(out_path) if out_path else Path(labels_npz).with_name("mask_neutrality_report.md")
    dest.write_text(report)
    return report, stats


# ----------------------------------------------------------------------------
# Arm batches FROM the artifact: dose-response gamma variants share this one
# consensus; delta and vega are ALWAYS true consensus (contract A6).
# ----------------------------------------------------------------------------

def build_arm_labels(labels_npz: str, arm_cfg: PINNConfig, seed: int = 42) -> dict[str, torch.Tensor]:
    """Batch dict for one arm from the consensus artifact (masked points dropped).

    x columns follow arm_cfg.inputs; price/delta/vega/vanna are always the true
    consensus; ONLY gamma routes through greek_labels.make_gamma_labels per
    (arm_cfg.label_source, arm_cfg.gamma_label_noise_sigma) — the dose-response
    axis. label_source "none" omits the gamma key. The retained-point set is
    mask_any-filtered, identical across arms (CRN-style comparability).
    """
    d = np.load(labels_npz, allow_pickle=False)
    names = [str(n) for n in d["param_names"]]
    n_points, n_skt = d["mask_any"].shape
    keep = ~d["mask_any"].ravel()
    cols = {"S": np.tile(d["S"], n_points), "K": np.tile(d["K"], n_points),
            "tau": np.tile(d["tau"], n_points)}
    for j, name in enumerate(names):
        cols[name] = np.repeat(d["params"][:, j], n_skt)
    labels = {g: d[f"consensus_{g}"].ravel()[keep] for g in LABEL_QUANTITIES}
    cols = {k: v[keep] for k, v in cols.items()}

    gamma = make_gamma_labels(arm_cfg.label_source, labels["gamma"],
                              sigma=arm_cfg.gamma_label_noise_sigma, seed=seed,
                              price=labels["price"], S=cols["S"], K=cols["K"],
                              tau=cols["tau"], r=float(d["r"]), q=float(d["q"]))
    dtype = torch.float64 if arm_cfg.dtype == "float64" else torch.float32
    x = torch.stack([torch.as_tensor(cols[name], dtype=dtype) for name in arm_cfg.inputs], 1)
    batch = {"x": x,
             "price": torch.as_tensor(labels["price"], dtype=dtype),
             "delta": torch.as_tensor(labels["delta"], dtype=dtype),
             "vega": torch.as_tensor(labels["vega"], dtype=dtype),
             "vanna": torch.as_tensor(labels["vanna"], dtype=dtype),
             # TRUE consensus gamma for EVERY arm (incl. corrupted-label and
             # label_source "none" arms): the loss-scale reference, so
             # loss_scale_gamma is bit-identical across the whole dose-response
             # ladder — arms differ only by flags/labels, never by normalization.
             # Never used as a training target.
             "gamma_ref": torch.as_tensor(labels["gamma"], dtype=dtype)}
    # v6: NO boundary-condition supervision (shared use_bc: false); the batch carries
    # no x_bc/price_bc. Constraints come from price+Greek supervision and the PDE residual.
    if gamma is not None:
        batch["gamma"] = torch.as_tensor(gamma, dtype=dtype)
    return batch


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--pinn-cfg", default="pinn_config.yaml")
    ap.add_argument("--n-points", type=int, required=True)
    ap.add_argument("--n-skt", type=int, default=None,
                    help="default: contract training_parameterization.sampling.n_skt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True, help="must not contain 'frozen'")
    ap.add_argument("--mc-subset-frac", type=float, default=None,
                    help="default: contract oracle.three_way_validation.mc_coverage_frac")
    args = ap.parse_args()
    res = generate_labels(args.contract, args.pinn_cfg, args.n_points, args.seed,
                          args.out_dir, n_skt=args.n_skt,
                          mc_subset_frac=args.mc_subset_frac)
    print(f"wrote {res['npz_path']}\nwrote {res['manifest_path']}")
    report, _ = mask_neutrality_report(res["npz_path"])
    print(report)