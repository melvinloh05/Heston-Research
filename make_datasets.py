"""Production dataset builders: chunked, resumable orchestration on top of
make_labels (TASK A: training/validation label artifact) and the oracle legs
(TASK B: evaluation-anchor grids — the OOD-type-1 leg and the F7 tables).

TASK A — generate_train_val: the hypercube label artifact at production size,
built by calling make_labels.generate_labels on parameter-point chunks of
<= 32. The full Latin hypercube is sampled ONCE up front and sliced into the
chunks, so chunking never changes the sample; the shared (S, K, tau) grid and
the global MC subset use the same [seed, 1] / [seed, 2] streams a single full
generate_labels call would use. Parts are resumable: a part is skipped when
its on-disk manifest's identity fields match what would be generated AND its
stored manifest_sha256 verifies against a recompute (skip-by-manifest-hash).
Parts merge into train_val_labels.npz with the labels.npz key layout plus
"split" (shape (n_param_points,), 0=train, 1=val).

SPLIT RULE (leakage-safe): the split is BY PARAMETER POINT, never by
(S, K, tau) row — all n_skt rows of one hypercube parameter point share one
split, drawn from np.random.default_rng([seed, 21]). Rationale: the rows of
one parameter point are correlated through that point's price surface, so
row-wise splitting leaks surface information from train into val.

TASK B — generate_anchor_grids: for each named eval regime, the FULL contract
grid (S x K x tau) through oracle_greeks-equivalent logic, chunked BY TAU
SLICE and resumable (per-part sidecar manifests). Legs: CF + FD on every
point; MC on a seeded STRATIFIED row subset — stratification DECLARED as: per
tau slice, ceil(mc_subset_frac * nS*nK) (S, K) cells drawn without
replacement, sequentially in tau order, from
np.random.default_rng([seed, 3, regime_index]); ADI exactly for the regimes
named by oracle.fourth_leg.required_for (near_feller,
feller_violating_volvol), whose MC path budget is multiplied by
oracle_greeks' near_feller_mc_multiplier. The ADI leg is one whole-grid part
per regime (call homogeneity: one solve serves all S, K; taus are snapshots),
NOT per-slice, so its cost is paid once. Cross-validation note:
cross_validate's wing floor uses max |consensus| within one call, so
MC-subset rows are re-validated in a subset-sized call whose floor can differ
marginally from the full grid's; the subset is stratified across tau and
spans the moneyness range, so the difference is negligible and is declared
here rather than hidden. Outputs per regime: grid-shaped consensus/
uncertainty/mask npz; across regimes: uncertainty_table.csv (mean and p95 of
the uncertainty field + mask rate, per regime x per greek — the F7 error-bar
source and the denominator of the improvement-to-oracle-noise ratio) and
holdout_masks.npz (tau_maturity_holdout = top tau slice;
moneyness_wing_holdout = S/K < 0.75 or > 1.30 — sanity-only per-contract
splits, saved as explicit index masks, i.e. VIEWS on the grid, not
partitions).

FREEZE DISCIPLINE unchanged from make_labels: both entry points refuse any
out_dir containing 'frozen'; promoting a reviewed artifact into data/frozen
is a HUMAN step.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

import make_labels as ml
from make_labels import (LABEL_QUANTITIES, _assert_not_frozen, _git_rev,
                         _sha256_file)
from oracle import (GREEK_NAMES, GreekSet, HestonParams, cross_validate,
                    heston_greeks_adi, heston_greeks_cf, heston_greeks_fd,
                    heston_greeks_mc)
from train_pinn import HESTON_PARAM_NAMES, anchors_from_contract, sample_hypercube_params

N_INFO = 4096                 # N = pinn shared.n_price_points (per-arm price budget)
TRAIN_MIN_ROWS = 5 * N_INFO   # the contract's information-matching cap 5N
VAL_MIN_ROWS = N_INFO
CHUNK_MAX = 32

SPLIT_RULE = (
    "Split is BY PARAMETER POINT, never by (S, K, tau) row: all n_skt rows of one "
    "hypercube parameter point share one split, drawn without replacement from "
    "np.random.default_rng([seed, 21]). Rationale: rows of one parameter point are "
    "correlated through that point's price surface; row-wise splitting leaks.")

# Schema of a {regime}_grid.npz, minus the per-quantity consensus_/uncertainty_/
# mask_ arrays. Asserted at write time so a key cannot be dropped silently; the
# grids are FROZEN artifacts, so `param_names` travelling with `params` (Q6) is
# what lets a consumer bind by name instead of by a shared import's ordering.
_ANCHOR_GRID_KEYS = frozenset({
    "S_axis", "K_axis", "tau_axis", "params", "param_names", "feller_ratio",
    "mc_mask", "mc_flat_idx", "adi_leg", "mask_any", "tol_rel", "seed", "r", "q"})

MC_STRATIFICATION = (
    "MC row subset stratified by tau slice: for each tau slice in tau order, "
    "ceil(mc_subset_frac * nS*nK) (S, K) cells drawn without replacement from "
    "np.random.default_rng([seed, 3, regime_index]); regime_index = position in the "
    "contract's regimes order.")


def _manifest_content_sha(manifest: dict) -> str:
    """Hash over the deterministic manifest fields (make_labels convention:
    everything except manifest_sha256 itself and runtime)."""
    payload = {k: v for k, v in manifest.items() if k not in ("manifest_sha256", "runtime")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _npz_content_sha(arrays: dict) -> str:
    """Content hash of an array dict (name, dtype, shape, raw bytes, sorted keys) —
    stable across runs, unlike the npz file bytes (zip metadata carries mtimes)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _part_complete(part_dir: Path, expected: dict) -> bool:
    """True iff the part's labels.npz + manifest exist, the stored manifest_sha256
    verifies against a recompute, and every expected identity field matches."""
    man_path, npz_path = part_dir / "manifest.json", part_dir / "labels.npz"
    if not (man_path.exists() and npz_path.exists()):
        return False
    try:
        man = json.loads(man_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if man.get("manifest_sha256") != _manifest_content_sha(man):
        return False
    return all(man.get(k) == v for k, v in expected.items())


# ----------------------------------------------------------------------------
# TASK A — chunked + resumable training/validation label artifact
# ----------------------------------------------------------------------------

def generate_train_val(contract_path: str, pinn_cfg_path: str, out_dir: str, seed: int,
                       *, n_param_points: int = 448, n_skt: int = 64,
                       mc_subset_frac: float = 0.10, val_param_frac: float = 0.20,
                       resume: bool = True, chunk_size: int = CHUNK_MAX,
                       leg_kwargs: dict[str, dict] | None = None,
                       min_train_rows: int = TRAIN_MIN_ROWS,
                       min_val_rows: int = VAL_MIN_ROWS) -> dict:
    """Production train/val label artifact via chunked make_labels.generate_labels.

    Samples the full hypercube once, generates parts of <= chunk_size parameter
    points (resumable, skip-by-manifest-hash), merges into train_val_labels.npz
    with a BY-PARAMETER-POINT split (see SPLIT_RULE), enforces the retained-row
    budget (TRAIN >= min_train_rows = 5N, VAL >= min_val_rows), writes the merged
    manifest and runs mask_neutrality_report on the merged artifact.
    min_train_rows/min_val_rows are smoke-test knobs; production uses defaults.
    """
    out = _assert_not_frozen(out_dir)
    assert 1 <= chunk_size <= CHUNK_MAX, f"chunk_size {chunk_size} violates the <= {CHUNK_MAX} rule"
    assert 0.0 < val_param_frac < 1.0, val_param_frac
    t0 = time.perf_counter()
    contract_sha, pinn_sha = _sha256_file(contract_path), _sha256_file(pinn_cfg_path)
    contract = yaml.safe_load(open(contract_path))
    pinn_raw = yaml.safe_load(open(pinn_cfg_path))
    hyc = pinn_raw.get("hypercube_sampling", {})
    samp = contract["training_parameterization"]["sampling"]
    ranges = {k: tuple(map(float, samp["ranges"][k])) for k in HESTON_PARAM_NAMES}
    anchors = anchors_from_contract(contract_path) if "regimes" in contract else None
    # ONE global sample (identical to a single full generate_labels call at this
    # seed); chunks below are slices of it, so chunking never changes the sample.
    params = sample_hypercube_params(
        ranges, n_param_points, seed,
        feller_min=float(hyc.get("feller_min", 0.40)),
        method=samp.get("method", "latin_hypercube"),
        anchors=anchors,
        excise_rel_radius=float(hyc.get("excision", {}).get("rel_radius", 0.10)))

    gc = contract["grid"]
    rng_skt = np.random.default_rng([seed, 1])           # same stream as generate_labels
    S = rng_skt.uniform(gc["S"]["min"], gc["S"]["max"], n_skt)
    K = rng_skt.uniform(gc["K"]["min"], gc["K"]["max"], n_skt)
    tau = rng_skt.uniform(gc["tau"]["min"], gc["tau"]["max"], n_skt)

    n_mc = max(1, int(np.ceil(mc_subset_frac * n_param_points)))
    mc_global = np.sort(np.random.default_rng([seed, 2]).choice(n_param_points, n_mc,
                                                                replace=False))

    n_val = int(round(val_param_frac * n_param_points))
    assert 1 <= n_val < n_param_points, (n_val, n_param_points)
    val_idx = np.random.default_rng([seed, 21]).choice(n_param_points, n_val, replace=False)
    split = np.zeros(n_param_points, dtype=np.int8)
    split[val_idx] = 1

    lk_norm = {name: dict((leg_kwargs or {}).get(name, {})) for name in ("cf", "fd", "mc", "adi")}
    parts_dir = out / "parts"
    offsets = list(range(0, n_param_points, chunk_size))
    part_infos, chunk_seconds, parts_reused = [], {}, []
    for ci, off in enumerate(offsets):
        stop = min(off + chunk_size, n_param_points)
        n_chunk = stop - off
        part_dir = parts_dir / f"part_{ci:04d}"
        local_mc = mc_global[(mc_global >= off) & (mc_global < stop)] - off
        expected = {"contract_sha256": contract_sha, "pinn_cfg_sha256": pinn_sha,
                    "seed": seed, "n_points": n_chunk, "n_skt": n_skt,
                    "leg_kwargs": lk_norm,
                    "mc_subset": [int(i) for i in local_mc],
                    "mc_seed_offset": off,
                    "params_override_sha256": hashlib.sha256(
                        np.ascontiguousarray(params[off:stop]).tobytes()).hexdigest(),
                    "skt_override": True}
        reused = resume and _part_complete(part_dir, expected)
        if not reused:
            tp = time.perf_counter()
            ml.generate_labels(contract_path, pinn_cfg_path, n_chunk, seed, str(part_dir),
                               n_skt=n_skt, mc_subset_frac=mc_subset_frac,
                               leg_kwargs=leg_kwargs, params=params[off:stop],
                               skt=(S, K, tau), mc_subset=local_mc, mc_seed_offset=off)
            chunk_seconds[part_dir.name] = time.perf_counter() - tp
        parts_reused.append(reused)
        man = json.loads((part_dir / "manifest.json").read_text())
        part_infos.append({"part": part_dir.name, "offset": off, "n_points": n_chunk,
                           "manifest_sha256": man["manifest_sha256"]})

    # ---- merge --------------------------------------------------------------
    parts = [np.load(parts_dir / f"part_{ci:04d}" / "labels.npz") for ci in range(len(offsets))]
    first = parts[0]
    for d in parts[1:]:
        assert (np.array_equal(d["S"], first["S"]) and np.array_equal(d["K"], first["K"])
                and np.array_equal(d["tau"], first["tau"])), \
            "parts disagree on the shared (S, K, tau) grid"
    merged = {"params": np.concatenate([d["params"] for d in parts]),
              "param_names": np.array(HESTON_PARAM_NAMES),
              "S": first["S"], "K": first["K"], "tau": first["tau"],
              "r": first["r"], "q": first["q"],
              "feller_ratio": np.concatenate([d["feller_ratio"] for d in parts]),
              "mc_subset": mc_global,
              "adi_points": np.concatenate([d["adi_points"] for d in parts]),
              "mask_any": np.concatenate([d["mask_any"] for d in parts]),
              "seed": np.int64(seed), "tol_rel": first["tol_rel"],
              "split": split}
    for g in LABEL_QUANTITIES:
        for kind in ("consensus", "uncertainty", "mask"):
            merged[f"{kind}_{g}"] = np.concatenate([d[f"{kind}_{g}"] for d in parts])
    assert np.array_equal(merged["params"], params), "merged parts != global sample"

    # ---- budget check (the info-matching cap 5N) ----------------------------
    mask_any = merged["mask_any"]
    retained_train = int((~mask_any[split == 0]).sum())
    retained_val = int((~mask_any[split == 1]).sum())
    mask_rate = float(mask_any.mean())
    ok = retained_train >= min_train_rows and retained_val >= min_val_rows
    print(f"budget check: retained TRAIN rows {retained_train} (need >= {min_train_rows}"
          f" = 5N), retained VAL rows {retained_val} (need >= {min_val_rows});"
          f" mask rate {mask_rate:.3f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        keep = 1.0 - mask_rate
        if keep <= 0.0:
            raise ValueError("budget check failed: every row is masked — no "
                             "n_param_points can satisfy the budget at this mask rate")
        n_needed = int(np.ceil(max(
            min_train_rows / ((1.0 - val_param_frac) * n_skt * keep),
            min_val_rows / (val_param_frac * n_skt * keep))))
        raise ValueError(
            f"budget check failed: retained TRAIN rows {retained_train} < "
            f"{min_train_rows} (5N) or retained VAL rows {retained_val} < "
            f"{min_val_rows} at observed mask rate {mask_rate:.3f}; need "
            f"n_param_points >= {n_needed} (got {n_param_points})")

    npz_path = out / "train_val_labels.npz"
    np.savez(npz_path, **merged)
    manifest = {
        "artifact": "train_val_labels",
        "contract_path": str(contract_path), "contract_sha256": contract_sha,
        "pinn_cfg_path": str(pinn_cfg_path), "pinn_cfg_sha256": pinn_sha,
        "seed": seed, "n_param_points": n_param_points, "n_skt": n_skt,
        "chunk_size": chunk_size, "leg_kwargs": lk_norm,
        "mc_subset_frac": mc_subset_frac, "mc_subset": [int(i) for i in mc_global],
        "val_param_frac": val_param_frac, "split_rule": SPLIT_RULE,
        "n_train_points": int((split == 0).sum()), "n_val_points": int(n_val),
        "budget": {"min_train_rows": int(min_train_rows), "min_val_rows": int(min_val_rows),
                   "retained_train_rows": retained_train,
                   "retained_val_rows": retained_val, "pass": True},
        "mask_rate": {g: float(merged[f"mask_{g}"].mean()) for g in LABEL_QUANTITIES}
                     | {"any": mask_rate},
        "parts": part_infos,
        "merged_sha256": _npz_content_sha(merged),
        "git_rev": _git_rev(),
    }
    manifest["manifest_sha256"] = _manifest_content_sha(manifest)
    manifest["runtime"] = {"wall_clock_s": time.perf_counter() - t0,
                           "chunk_seconds": chunk_seconds,
                           "parts_reused": parts_reused}
    manifest_path = out / "train_val_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    ml.mask_neutrality_report(str(npz_path))   # saves mask_neutrality_report.md alongside
    print("Review mask_neutrality_report.md, then a HUMAN copies this directory to "
          "data/frozen/<tag> and git-tags it.")
    return {"npz_path": str(npz_path), "manifest_path": str(manifest_path),
            "manifest": manifest,
            "report_path": str(out / "mask_neutrality_report.md")}


# ----------------------------------------------------------------------------
# TASK B — evaluation-anchor grids (OOD-type-1 leg + F7 tables)
# ----------------------------------------------------------------------------

def _json_part_complete(json_path: Path, npz_path: Path, ident: dict) -> bool:
    """True iff the sidecar part manifest exists and equals `ident` exactly."""
    if not (json_path.exists() and npz_path.exists()):
        return False
    try:
        return json.loads(json_path.read_text()) == ident
    except (json.JSONDecodeError, OSError):
        return False


def _write_part(npz_path: Path, json_path: Path, arrays: dict, ident: dict) -> None:
    np.savez(npz_path, **arrays)
    json_path.write_text(json.dumps(ident, indent=2))


def generate_anchor_grids(contract_path: str, out_dir: str, seed: int, *,
                          mc_subset_frac: float = 0.10, resume: bool = True,
                          mc_paths: int = 200_000, near_feller_mc_multiplier: int = 4,
                          leg_kwargs: dict[str, dict] | None = None,
                          grid_override: dict | None = None) -> dict:
    """Full-contract-grid oracle consensus for every named regime (see module
    docstring for leg routing, MC stratification and the chunk/resume layout).

    grid_override (smoke tests only; recorded in the manifest) updates the
    contract's per-axis grid spec, e.g. {"S": {"n": 4}}. Production runs the
    contract grid unchanged.
    """
    out = _assert_not_frozen(out_dir)
    t0 = time.perf_counter()
    contract_sha = _sha256_file(contract_path)
    contract = yaml.safe_load(open(contract_path))
    gc = {ax: dict(contract["grid"][ax]) for ax in ("S", "K", "tau")}
    for ax, upd in (grid_override or {}).items():
        gc[ax].update(upd)
    r, q = float(contract["grid"]["r"]), float(contract["grid"]["q"])
    axes = {ax: np.linspace(float(gc[ax]["min"]), float(gc[ax]["max"]), int(gc[ax]["n"]))
            for ax in ("S", "K", "tau")}
    S_ax, K_ax, T_ax = axes["S"], axes["K"], axes["tau"]
    nS, nK, nT = S_ax.size, K_ax.size, T_ax.size
    Sg, Kg, Tg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")
    shape, n_slice_rows = Sg.shape, nS * nK
    tol = float(contract["oracle"]["three_way_validation"]["agreement_tol_rel"])
    required = contract["oracle"]["fourth_leg"]["required_for"]
    regime_order = list(contract["regimes"])
    lk = {name: dict((leg_kwargs or {}).get(name, {})) for name in ("cf", "fd", "mc", "adi")}
    mc_kw = dict(lk["mc"])
    mc_base_paths = int(mc_kw.pop("n_paths", mc_paths))
    n_pick = int(np.ceil(mc_subset_frac * n_slice_rows))
    assert 1 <= n_pick <= n_slice_rows, n_pick

    grid_spec = {ax: {k: float(v) if k != "n" else int(v) for k, v in gc[ax].items()}
                 for ax in ("S", "K", "tau")}
    out.mkdir(parents=True, exist_ok=True)
    table_rows, regime_npz, regimes_meta = [], {}, {}
    parts_runtime: dict[str, float] = {}
    for ridx, name in enumerate(regime_order):
        reg = contract["regimes"][name]
        p = HestonParams(**{k: float(reg[k]) for k in HESTON_PARAM_NAMES})
        # fourth-leg routing mirrors oracle_greeks: required_for is authoritative,
        # the [0.40, 0.60] band clause catches feller-stressed points
        fourth = name in required or 0.40 <= p.feller_ratio <= 0.60
        n_mc_eff = mc_base_paths * (near_feller_mc_multiplier if fourth else 1)

        # declared stratified MC subset (MC_STRATIFICATION)
        rng = np.random.default_rng([seed, 3, ridx])
        mc_mask = np.zeros(shape, dtype=bool)
        mc_rows_per_slice = []
        for j in range(nT):
            rows = np.sort(rng.choice(n_slice_rows, n_pick, replace=False))
            m2 = np.zeros(n_slice_rows, dtype=bool)
            m2[rows] = True
            mc_mask[:, :, j] = m2.reshape(nS, nK)
            mc_rows_per_slice.append(rows)

        part_dir = out / "parts" / name
        part_dir.mkdir(parents=True, exist_ok=True)
        cf = {g: np.empty(shape) for g in GREEK_NAMES}
        fd = {g: np.empty(shape) for g in GREEK_NAMES}
        mc_val = {g: np.full(shape, np.nan) for g in GREEK_NAMES}
        mc_se = {g: np.full(shape, np.nan) for g in GREEK_NAMES}
        for j in range(nT):
            rows = mc_rows_per_slice[j]
            mc_seed = seed + 104729 * ridx + 7919 * j
            ident = {"kind": "anchor_slice", "contract_sha256": contract_sha,
                     "seed": seed, "regime": name, "regime_index": ridx,
                     "params": [float(getattr(p, k)) for k in HESTON_PARAM_NAMES],
                     "tau_index": j, "tau": float(T_ax[j]), "grid": grid_spec,
                     "leg_kwargs": {"cf": lk["cf"], "fd": lk["fd"], "mc": lk["mc"]},
                     "mc_subset_frac": mc_subset_frac,
                     "mc_rows": [int(i) for i in rows],
                     "mc_n_paths": int(n_mc_eff), "mc_seed": int(mc_seed)}
            npz_p = part_dir / f"slice_{j:02d}.npz"
            json_p = part_dir / f"slice_{j:02d}.json"
            if not (resume and _json_part_complete(json_p, npz_p, ident)):
                tp = time.perf_counter()
                S2, K2 = Sg[:, :, j].ravel(), Kg[:, :, j].ravel()
                t2 = np.full(S2.shape, float(T_ax[j]))
                cf_j = heston_greeks_cf(S2, K2, t2, p, r, q, **lk["cf"])
                fd_j = heston_greeks_fd(S2, K2, t2, p, r, q, **lk["fd"])
                mc_j, mc_se_j = heston_greeks_mc(
                    S2[rows], K2[rows], np.full(rows.size, float(T_ax[j])), p, r, q,
                    n_paths=n_mc_eff, seed=mc_seed, **mc_kw)
                arrays = {"mc_rows": rows}
                for g in GREEK_NAMES:
                    arrays[f"cf_{g}"] = getattr(cf_j, g)
                    arrays[f"fd_{g}"] = getattr(fd_j, g)
                    arrays[f"mc_{g}"] = getattr(mc_j, g)
                    arrays[f"mc_se_{g}"] = getattr(mc_se_j, g)
                _write_part(npz_p, json_p, arrays, ident)
                parts_runtime[f"{name}/slice_{j:02d}"] = time.perf_counter() - tp
            d = np.load(npz_p)
            assert np.array_equal(d["mc_rows"], rows)
            for g in GREEK_NAMES:
                cf[g][:, :, j] = d[f"cf_{g}"].reshape(nS, nK)
                fd[g][:, :, j] = d[f"fd_{g}"].reshape(nS, nK)
                mc_val[g][:, :, j].flat[rows] = d[f"mc_{g}"]
                mc_se[g][:, :, j].flat[rows] = d[f"mc_se_{g}"]

        legs = {"cf": GreekSet(**cf), "fd": GreekSet(**fd)}
        if fourth:
            ident = {"kind": "anchor_adi", "contract_sha256": contract_sha,
                     "seed": seed, "regime": name, "regime_index": ridx,
                     "params": [float(getattr(p, k)) for k in HESTON_PARAM_NAMES],
                     "grid": grid_spec, "leg_kwargs": {"adi": lk["adi"]}}
            npz_p, json_p = part_dir / "adi.npz", part_dir / "adi.json"
            if not (resume and _json_part_complete(json_p, npz_p, ident)):
                tp = time.perf_counter()
                adi = heston_greeks_adi(Sg, Kg, Tg, p, r, q, **lk["adi"])
                _write_part(npz_p, json_p,
                            {f"adi_{g}": getattr(adi, g) for g in GREEK_NAMES}, ident)
                parts_runtime[f"{name}/adi"] = time.perf_counter() - tp
            d = np.load(npz_p)
            legs["adi"] = GreekSet(**{g: d[f"adi_{g}"] for g in GREEK_NAMES})

        # cross-validate: full grid without MC, then re-validate the MC subset
        # rows with the MC leg folded in (see wing-floor note, module docstring)
        cons, unc, mask = cross_validate(legs, tol)
        idx = np.flatnonzero(mc_mask.ravel())
        sub = lambda a: np.ascontiguousarray(a).reshape(-1)[idx]
        legs_sub = {nm: GreekSet(**{g: sub(getattr(gs, g)) for g in GREEK_NAMES})
                    for nm, gs in legs.items()}
        legs_sub["mc"] = GreekSet(**{g: sub(mc_val[g]) for g in GREEK_NAMES})
        se_sub = GreekSet(**{g: sub(mc_se[g]) for g in GREEK_NAMES})
        cons_s, unc_s, mask_s = cross_validate(legs_sub, tol, se_sub)
        for g in LABEL_QUANTITIES:
            getattr(cons, g).flat[idx] = getattr(cons_s, g)
            getattr(unc, g).flat[idx] = getattr(unc_s, g)
            mask[g].flat[idx] = mask_s[g]
        mask_any = np.any(np.stack([mask[g] for g in LABEL_QUANTITIES]), axis=0)

        arrays = {"S_axis": S_ax, "K_axis": K_ax, "tau_axis": T_ax,
                  "params": np.array([getattr(p, k) for k in HESTON_PARAM_NAMES]),
                  # Q6: the names travel WITH the vector, exactly as the label
                  # artifact does (make_labels.py:187). A frozen grid outlives the
                  # shared HESTON_PARAM_NAMES import that used to be the only thing
                  # keeping producer and consumer in the same order.
                  "param_names": np.array(HESTON_PARAM_NAMES),
                  "feller_ratio": np.float64(p.feller_ratio),
                  "mc_mask": mc_mask, "mc_flat_idx": idx,
                  "adi_leg": np.bool_(fourth), "mask_any": mask_any,
                  "tol_rel": np.float64(tol), "seed": np.int64(seed),
                  "r": np.float64(r), "q": np.float64(q)}
        assert _ANCHOR_GRID_KEYS <= set(arrays), \
            f"anchor grid missing {sorted(_ANCHOR_GRID_KEYS - set(arrays))}"
        for g in LABEL_QUANTITIES:
            arrays[f"consensus_{g}"] = getattr(cons, g)
            arrays[f"uncertainty_{g}"] = getattr(unc, g)
            arrays[f"mask_{g}"] = mask[g]
        npz_path = out / f"{name}_grid.npz"
        np.savez(npz_path, **arrays)
        regime_npz[name] = str(npz_path)
        for g in LABEL_QUANTITIES:
            u = getattr(unc, g)
            table_rows.append({"regime": name, "greek": g,
                               "uncertainty_mean": float(np.nanmean(u)),
                               "uncertainty_p95": float(np.nanpercentile(u, 95)),
                               "mask_rate": float(mask[g].mean())})
        regimes_meta[name] = {
            "params": {k: float(getattr(p, k)) for k in HESTON_PARAM_NAMES},
            "feller_ratio": float(p.feller_ratio),
            "legs": sorted(legs) + ["mc"], "mc_on_subset_only": True,
            "mc_n_paths_effective": int(n_mc_eff),
            "mc_rows_per_slice": int(n_pick),
            "mask_rate_any": float(mask_any.mean())}

    # ---- F7 uncertainty table (error-bar source / oracle-noise denominator) --
    table_path = out / "uncertainty_table.csv"
    with open(table_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["regime", "greek", "uncertainty_mean",
                                           "uncertainty_p95", "mask_rate"])
        w.writeheader()
        w.writerows(table_rows)

    # ---- holdout definitions: explicit index masks (views on the grid) ------
    tau_hold = np.zeros(shape, dtype=bool)
    tau_hold[:, :, -1] = True                       # top tau slice (tau = 1.0)
    ratio = Sg / Kg
    wing_hold = (ratio < 0.75) | (ratio > 1.30)     # gamma metric: absolute_only
    holdout_path = out / "holdout_masks.npz"
    np.savez(holdout_path, tau_maturity_holdout=tau_hold,
             moneyness_wing_holdout=wing_hold,
             tau_flat_idx=np.flatnonzero(tau_hold.ravel()),
             wing_flat_idx=np.flatnonzero(wing_hold.ravel()))

    manifest = {
        "artifact": "anchor_grids",
        "contract_path": str(contract_path), "contract_sha256": contract_sha,
        "seed": seed, "grid": grid_spec, "grid_override": grid_override,
        "tol_rel": tol, "regime_order": regime_order, "regimes": regimes_meta,
        "mc_subset_frac": mc_subset_frac, "mc_stratification": MC_STRATIFICATION,
        "mc_paths": mc_base_paths,
        "near_feller_mc_multiplier": near_feller_mc_multiplier,
        "adi_regimes": [nm for nm in regime_order if "adi" in regimes_meta[nm]["legs"]],
        "leg_kwargs": lk,
        "holdouts": {
            "tau_maturity_holdout": {
                "definition": "top tau slice (grid index nT-1; tau = tau_max = 1.0 "
                              "on the contract grid); sanity-only per-contract split",
                "n_points": int(tau_hold.sum()),
                "flat_idx": [int(i) for i in np.flatnonzero(tau_hold.ravel())]},
            "moneyness_wing_holdout": {
                "definition": "points with S/K < 0.75 or S/K > 1.30; sanity-only; "
                              "gamma metric absolute_only per contract",
                "n_points": int(wing_hold.sum()),
                "flat_idx": [int(i) for i in np.flatnonzero(wing_hold.ravel())]},
            "note": "masks are VIEWS on the anchor grids (npz: holdout_masks.npz), "
                    "not partitions; flat indices are C-order on (nS, nK, nT)"},
        "uncertainty_table": str(table_path),
        "git_rev": _git_rev(),
    }
    manifest["manifest_sha256"] = _manifest_content_sha(manifest)
    manifest["runtime"] = {"wall_clock_s": time.perf_counter() - t0,
                           "part_seconds": parts_runtime}
    manifest_path = out / "anchors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Anchor grids written. Promoting any artifact into data/frozen stays a "
          "HUMAN step.")
    return {"out_dir": str(out), "manifest_path": str(manifest_path),
            "manifest": manifest, "regime_npz": regime_npz,
            "table_path": str(table_path), "holdout_path": str(holdout_path)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tv = sub.add_parser("train_val", help="TASK A: chunked train/val label artifact")
    tv.add_argument("--contract", default="heston_benchmark_v6.yaml")
    tv.add_argument("--pinn-cfg", default="pinn_config.yaml")
    tv.add_argument("--out-dir", required=True, help="must not contain 'frozen'")
    tv.add_argument("--seed", type=int, default=42)
    tv.add_argument("--n-param-points", type=int, default=448)
    tv.add_argument("--n-skt", type=int, default=64)
    tv.add_argument("--mc-subset-frac", type=float, default=0.10)
    tv.add_argument("--val-param-frac", type=float, default=0.20)
    tv.add_argument("--no-resume", action="store_true")
    an = sub.add_parser("anchors", help="TASK B: evaluation-anchor grids")
    an.add_argument("--contract", default="heston_benchmark_v6.yaml")
    an.add_argument("--out-dir", required=True, help="must not contain 'frozen'")
    an.add_argument("--seed", type=int, default=42)
    an.add_argument("--mc-subset-frac", type=float, default=0.10)
    an.add_argument("--mc-paths", type=int, default=200_000)
    an.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.cmd == "train_val":
        res = generate_train_val(args.contract, args.pinn_cfg, args.out_dir, args.seed,
                                 n_param_points=args.n_param_points, n_skt=args.n_skt,
                                 mc_subset_frac=args.mc_subset_frac,
                                 val_param_frac=args.val_param_frac,
                                 resume=not args.no_resume)
        print(f"wrote {res['npz_path']}\nwrote {res['manifest_path']}")
    else:
        res = generate_anchor_grids(args.contract, args.out_dir, args.seed,
                                    mc_subset_frac=args.mc_subset_frac,
                                    mc_paths=args.mc_paths,
                                    resume=not args.no_resume)
        print(f"wrote {res['manifest_path']}\nwrote {res['table_path']}")
