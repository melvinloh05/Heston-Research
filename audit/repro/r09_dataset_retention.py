"""R09 — measured row retention from label generation to ArmDataset(..., "train").

Sizing input for audit I1: the info-matching ladder asks for 5N = 20480 training
rows, and subsample_train silently caps at the frozen train split. Nothing in the
repo records what fraction of generated rows survives the oracle mask, so it is
measured here on the REAL production code path and extrapolated.

What this does
--------------
1. Draws the production hypercube (n_param_points, seed) with the production
   Feller floor + anchor excision, and the production shared (S, K, tau) grid
   from the same np.random.default_rng([seed, 1]) stream make_datasets uses.
2. Takes a random PROBE subset of those parameter points and runs
   make_labels.generate_labels on them UNCHANGED (production leg kwargs: CF + FD
   everywhere, MC 200k on the declared subset, ADI on Feller ratio in
   [0.40, 0.60]) — so the measured mask is the mask production would produce.
3. Reports mask rate overall, per label quantity, and split by which legs ran,
   plus the ADI-band fraction over the FULL parameter sample.
4. Walks the retention chain end to end on the probe artifact:
   generated rows -> mask -> by-parameter-point train/val split -> ArmDataset.

Writes nothing outside the scratchpad. Does not touch data/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = "/Users/melvin/Documents/Heston Research"
sys.path.insert(0, REPO)

import make_labels as ml  # noqa: E402
from make_labels import LABEL_QUANTITIES  # noqa: E402
from train_pinn import (HESTON_PARAM_NAMES, anchors_from_contract,  # noqa: E402
                        sample_hypercube_params)

CONTRACT = f"{REPO}/heston_benchmark_v6.yaml"
PINN_CFG = f"{REPO}/pinn_config.yaml"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/r09_probe")

SEED = 42
N_PARAM_POINTS = 448      # make_datasets.generate_train_val default
N_SKT = 64                # make_datasets.generate_train_val default
MC_SUBSET_FRAC = 0.10
VAL_PARAM_FRAC = 0.20
N_PROBE = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# (1) the production sample, verbatim from make_datasets.generate_train_val
# ---------------------------------------------------------------------------
contract = yaml.safe_load(open(CONTRACT))
pinn_raw = yaml.safe_load(open(PINN_CFG))
hyc = pinn_raw.get("hypercube_sampling", {})
samp = contract["training_parameterization"]["sampling"]
ranges = {k: tuple(map(float, samp["ranges"][k])) for k in HESTON_PARAM_NAMES}
anchors = anchors_from_contract(CONTRACT) if "regimes" in contract else None

params_full = sample_hypercube_params(
    ranges, N_PARAM_POINTS, SEED,
    feller_min=float(hyc.get("feller_min", 0.40)),
    method=samp.get("method", "latin_hypercube"),
    anchors=anchors,
    excise_rel_radius=float(hyc.get("excision", {}).get("rel_radius", 0.10)))

gc = contract["grid"]
rng_skt = np.random.default_rng([SEED, 1])
S = rng_skt.uniform(gc["S"]["min"], gc["S"]["max"], N_SKT)
K = rng_skt.uniform(gc["K"]["min"], gc["K"]["max"], N_SKT)
tau = rng_skt.uniform(gc["tau"]["min"], gc["tau"]["max"], N_SKT)

feller_full = 2.0 * params_full[:, 0] * params_full[:, 1] / params_full[:, 2] ** 2
band_full = (feller_full >= 0.40) & (feller_full <= 0.60)

rule("(1) production parameter sample")
print(f"  sample_hypercube_params(n={N_PARAM_POINTS}, seed={SEED}) delivers "
      f"EXACTLY {params_full.shape[0]} points")
print(f"  -> Feller rejection + anchor excision cost ZERO delivered points "
      f"(the sampler resamples until n are accepted)")
print(f"  ADI-band points (feller in [0.40, 0.60]): {int(band_full.sum())} / "
      f"{N_PARAM_POINTS} = {band_full.mean():.4f}")
print(f"  shared (S, K, tau) grid: {N_SKT} triples, drawn once, shared by every "
      f"parameter point -> rows per parameter point = n_skt = {N_SKT}")

# ---------------------------------------------------------------------------
# (2) probe subset through the UNCHANGED generate_labels path
# ---------------------------------------------------------------------------
probe_rng = np.random.default_rng([SEED, 909])
probe_idx = np.sort(probe_rng.choice(N_PARAM_POINTS, N_PROBE, replace=False))
probe_params = np.ascontiguousarray(params_full[probe_idx])
n_mc = max(1, int(np.ceil(MC_SUBSET_FRAC * N_PROBE)))
mc_local = np.sort(probe_rng.choice(N_PROBE, n_mc, replace=False))

rule(f"(2) generate_labels on a {N_PROBE}-point probe (production leg kwargs)")
feller_probe = 2.0 * probe_params[:, 0] * probe_params[:, 1] / probe_params[:, 2] ** 2
band_probe = (feller_probe >= 0.40) & (feller_probe <= 0.60)
print(f"  probe points {N_PROBE}; MC points {n_mc}; ADI-band points "
      f"{int(band_probe.sum())}")
print("  running (CF + FD every point, MC 200k paths on the subset, ADI on band "
      "points) ...")
t0 = time.perf_counter()
res = ml.generate_labels(CONTRACT, PINN_CFG, N_PROBE, SEED, str(OUT),
                         n_skt=N_SKT, params=probe_params, skt=(S, K, tau),
                         mc_subset=mc_local, mc_seed_offset=0)
print(f"  done in {time.perf_counter() - t0:.1f} s -> {res['npz_path']}")

d = np.load(res["npz_path"], allow_pickle=False)
mask_any = d["mask_any"]
n_pts, n_skt = mask_any.shape
legs_per_point = res["manifest"]["legs_per_point"]

rule("(3) measured mask rate")
print(f"  rows generated: {n_pts} x {n_skt} = {n_pts * n_skt}")
print(f"  mask_any rate      : {mask_any.mean():.4f}   "
      f"-> retention {1 - mask_any.mean():.4f}")
for g in LABEL_QUANTITIES:
    print(f"    mask_{g:<6s}       : {d[f'mask_{g}'].mean():.4f}")

print("\n  by leg set (rows):")
by_legs: dict[str, list[int]] = {}
for i, legs in enumerate(legs_per_point):
    by_legs.setdefault("+".join(legs), []).append(i)
for key in sorted(by_legs):
    rows = mask_any[by_legs[key]]
    print(f"    {key:<16s} n_points {len(by_legs[key]):>3d}  rows {rows.size:>5d}  "
          f"mask {rows.mean():.4f}  retention {1 - rows.mean():.4f}")

# ---------------------------------------------------------------------------
# (4) the retention chain, walked on the probe artifact
# ---------------------------------------------------------------------------
rule("(4) retention chain: generated -> mask -> split -> ArmDataset")
n_val = int(round(VAL_PARAM_FRAC * n_pts))
val_idx = np.random.default_rng([SEED, 21]).choice(n_pts, n_val, replace=False)
split = np.zeros(n_pts, dtype=np.int8)
split[val_idx] = 1
gen_rows = n_pts * n_skt
kept_rows = int((~mask_any).sum())
train_rows = int((~mask_any[split == 0]).sum())
val_rows = int((~mask_any[split == 1]).sum())
print(f"  generated rows                       : {gen_rows}")
print(f"  after oracle mask (~mask_any)        : {kept_rows}"
      f"   ({kept_rows / gen_rows:.4f})")
print(f"  train split (split==0, {int((split == 0).sum())} points) : {train_rows}"
      f"   ({train_rows / gen_rows:.4f} of generated)")
print(f"  val   split (split==1, {n_val} points) : {val_rows}"
      f"   ({val_rows / gen_rows:.4f} of generated)")

# the same npz, re-read through the real consumer
sys.path.insert(0, REPO)
from SobolevPINN import load_arm  # noqa: E402
from train_pinn import ArmDataset  # noqa: E402

with np.load(res["npz_path"], allow_pickle=False) as z:
    arrays = {k: z[k] for k in z.files}
arrays["split"] = split
split_npz = OUT / "labels_with_split.npz"
np.savez(split_npz, **arrays)

cfg = load_arm(PINN_CFG, "info_matched_baseline")
tr = ArmDataset(str(split_npz), cfg, "train", seed=SEED)
va = ArmDataset(str(split_npz), cfg, "val", seed=SEED)
print(f"\n  ArmDataset(..., 'train').n_rows      : {tr.n_rows}"
      f"   (chain predicted {train_rows})")
print(f"  ArmDataset(..., 'val').n_rows        : {va.n_rows}"
      f"   (chain predicted {val_rows})")
print(f"  extra filters between build_arm_labels and ArmDataset: "
      f"{'NONE' if (tr.n_rows == train_rows and va.n_rows == val_rows) else 'SOME'}")

# ---------------------------------------------------------------------------
# (5) extrapolation to the production sizing
# ---------------------------------------------------------------------------
rule("(5) extrapolation")
keep = 1.0 - float(mask_any.mean())
N = int(pinn_raw["shared"]["n_price_points"])
need_train, need_val = 5 * N, N
print(f"  N = shared.n_price_points = {N}; need train >= 5N = {need_train}, "
      f"val >= N = {need_val}")
print(f"  measured retention keep = {keep:.4f}")
for nskt in (64,):
    n_train_needed = need_train / ((1 - VAL_PARAM_FRAC) * nskt * keep)
    n_val_needed = need_val / (VAL_PARAM_FRAC * nskt * keep)
    n_min = int(np.ceil(max(n_train_needed, n_val_needed)))
    print(f"  n_skt={nskt}: n_param_points >= max({n_train_needed:.1f} [train], "
          f"{n_val_needed:.1f} [val]) = {n_min}")
    for margin in (0.10, 0.15, 0.20, 0.25):
        print(f"      +{margin:.0%} margin -> {int(np.ceil(n_min * (1 + margin)))}"
              f"  (round up to a multiple of 32: "
              f"{int(np.ceil(n_min * (1 + margin) / 32) * 32)})")

print("\n  default n_param_points=448 at this retention:")
print(f"    train rows ~ {int(448 * (1 - VAL_PARAM_FRAC)) * 64 * keep:.0f} "
      f"(need {need_train}) -> "
      f"{'PASS' if int(448 * 0.8) * 64 * keep >= need_train else 'FAIL'}")

summary = {"n_param_points_full": N_PARAM_POINTS, "n_skt": N_SKT,
           "adi_band_frac_full_sample": float(band_full.mean()),
           "probe_points": N_PROBE, "probe_rows": gen_rows,
           "mask_any_rate": float(mask_any.mean()), "retention": keep,
           "mask_rate_per_quantity": {g: float(d[f"mask_{g}"].mean())
                                      for g in LABEL_QUANTITIES},
           "arm_dataset_train_rows": tr.n_rows, "arm_dataset_val_rows": va.n_rows}
(OUT / "r09_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nsummary -> {OUT / 'r09_summary.json'}")
