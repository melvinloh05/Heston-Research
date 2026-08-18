"""R09b — retention per LEG CATEGORY, composed at the production mix.

R09's 40-point probe happened to draw 4 ADI-band points (10% of the probe) where
the production 448-point sample holds 20 (4.46%), and band points mask ~5x harder
than CF+FD-only points. A single small probe therefore biases the mask rate. This
script measures each category separately and composes them with the production
composition, which is EXACTLY known (the band set and the MC subset are both
deterministic functions of the seed).

Categories, with the production weights they carry:
  cf+fd          non-band, not in the MC subset
  cf+fd+mc       non-band, in the MC subset (mc_subset_frac = 0.10)
  adi+cf+fd(+mc) feller ratio in [0.40, 0.60] -> the contract's fourth-leg band

The band category is measured on ALL 20 production band points, so its
contribution is exact rather than estimated. The other two are measured on random
subsets of the production points that fall in them.

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
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/r09b")

SEED = 42
N_PARAM_POINTS = 448
N_SKT = 64
MC_SUBSET_FRAC = 0.10
VAL_PARAM_FRAC = 0.20
N_PLAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 160   # cf+fd probe size
N_MCPTS = int(sys.argv[3]) if len(sys.argv) > 3 else 24    # cf+fd+mc probe size


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


contract = yaml.safe_load(open(CONTRACT))
pinn_raw = yaml.safe_load(open(PINN_CFG))
hyc = pinn_raw.get("hypercube_sampling", {})
samp = contract["training_parameterization"]["sampling"]
ranges = {k: tuple(map(float, samp["ranges"][k])) for k in HESTON_PARAM_NAMES}
anchors = anchors_from_contract(CONTRACT) if "regimes" in contract else None
params = sample_hypercube_params(
    ranges, N_PARAM_POINTS, SEED, feller_min=float(hyc.get("feller_min", 0.40)),
    method=samp.get("method", "latin_hypercube"), anchors=anchors,
    excise_rel_radius=float(hyc.get("excision", {}).get("rel_radius", 0.10)))

gc = contract["grid"]
rng_skt = np.random.default_rng([SEED, 1])
S = rng_skt.uniform(gc["S"]["min"], gc["S"]["max"], N_SKT)
K = rng_skt.uniform(gc["K"]["min"], gc["K"]["max"], N_SKT)
tau = rng_skt.uniform(gc["tau"]["min"], gc["tau"]["max"], N_SKT)

feller = 2.0 * params[:, 0] * params[:, 1] / params[:, 2] ** 2
band = (feller >= 0.40) & (feller <= 0.60)
n_mc = max(1, int(np.ceil(MC_SUBSET_FRAC * N_PARAM_POINTS)))
mc_global = np.sort(np.random.default_rng([SEED, 2]).choice(N_PARAM_POINTS, n_mc,
                                                            replace=False))
is_mc = np.zeros(N_PARAM_POINTS, dtype=bool)
is_mc[mc_global] = True

cat = np.where(band, "band", np.where(is_mc, "cf+fd+mc", "cf+fd"))
weights = {c: float((cat == c).mean()) for c in ("cf+fd", "cf+fd+mc", "band")}

rule("(0) production composition at n_param_points=448, seed=42 (EXACT)")
for c in ("cf+fd", "cf+fd+mc", "band"):
    print(f"  {c:<10s} {int((cat == c).sum()):>4d} points   weight {weights[c]:.4f}")
print(f"  (band = feller ratio in [0.40, 0.60]; MC subset = "
      f"ceil(0.10*448) = {n_mc} points, rng [seed, 2])")


def measure(name: str, idx: np.ndarray, mc_local: np.ndarray, out_sub: str) -> dict:
    """Run generate_labels UNCHANGED on these production points; return mask stats."""
    t0 = time.perf_counter()
    res = ml.generate_labels(CONTRACT, PINN_CFG, idx.size, SEED, str(OUT / out_sub),
                             n_skt=N_SKT,
                             params=np.ascontiguousarray(params[idx]),
                             skt=(S, K, tau), mc_subset=mc_local, mc_seed_offset=0)
    d = np.load(res["npz_path"], allow_pickle=False)
    m = d["mask_any"]
    per_point = m.mean(axis=1)
    stat = {"n_points": int(idx.size), "n_rows": int(m.size),
            "mask_rate": float(m.mean()), "retention": float(1 - m.mean()),
            "per_point_mask_mean": float(per_point.mean()),
            "per_point_mask_sd": float(per_point.std(ddof=1)) if idx.size > 1 else float("nan"),
            "per_greek": {g: float(d[f"mask_{g}"].mean()) for g in LABEL_QUANTITIES},
            "legs_seen": sorted({"+".join(l) for l in res["manifest"]["legs_per_point"]}),
            "seconds": time.perf_counter() - t0}
    se = stat["per_point_mask_sd"] / np.sqrt(idx.size)
    stat["mask_rate_se_clustered"] = float(se)
    print(f"\n  [{name}] {idx.size} points, {m.size} rows, {stat['seconds']:.0f} s")
    print(f"    legs seen        : {stat['legs_seen']}")
    print(f"    mask rate        : {stat['mask_rate']:.4f}  "
          f"+/- {se:.4f} (SE clustered by parameter point)")
    print(f"    retention        : {stat['retention']:.4f}")
    print("    per greek        : " + ", ".join(f"{g} {v:.4f}"
                                                for g, v in stat["per_greek"].items()))
    return stat


rule("(1) per-category measurement (production leg kwargs, unchanged code path)")
rng = np.random.default_rng([SEED, 911])

plain_pool = np.flatnonzero(cat == "cf+fd")
plain_idx = np.sort(rng.choice(plain_pool, min(N_PLAIN, plain_pool.size), replace=False))
st_plain = measure("cf+fd", plain_idx, np.array([], dtype=np.int64), "plain")

mc_pool = np.flatnonzero(cat == "cf+fd+mc")
mc_idx = np.sort(rng.choice(mc_pool, min(N_MCPTS, mc_pool.size), replace=False))
st_mc = measure("cf+fd+mc", mc_idx, np.arange(mc_idx.size), "mc")

band_idx = np.flatnonzero(cat == "band")          # ALL of them: exact, not sampled
band_mc_local = np.flatnonzero(is_mc[band_idx])
st_band = measure("band (ALL production band points)", band_idx, band_mc_local, "band")

rule("(2) composed mask rate at the production mix")
comp = (weights["cf+fd"] * st_plain["mask_rate"]
        + weights["cf+fd+mc"] * st_mc["mask_rate"]
        + weights["band"] * st_band["mask_rate"])
keep = 1.0 - comp
var = ((weights["cf+fd"] * st_plain["mask_rate_se_clustered"]) ** 2
       + (weights["cf+fd+mc"] * st_mc["mask_rate_se_clustered"]) ** 2)
se_comp = float(np.sqrt(var))       # band term contributes no sampling error (census)
for c, st in (("cf+fd", st_plain), ("cf+fd+mc", st_mc), ("band", st_band)):
    print(f"  {c:<10s} weight {weights[c]:.4f} x mask {st['mask_rate']:.4f} "
          f"= {weights[c] * st['mask_rate']:.5f}")
print(f"  composed mask rate  : {comp:.4f} +/- {se_comp:.4f}")
print(f"  composed RETENTION  : {keep:.4f}  (95% lower bound "
      f"{keep - 1.96 * se_comp:.4f})")

rule("(3) sizing")
N = int(pinn_raw["shared"]["n_price_points"])
need_train, need_val = 5 * N, N
print(f"  N = {N}; need train >= 5N = {need_train}, val >= N = {need_val}; "
      f"n_skt = {N_SKT}; val_param_frac = {VAL_PARAM_FRAC}")


def sizing(k: float, label: str) -> None:
    nt = need_train / ((1 - VAL_PARAM_FRAC) * N_SKT * k)
    nv = need_val / (VAL_PARAM_FRAC * N_SKT * k)
    n_min = int(np.ceil(max(nt, nv)))
    print(f"\n  {label} (keep = {k:.4f})")
    print(f"    binding constraint: {'TRAIN' if nt >= nv else 'VAL'} "
          f"(train {nt:.1f}, val {nv:.1f})")
    print(f"    minimum n_param_points = {n_min}")
    for margin in (0.10, 0.15, 0.20, 0.25, 0.30):
        n = int(np.ceil(n_min * (1 + margin)))
        n32 = int(np.ceil(n / 32) * 32)
        print(f"      +{margin:.0%} -> {n:>4d}  (multiple of 32: {n32:>4d})")


sizing(keep, "at the composed point estimate")
sizing(keep - 1.96 * se_comp, "at the 95% lower retention bound")

print("\n  default n_param_points = 448 at the composed retention:")
tr448 = int(round((1 - VAL_PARAM_FRAC) * 448)) * N_SKT * keep
va448 = int(round(VAL_PARAM_FRAC * 448)) * N_SKT * keep
print(f"    train rows ~ {tr448:.0f} (need {need_train}) -> "
      f"{'PASS' if tr448 >= need_train else 'FAIL'}, "
      f"margin {tr448 - need_train:+.0f} rows ({tr448 / need_train - 1:+.2%})")
print(f"    val   rows ~ {va448:.0f} (need {need_val}) -> "
      f"{'PASS' if va448 >= need_val else 'FAIL'}")

(OUT / "r09b_summary.json").write_text(json.dumps(
    {"weights": weights, "cf+fd": st_plain, "cf+fd+mc": st_mc, "band": st_band,
     "composed_mask_rate": comp, "composed_retention": keep,
     "composed_se": se_comp}, indent=2))
print(f"\nsummary -> {OUT / 'r09b_summary.json'}")
