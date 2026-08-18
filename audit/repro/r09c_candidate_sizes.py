"""R09c — exact composed retention per CANDIDATE n_param_points at the production seed.

R09b measured retention per leg category. The composition is not a constant: the
ADI-band share (feller ratio in [0.40, 0.60]) is a deterministic function of
(n_param_points, seed) through the Latin hypercube, and band points mask ~7.6x
harder than plain CF+FD points, so the band share drives the answer. This script
counts the band and MC members EXACTLY for each candidate size at the production
seed and composes them with R09b's measured per-category rates.

It also reports the band share across other seeds, so the margin covers the case
where the artifact is ever regenerated at a different seed.

Cheap: parameter sampling only, no oracle legs. Writes nothing outside stdout.
"""
from __future__ import annotations

import sys

import numpy as np
import yaml

REPO = "/Users/melvin/Documents/Heston Research"
sys.path.insert(0, REPO)

from train_pinn import (HESTON_PARAM_NAMES, anchors_from_contract,  # noqa: E402
                        sample_hypercube_params)

CONTRACT = f"{REPO}/heston_benchmark_v6.yaml"
PINN_CFG = f"{REPO}/pinn_config.yaml"

# ---- measured in R09b (mask rate per leg category, production leg kwargs) ----
MASK_PLAIN = 0.0685      # cf+fd,     160 points, SE 0.0034
MASK_MC = 0.0749         # cf+fd+mc,   24 points, SE 0.0076
MASK_BAND = 0.5211       # adi+cf+fd,  20 points (census at n=448), SE 0.0522

SEED = 42
N_SKT = 64
MC_SUBSET_FRAC = 0.10
VAL_PARAM_FRAC = 0.20

contract = yaml.safe_load(open(CONTRACT))
pinn_raw = yaml.safe_load(open(PINN_CFG))
hyc = pinn_raw.get("hypercube_sampling", {})
samp = contract["training_parameterization"]["sampling"]
ranges = {k: tuple(map(float, samp["ranges"][k])) for k in HESTON_PARAM_NAMES}
anchors = anchors_from_contract(CONTRACT) if "regimes" in contract else None
N = int(pinn_raw["shared"]["n_price_points"])
NEED_TRAIN, NEED_VAL = 5 * N, N


def compose(n_points: int, seed: int) -> dict:
    p = sample_hypercube_params(
        ranges, n_points, seed, feller_min=float(hyc.get("feller_min", 0.40)),
        method=samp.get("method", "latin_hypercube"), anchors=anchors,
        excise_rel_radius=float(hyc.get("excision", {}).get("rel_radius", 0.10)))
    feller = 2.0 * p[:, 0] * p[:, 1] / p[:, 2] ** 2
    band = (feller >= 0.40) & (feller <= 0.60)
    n_mc = max(1, int(np.ceil(MC_SUBSET_FRAC * n_points)))
    is_mc = np.zeros(n_points, dtype=bool)
    is_mc[np.random.default_rng([seed, 2]).choice(n_points, n_mc, replace=False)] = True
    w_band = band.mean()
    w_mc = (is_mc & ~band).mean()
    w_plain = (~is_mc & ~band).mean()
    mask = w_plain * MASK_PLAIN + w_mc * MASK_MC + w_band * MASK_BAND
    keep = 1.0 - mask

    # SPLIT-AWARE: make_datasets draws the val points from rng [seed, 21], so the
    # band points (which mask 7.6x harder) do NOT distribute evenly between the
    # splits. Compose each split with ITS OWN category counts.
    n_val = int(round(VAL_PARAM_FRAC * n_points))
    n_train = n_points - n_val
    val_idx = np.random.default_rng([seed, 21]).choice(n_points, n_val, replace=False)
    split = np.zeros(n_points, dtype=np.int8)
    split[val_idx] = 1

    def split_rows(want: int) -> tuple[float, int, float]:
        sel = split == want
        nb = int((band & sel).sum())
        nm = int((is_mc & ~band & sel).sum())
        npl = int((~is_mc & ~band & sel).sum())
        n = nb + nm + npl
        k = 1.0 - (npl * MASK_PLAIN + nm * MASK_MC + nb * MASK_BAND) / n
        return n * N_SKT * k, nb, k

    train_rows, n_band_train, keep_train = split_rows(0)
    val_rows, _, keep_val = split_rows(1)
    return {"n_points": n_points, "seed": seed, "n_band": int(band.sum()),
            "w_band": float(w_band), "mask": float(mask), "keep": float(keep),
            "train_rows": train_rows, "val_rows": val_rows,
            "keep_train": keep_train, "keep_val": keep_val,
            "n_band_train": n_band_train,
            "n_train_points": n_train, "n_val_points": n_val,
            "n_mc": n_mc, "n_plain": int((~is_mc & ~band).sum())}


print("=" * 86)
print(f"candidate sizes at the production seed {SEED} "
      f"(n_skt={N_SKT}, val_param_frac={VAL_PARAM_FRAC})")
print(f"need TRAIN >= 5N = {NEED_TRAIN} rows, VAL >= N = {NEED_VAL} rows")
print("=" * 86)
hdr = (f"{'n_pts':>6} {'band':>5} {'bnd_tr':>7} {'keep_tr':>8} {'train_rows':>11} "
       f"{'margin':>9} {'val_rows':>9} {'hours':>7} {'verdict':>8}")
print(hdr)
print("-" * len(hdr))

# measured per-point wall clock (this machine, single core), from R09b:
#   cf+fd 91 s / 160 pts, cf+fd+mc 41 s / 24 pts, adi+cf+fd 461 s / 20 pts
SEC_PLAIN, SEC_MC, SEC_BAND = 91 / 160, 41 / 24, 461 / 20

for n in (384, 416, 448, 480, 512, 544, 576, 608, 640):
    r = compose(n, SEED)
    marg = r["train_rows"] / NEED_TRAIN - 1
    ok = r["train_rows"] >= NEED_TRAIN and r["val_rows"] >= NEED_VAL
    hours = (r["n_plain"] * SEC_PLAIN + (r["n_mc"] - r["n_band"] * 0) * SEC_MC
             + r["n_band"] * SEC_BAND) / 3600
    print(f"{n:>6} {r['n_band']:>5} {r['n_band_train']:>7} {r['keep_train']:>8.4f} "
          f"{r['train_rows']:>11.0f} {marg:>+8.1%} {r['val_rows']:>9.0f} "
          f"{hours:>7.2f} {'PASS' if ok else 'FAIL':>8}")
print("  band = ADI-band points total; bnd_tr = how many landed in the TRAIN split")
print("  (rng [seed, 21]); keep_tr = retention of the TRAIN split specifically.")
print("  hours = measured single-core wall clock on this machine (CF+FD 0.57 s/pt,")
print("  +MC 1.71 s/pt, +ADI 23.05 s/pt); chunked + resumable, so it can be split.")

print("\n" + "=" * 86)
print("band share across seeds at n_param_points = 448 and 512")
print("(the artifact is generated at ONE seed, but the margin should survive a reseed)")
print("=" * 86)
for n in (448, 512):
    shares, keeps, trains = [], [], []
    for s in range(42, 52):
        r = compose(n, s)
        shares.append(r["w_band"]); keeps.append(r["keep"]); trains.append(r["train_rows"])
    print(f"  n={n}: band share min {min(shares):.4f} / max {max(shares):.4f}; "
          f"keep min {min(keeps):.4f}; train rows min {min(trains):.0f} "
          f"(need {NEED_TRAIN}) -> worst-case "
          f"{'PASS' if min(trains) >= NEED_TRAIN else 'FAIL'}")

print("\n" + "=" * 86)
print("sensitivity: train rows vs band share, at n = 448 / 512 / 544")
print("=" * 86)
w_mc_nom = 0.10
print(f"{'w_band':>8} {'keep':>8} " + " ".join(f"{n:>10}" for n in (448, 512, 544)))
for wb in (0.02, 0.045, 0.06, 0.08, 0.10, 0.12, 0.15):
    w_mc = (1 - wb) * w_mc_nom
    w_pl = (1 - wb) * (1 - w_mc_nom)
    keep = 1.0 - (w_pl * MASK_PLAIN + w_mc * MASK_MC + wb * MASK_BAND)
    cells = []
    for n in (448, 512, 544):
        n_val = int(round(VAL_PARAM_FRAC * n))
        rows = (n - n_val) * N_SKT * keep
        cells.append(f"{rows:>10.0f}" + ("*" if rows < NEED_TRAIN else " "))
    print(f"{wb:>8.3f} {keep:>8.4f} " + " ".join(cells))
print(f"  (* = below the 5N = {NEED_TRAIN} requirement)")

print("\n" + "=" * 86)
print("disk footprint of train_val_labels.npz (uncompressed np.savez)")
print("=" * 86)
for n in (448, 512, 544):
    per_pt_arrays = 15 * 8 * n * N_SKT          # 5 quantities x (consensus, unc) f8 + mask b1
    cons = 5 * n * N_SKT * 8
    unc = 5 * n * N_SKT * 8
    msk = 5 * n * N_SKT * 1
    mask_any = n * N_SKT * 1
    params = n * 5 * 8
    feller = n * 8
    adi_pts = n * 1
    split = n * 1
    skt = 3 * N_SKT * 8
    total = cons + unc + msk + mask_any + params + feller + adi_pts + split + skt
    print(f"  n={n:>4}: rows {n * N_SKT:>7}  consensus {cons / 1e6:>6.2f} MB  "
          f"uncertainty {unc / 1e6:>6.2f} MB  masks {(msk + mask_any) / 1e6:>5.2f} MB  "
          f"TOTAL {total / 1e6:>6.2f} MB")
print("  (float64 for consensus_/uncertainty_, bool for mask_; parts/ adds ~1x more "
       "before it can be deleted)")
