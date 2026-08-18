"""R13 — mask selectivity on the TRAIN/VAL hypercube, where Feller ratio VARIES.

The anchor grids hold one parameter vector per regime, so Feller ratio is constant
within a grid and the "is the mask selective on Feller ratio" question can only be
asked on the hypercube label artifact. This probe runs make_labels.generate_labels
UNCHANGED on production parameter points (seed 42, n_param_points=448 sample,
n_skt=64, production leg kwargs) and reports:

  - per-parameter-point mask rate vs Feller ratio (Spearman + binned)
  - masked vs surviving row distributions: moneyness, tau, |consensus_g|
  - mask rate by |consensus_gamma| / |consensus_vega| decile
  - the contract's own three neutrality checks via make_labels.mask_neutrality_report

Composition note: ALL 20 production band points (Feller in [0.40, 0.60]) are
censused and combined with a random sample of non-band points, so band points are
over-represented relative to the 4.46% production weight. Per-point statistics are
therefore reported against Feller ratio directly; any pooled rate is re-weighted to
the production mix and labelled as such.

Usage: python audit/repro/r13_hypercube_mask_selectivity.py <scratch_dir> [n_plain]
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)

import make_labels as ml  # noqa: E402
from make_labels import LABEL_QUANTITIES  # noqa: E402
from train_pinn import (HESTON_PARAM_NAMES, anchors_from_contract,  # noqa: E402
                        sample_hypercube_params)

CONTRACT = f"{REPO}/heston_benchmark_v6.yaml"
PINN_CFG = f"{REPO}/pinn_config.yaml"
SEED, N_PARAM_POINTS, N_SKT, MC_SUBSET_FRAC = 42, 448, 64, 0.10


def main(out_dir: str, n_plain: int = 128) -> None:
    out = Path(out_dir)
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
    is_mc = np.zeros(N_PARAM_POINTS, bool)
    is_mc[np.random.default_rng([SEED, 2]).choice(N_PARAM_POINTS, n_mc, replace=False)] = True

    rng = np.random.default_rng([SEED, 913])
    plain_pool = np.flatnonzero(~band)
    idx = np.sort(np.concatenate([
        rng.choice(plain_pool, min(n_plain, plain_pool.size), replace=False),
        np.flatnonzero(band)]))
    mc_local = np.flatnonzero(is_mc[idx])
    print(f"probe: {idx.size} production parameter points "
          f"({int(band[idx].sum())} band = ALL production band points, "
          f"{int(is_mc[idx].sum())} in the MC subset); production band weight "
          f"{band.mean():.4f}")

    t0 = time.perf_counter()
    res = ml.generate_labels(CONTRACT, PINN_CFG, idx.size, SEED, str(out / "probe"),
                             n_skt=N_SKT, params=np.ascontiguousarray(params[idx]),
                             skt=(S, K, tau), mc_subset=mc_local, mc_seed_offset=0)
    print(f"generate_labels wall clock {time.perf_counter() - t0:.0f}s")
    d = np.load(res["npz_path"], allow_pickle=False)
    m = np.asarray(d["mask_any"], bool)                       # (n_points, n_skt)
    fr = np.asarray(d["feller_ratio"], float)
    point_rate = m.mean(axis=1)

    print("\n--- (1) per-point mask rate vs Feller ratio ---")
    from scipy.stats import spearmanr
    rho, p = spearmanr(fr, point_rate)
    print(f"  Spearman rho(feller, point mask rate) = {rho:+.4f}  (p={p:.2e}, "
          f"n={fr.size} points)")
    edges = [0.40, 0.60, 1.0, 2.0, 4.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (fr >= lo) & (fr < hi)
        if sel.any():
            print(f"  feller [{lo:.2f}, {hi if np.isfinite(hi) else 'inf'}): "
                  f"n={int(sel.sum()):>4d}  mask rate {point_rate[sel].mean():.4f}")

    print("\n--- (2) masked vs surviving ROW distributions ---")
    Sf = np.tile(S, (idx.size, 1)).ravel()
    Kf = np.tile(K, (idx.size, 1)).ravel()
    Tf = np.tile(tau, (idx.size, 1)).ravel()
    frf = np.repeat(fr, N_SKT)
    mf = m.ravel()
    print(f"  {'quantity':>22s} {'masked q25/q50/q75':>32s} {'surviving q25/q50/q75':>32s}")
    for lbl, v in (("feller ratio", frf), ("moneyness S/K", Sf / Kf), ("tau", Tf)):
        a = np.percentile(v[mf], [25, 50, 75])
        b = np.percentile(v[~mf], [25, 50, 75])
        print(f"  {lbl:>22s} {a[0]:10.4f}{a[1]:11.4f}{a[2]:11.4f}   "
              f"{b[0]:10.4f}{b[1]:11.4f}{b[2]:11.4f}")
    for g in LABEL_QUANTITIES:
        c = np.abs(np.asarray(d[f"consensus_{g}"], float).ravel())
        a = np.percentile(c[mf], [25, 50, 75])
        b = np.percentile(c[~mf], [25, 50, 75])
        print(f"  {'|consensus_' + g + '|':>22s} {a[0]:10.4g}{a[1]:11.4g}{a[2]:11.4g}   "
              f"{b[0]:10.4g}{b[1]:11.4g}{b[2]:11.4g}")

    print("\n--- (3) mask rate by |consensus| decile ---")
    for g in ("gamma", "vega"):
        c = np.abs(np.asarray(d[f"consensus_{g}"], float).ravel())
        order = np.argsort(c)
        print(f"  {g:>6s}: " + " ".join(f"{k+1}:{mf[ch].mean():.3f}"
                                        for k, ch in enumerate(np.array_split(order, 10))))

    print("\n--- (4) the contract's own neutrality report on this probe ---")
    md, stats = ml.mask_neutrality_report(res["npz_path"], str(out / "r13_neutrality.md"))
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/r13",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
