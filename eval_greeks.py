"""OOD-parameter Greek evaluation of trained arms against the P8 anchor grids.

This is the HELD-OUT Greek-RMSE metric layer (contract PRIMARY metrics: OOD-param
Greek RMSE + tails on near_feller / strong_neg_corr for Delta, Gamma, Vega, vanna).
It scores each trained arm on the frozen evaluation-anchor grids produced by
make_datasets.generate_anchor_grids ({regime}_grid.npz: grid-shaped consensus /
uncertainty / mask fields on the full (S, K, tau) contract grid).

EVALUATION-POINT CONVENTION — READ THIS (it differs from the hedging provider):
    Here the REGIME *is* the evaluation point. The network is queried on the full
    (S, K, tau) grid with the regime's OWN (kappa, theta, xi, rho, v0) written into
    the parameter input columns — we are asking "how well did the arm learn the price
    surface AT this OOD parameter vector". This is the OPPOSITE of PINNProvider.evaluate
    (pinn_provider.py), which FREEZES the parameter columns at theta_train (the baseline)
    because there the arm is hedging a path under a single fixed model. Do not confuse the
    two: hedging pins theta_train and varies only (S, v0); Greek eval pins the regime and
    sweeps the whole grid. We therefore reach past PINNProvider.evaluate straight to the
    underlying model (greeks_eval), injecting the regime params ourselves.

WING RULE (contract): on the moneyness_wing_holdout slice (S/K < 0.75 or > 1.30) the
GAMMA metric is ABSOLUTE ONLY — deep ITM/OTM Gamma is ~0 and a relative error there is
meaningless. rel_rmse for Gamma is computed on the body (non-wing) points only and is
left BLANK (NaN) on a pure-wing slice; the absolute rmse/quantiles include every point.

F7 reporting discipline: primary metrics on near_feller + strong_neg_corr; sanity on
baseline + high_variance + feller_violating_volvol (the boundary regime is kept WITH
error bars, never dropped). strong_neg_corr is genuine rho extrapolation — larger seed
variance is PRE-REGISTERED; it is reported (seed-std), not suppressed. Threshold verdicts
live in P13; this module only pre-checks and prints.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from Hedging_backtest import contract_thresholds
from pinn_provider import build_providers
from train_pinn import HESTON_PARAM_NAMES

GREEKS = ("price", "delta", "gamma", "vega", "vanna")
WING_LO, WING_HI = 0.75, 1.30          # moneyness_wing_holdout: S/K outside [0.75, 1.30]
BASELINE_ARM = "standard_pinn"         # comparison reference (Hainaut & Casas PDE baseline)


# ---------------------------------------------------------------------------
# small numeric helpers (nan-aware, empty-safe)
# ---------------------------------------------------------------------------

def _rms(a) -> float:
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def reduction_vs_baseline(rmse_arm: float, rmse_base: float) -> float:
    """1 - rmse_arm / rmse_base (fraction of the baseline error removed)."""
    return (1.0 - rmse_arm / rmse_base
            if np.isfinite(rmse_base) and rmse_base > 0 else float("nan"))


def improvement_to_oracle_noise(rmse_arm: float, rmse_base: float,
                                oracle_unc_rms: float) -> float:
    """(rmse_base - rmse_arm) / oracle_unc_rms — error reduction in oracle-noise units."""
    return ((rmse_base - rmse_arm) / oracle_unc_rms
            if np.isfinite(oracle_unc_rms) and oracle_unc_rms > 0 else float("nan"))


def price_parity(rel_price_arm: float, rel_price_base: float) -> float:
    """rel_price_rmse_arm / rel_price_rmse_base - 1 (0 = matched price accuracy)."""
    return (rel_price_arm / rel_price_base - 1.0
            if np.isfinite(rel_price_base) and rel_price_base > 0 else float("nan"))


def aggregate_seeds(values) -> tuple[float, float]:
    """(mean, sample-std ddof=1) over the finite entries; std=0 for a single seed,
    (nan, nan) when nothing is finite. The seed-aggregation contract for every table."""
    v = np.asarray([x for x in values if x is not None], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else 0.0)


# ---------------------------------------------------------------------------
# per-arm, per-regime evaluation
# ---------------------------------------------------------------------------

def eval_arm_on_regime(provider_or_model, regime_npz, regime_name: str, *,
                       restrict=None) -> dict:
    """Greek metrics of one arm on one anchor-grid regime, at the regime's OWN params.

    `provider_or_model` is a pinn_provider.PINNProvider (its `.model` is used) or a bare
    SobolevPINN. The network is evaluated on every UNMASKED grid point (mask_any) of
    `regime_npz`, with (kappa, theta, xi, rho, v0) taken from the regime's `params` field
    and injected into the parameter input columns — the regime IS the evaluation point (see
    module docstring; this is NOT the theta_train-frozen hedging convention). `restrict`
    (optional flat/grid boolean mask, C-order on (nS, nK, nT)) intersects the unmasked set
    to score a slice (wing / tau holdout).

    Returns {greek: {n_unmasked, rmse, rel_rmse, p50, p90, p95, p99, oracle_unc_rms}} for
    greek in ("price","delta","gamma","vega","vanna"). rel_rmse = rmse / rms(consensus);
    for GAMMA it is computed on body (non-wing) points only and is NaN on a pure-wing slice
    (contract wing rule). oracle_unc_rms is the rms of the per-point uncertainty field on
    the same points. improvement_to_oracle_noise is left to the comparison layer.
    """
    d = np.load(regime_npz)
    model = getattr(provider_or_model, "model", provider_or_model)
    cfg = model.cfg
    p0 = next(model.parameters())
    dtype, device = p0.dtype, p0.device

    S_ax, K_ax, T_ax = (np.asarray(d[f"{a}_axis"], float) for a in ("S", "K", "tau"))
    Sg, Kg, Tg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")
    Sf, Kf, Tf = Sg.ravel(), Kg.ravel(), Tg.ravel()
    reg = {k: float(v) for k, v in zip(HESTON_PARAM_NAMES, np.asarray(d["params"], float))}

    keep = ~np.asarray(d["mask_any"], bool).ravel()          # unmasked = legs agree
    wing = (Sf / Kf < WING_LO) | (Sf / Kf > WING_HI)
    sel = keep & (np.asarray(restrict, bool).ravel() if restrict is not None else True)
    idx = np.flatnonzero(sel)

    if idx.size == 0:                                        # nothing to score on this slice
        return {g: {"n_unmasked": 0, "rmse": float("nan"), "rel_rmse": float("nan"),
                    "p50": float("nan"), "p90": float("nan"), "p95": float("nan"),
                    "p99": float("nan"), "oracle_unc_rms": float("nan")} for g in GREEKS}

    col = {"S": Sf[idx], "K": Kf[idx], "tau": Tf[idx]}
    for k in HESTON_PARAM_NAMES:
        col[k] = np.full(idx.size, reg[k])                  # regime's OWN params, all columns
    col["v"] = col["v0"]                                    # option_B alias -> spot variance
    x = torch.as_tensor(np.stack([col[n] for n in cfg.inputs], axis=1),
                        dtype=dtype, device=device)
    chunk = getattr(provider_or_model, "chunk", None) or cfg.second_order_microbatch or None
    greeks = model.greeks_eval(x, need=("delta", "gamma", "vega", "vanna"), chunk=chunk)
    with torch.no_grad():
        price = model(x)
    pred = {"price": price, "delta": greeks["delta"], "gamma": greeks["gamma"],
            "vega": greeks["vega"], "vanna": greeks["vanna"]}

    wing_sel = wing[idx]
    out = {}
    for g in GREEKS:
        cons = np.asarray(d[f"consensus_{g}"], float).ravel()[idx]
        unc = np.asarray(d[f"uncertainty_{g}"], float).ravel()[idx]
        err = pred[g].detach().to("cpu", torch.float64).numpy() - cons
        abserr = np.abs(err)
        rmse = _rms(err)
        if g == "gamma":                                    # WING RULE: absolute only on wing
            body = ~wing_sel
            denom = _rms(cons[body]) if body.any() else float("nan")
            rel = _rms(err[body]) / denom if (body.any() and denom > 0) else float("nan")
        else:
            denom = _rms(cons)
            rel = rmse / denom if denom > 0 else float("nan")
        p50, p90, p95, p99 = (float(x_) for x_ in np.percentile(abserr, [50, 90, 95, 99]))
        out[g] = {"n_unmasked": int(idx.size), "rmse": rmse, "rel_rmse": rel,
                  "p50": p50, "p90": p90, "p95": p95, "p99": p99,
                  "oracle_unc_rms": _rms(unc)}
    return out


# ---------------------------------------------------------------------------
# CSV schemas + writer
# ---------------------------------------------------------------------------

PERSEED_COLS = ["regime", "slice", "arm", "seed", "greek", "n_unmasked",
                "rmse", "rel_rmse", "p50", "p90", "p95", "p99", "oracle_unc_rms",
                "reduction_vs_standard_pinn", "improvement_to_oracle_noise", "price_parity"]

_AGG_METRICS = ("rmse", "rel_rmse", "p95", "oracle_unc_rms",
                "reduction_vs_standard_pinn", "improvement_to_oracle_noise", "price_parity")
AGG_COLS = ["regime", "slice", "arm", "greek", "n_seeds"] + \
           [f"{m}_{s}" for m in _AGG_METRICS for s in ("mean", "std")]

THRESHOLD_COLS = ["regime", "arm",
                  "gamma_reduction_mean", "gamma_reduction_std",
                  "vega_reduction_mean", "vega_reduction_std",
                  "price_parity_mean", "price_parity_std",
                  "gamma_ge_0.15", "vega_ge_0.15", "price_parity_within_0.10", "pass"]


def _write_csv(path, cols, rows) -> None:
    """Write `rows` with the fixed `cols` header; non-finite floats become blank cells."""
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean = {}
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float) and not math.isfinite(v):
                    v = ""
                clean[c] = v
            w.writerow(clean)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def _slice_masks(regime_npz) -> dict:
    """wing / tau holdout flat masks (C-order) recomputed from the regime grid geometry."""
    d = np.load(regime_npz)
    S_ax, K_ax, T_ax = (np.asarray(d[f"{a}_axis"], float) for a in ("S", "K", "tau"))
    Sg, Kg, Tg = np.meshgrid(S_ax, K_ax, T_ax, indexing="ij")
    wing = ((Sg / Kg < WING_LO) | (Sg / Kg > WING_HI)).ravel()
    tau = np.zeros(Tg.shape, bool)
    tau[:, :, -1] = True                                    # top tau slice (tau = tau_max)
    return {"wing": wing, "tau": tau.ravel()}


def _collect(regime_names, slice_kinds_for, provs, arms, seeds, anchors_dir) -> dict:
    """recs[(regime, slice, seed, arm)] = {greek: metricdict} for every needed cell."""
    recs = {}
    for regime in regime_names:
        npz = str(Path(anchors_dir) / f"{regime}_grid.npz")
        slices = {"full": None}
        for kind in slice_kinds_for(regime):
            slices[kind] = _slice_masks(npz)[kind]
        for seed in seeds:
            for arm in arms:
                for slname, mask in slices.items():
                    recs[(regime, slname, seed, arm)] = eval_arm_on_regime(
                        provs[seed][arm], npz, regime, restrict=mask)
    return recs


def _perseed_rows(recs, regime_names, slice_kinds_for, seeds, arms) -> list:
    """Per (regime, slice, seed, arm, greek) rows with the comparison-to-baseline columns."""
    rows = []
    for regime in regime_names:
        for slname in ("full", *slice_kinds_for(regime)):
            for seed in seeds:
                base = recs.get((regime, slname, seed, BASELINE_ARM))
                for arm in arms:
                    m = recs.get((regime, slname, seed, arm))
                    if m is None:
                        continue
                    pp = (price_parity(m["price"]["rel_rmse"], base["price"]["rel_rmse"])
                          if base is not None else float("nan"))
                    for g in GREEKS:
                        red = (reduction_vs_baseline(m[g]["rmse"], base[g]["rmse"])
                               if base is not None else float("nan"))
                        imp = (improvement_to_oracle_noise(
                                   m[g]["rmse"], base[g]["rmse"], m[g]["oracle_unc_rms"])
                               if base is not None else float("nan"))
                        rows.append({"regime": regime, "slice": slname, "arm": arm,
                                     "seed": seed, "greek": g, **m[g],
                                     "reduction_vs_standard_pinn": red,
                                     "improvement_to_oracle_noise": imp,
                                     "price_parity": pp})
    return rows


def _agg_rows(perseed_rows) -> list:
    """Seed-aggregate per-seed rows over (regime, slice, arm, greek) -> mean/std columns."""
    groups: dict = {}
    for r in perseed_rows:
        groups.setdefault((r["regime"], r["slice"], r["arm"], r["greek"]), []).append(r)
    agg = []
    for (regime, slname, arm, greek), rs in groups.items():
        rec = {"regime": regime, "slice": slname, "arm": arm, "greek": greek,
               "n_seeds": len(rs)}
        for metric in _AGG_METRICS:
            mean, std = aggregate_seeds([r[metric] for r in rs])
            rec[f"{metric}_mean"], rec[f"{metric}_std"] = mean, std
        agg.append(rec)
    return agg


def _threshold_rows(agg_rows, primary_regimes, arms, thresholds: dict) -> list:
    """OOD Greek pre-check at the CONTRACT's thresholds (audit C1/A1).

    `thresholds` is a Hedging_backtest.contract_thresholds dict: Gamma & Vega RMSE
    reduction >= acceptance_thresholds.ood_{gamma,vega}_rmse_reduction_min at price
    parity within acceptance_thresholds.price_parity_within. No literals here.
    """
    g_min = float(thresholds["ood_gamma_reduction_min"])
    v_min = float(thresholds["ood_vega_reduction_min"])
    p_tol = float(thresholds["price_parity_within"])
    check_arms = [a for a in ("rung3", "rung2") if a in arms]
    lut = {(r["regime"], r["arm"], r["greek"]): r for r in agg_rows if r["slice"] == "full"}
    rows = []
    for regime in primary_regimes:
        for arm in check_arms:
            gr = lut.get((regime, arm, "gamma"))
            vr = lut.get((regime, arm, "vega"))
            if gr is None or vr is None:
                continue
            g_red, g_std = gr["reduction_vs_standard_pinn_mean"], gr["reduction_vs_standard_pinn_std"]
            v_red, v_std = vr["reduction_vs_standard_pinn_mean"], vr["reduction_vs_standard_pinn_std"]
            pp, pp_std = gr["price_parity_mean"], gr["price_parity_std"]
            g_ok, v_ok = g_red >= g_min, v_red >= v_min
            p_ok = math.isfinite(pp) and abs(pp) <= p_tol
            rows.append({"regime": regime, "arm": arm,
                         "gamma_reduction_mean": g_red, "gamma_reduction_std": g_std,
                         "vega_reduction_mean": v_red, "vega_reduction_std": v_std,
                         "price_parity_mean": pp, "price_parity_std": pp_std,
                         # NOTE: these three column NAMES carry the contract's
                         # current numbers; the values they hold are computed from
                         # `thresholds` (see audit/FINDINGS_ADDENDUM.md).
                         "gamma_ge_0.15": bool(g_ok), "vega_ge_0.15": bool(v_ok),
                         "price_parity_within_0.10": bool(p_ok),
                         "pass": bool(g_ok and v_ok and p_ok)})
    return rows


def run_greek_eval(cfg, ckpt_root, arms, seeds, anchors_dir, out_dir,
                   thresholds: dict | None = None) -> dict:
    """Score every arm x seed on the anchor grids; write the primary + sanity + comparison
    tables and the threshold pre-check. Returns the written paths and the in-memory rows.

    `cfg` is the contract (heston_benchmark_v6.yaml, dict or path). Primary regimes are
    cfg.splits.heldout_greek_and_hedging (near_feller, strong_neg_corr); sanity regimes are
    the remaining eval anchors (baseline, high_variance, feller_violating_volvol) and are
    scored on the full grid plus the wing / tau holdout slices. `arms` are ENGINE names
    (pinn_provider._ARM_DIR maps them to checkpoint dirs). standard_pinn is the comparison
    baseline; include it in `arms` for the comparison columns to be populated.

    `thresholds` is a Hedging_backtest.contract_thresholds dict; it defaults to the
    one derived from `cfg` itself, so the pre-check thresholds are ALWAYS the
    contract's (audit C1/A1) and never a literal in this module.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml.safe_load(Path(cfg).read_text())
    if thresholds is None:
        thresholds = contract_thresholds(cfg)
    r, q = float(cfg["grid"]["r"]), float(cfg["grid"]["q"])
    arms, seeds = list(arms), list(seeds)
    primary = list(cfg["splits"]["heldout_greek_and_hedging"])
    sanity = [rg for rg in cfg["splits"]["eval_anchors_heston"] if rg not in primary]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # one provider registry per seed (loud error if a checkpoint is missing)
    provs = {s: build_providers(cfg, ckpt_root, arms, s, r, q, include_oracle=False)
             for s in seeds}

    primary_slices = lambda regime: ()                       # primary: full grid only
    sanity_slices = lambda regime: ("wing", "tau")           # sanity: + wing / tau holdouts

    prim_recs = _collect(primary, primary_slices, provs, arms, seeds, anchors_dir)
    san_recs = _collect(sanity, sanity_slices, provs, arms, seeds, anchors_dir)

    prim_rows = _perseed_rows(prim_recs, primary, primary_slices, seeds, arms)
    san_rows = _perseed_rows(san_recs, sanity, sanity_slices, seeds, arms)
    prim_agg = _agg_rows(prim_rows)
    san_agg = _agg_rows(san_rows)
    thresh = _threshold_rows(prim_agg, primary, arms, thresholds)

    paths = {
        "ood_param_greeks": out / "ood_param_greeks.csv",
        "ood_param_greeks_agg": out / "ood_param_greeks_agg.csv",
        "sanity_in_domain": out / "sanity_in_domain.csv",
        "sanity_in_domain_agg": out / "sanity_in_domain_agg.csv",
        "threshold_precheck": out / "threshold_precheck.csv",
    }
    _write_csv(paths["ood_param_greeks"], PERSEED_COLS, prim_rows)
    _write_csv(paths["ood_param_greeks_agg"], AGG_COLS, prim_agg)
    _write_csv(paths["sanity_in_domain"], PERSEED_COLS, san_rows)
    _write_csv(paths["sanity_in_domain_agg"], AGG_COLS, san_agg)
    _write_csv(paths["threshold_precheck"], THRESHOLD_COLS, thresh)

    print(f"Greek eval: {len(arms)} arms x {len(seeds)} seeds; primary {primary}, "
          f"sanity {sanity}. Wrote {', '.join(p.name for p in paths.values())} to {out}.")
    print("THRESHOLD PRE-CHECK (informational; verdicts live in P13 — Gamma reduction "
          f">= {thresholds['ood_gamma_reduction_min']} & Vega reduction >= "
          f"{thresholds['ood_vega_reduction_min']} at price parity within "
          f"{thresholds['price_parity_within']}):")
    if not thresh:
        print("  (no rung2/rung3 arm present — nothing to pre-check)")
    for t in thresh:
        print(f"  {t['regime']:>16s} {t['arm']:>6s}: "
              f"Gamma red {t['gamma_reduction_mean']:+.3f}+/-{t['gamma_reduction_std']:.3f}, "
              f"Vega red {t['vega_reduction_mean']:+.3f}+/-{t['vega_reduction_std']:.3f}, "
              f"price parity {t['price_parity_mean']:+.3f} -> "
              f"{'PASS' if t['pass'] else 'no'}")

    return {"paths": {k: str(v) for k, v in paths.items()},
            "primary": prim_rows, "primary_agg": prim_agg,
            "sanity": san_rows, "sanity_agg": san_agg,
            "threshold": thresh, "primary_regimes": primary, "sanity_regimes": sanity}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contract", default="heston_benchmark_v6.yaml")
    ap.add_argument("--ckpt-root", required=True, help="results/grid root (<arm>/s<seed>/best.pt)")
    ap.add_argument("--arms", required=True, help="comma-separated engine arm names")
    ap.add_argument("--seeds", required=True, help="comma-separated seeds")
    ap.add_argument("--anchors-dir", required=True, help="generate_anchor_grids output dir")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_greek_eval(args.contract, args.ckpt_root,
                   [a.strip() for a in args.arms.split(",")],
                   [int(s) for s in args.seeds.split(",")],
                   args.anchors_dir, args.out_dir)
