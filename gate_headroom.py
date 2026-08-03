"""Oracle-headroom gate (contract `oracle_headroom_gate`): the week-one power
check that RUNS BEFORE ALL TRAINING (runs_before: all_training).

Hedge the confirmatory instrument twice on the same CRN paths: (a) with exact
theta_train oracle Greeks; (b) with the same oracle whose DELTA is corrupted
by an error consistent with a PINN gamma error of a requested size. The
delta-only CVaR95 spread between (a) and (b) is the CEILING on any effect the
project can show on the primary metric (Sakuma Table 5 — oracle ~= network
hedge — is the power warning). The gate itself decides nothing: the go/no-go
is a HUMAN decision (contract decision_rule: spread below the pre-registered
10% relative CVaR95 threshold at the chosen frequency and TC tiers => RETUNE
before M4, levers in order: rebalancing frequency, misspecification severity).

Corruption models (NoisyOracleProvider; price and gamma pass through
UNCHANGED, only delta is corrupted):
- mode="field" (PRIMARY): a FROZEN random smooth field eta(S, v, tau) built
  from random Fourier features, calibrated so the std of the ANALYTIC
  d(eta)/dS over a reference state cloud equals sigma_gamma_target — a delta
  error consistent with a gamma error of the requested size, smooth in state
  and persistent in time, like a real trained net's error.
- mode="iid" (CONTRAST ONLY, reported separately): i.i.d. per-call delta
  noise at the SAME spatial amplitude as the field (sigma_delta = std of the
  field's delta error over the reference cloud) but redrawn every call — the
  turnover-inflating strawman that isolates TEMPORAL structure: identical
  delta-error magnitude, zero persistence. Its spread OVERSTATES the cost
  channel and is shown only to bracket the field mode.

SPEC BUG-FIX (documented deviation from the literal task spec, evidence-based):
  Two changes were required for the corruption model to actually represent a
  GAMMA error and for the field-vs-iid contrast to mean what the contract
  intends.
  (1) The field is ANISOTROPIC, S-dominated (bandwidth (1.0, 0.1, 0.1) in
      (S, v, tau)) rather than isotropic bw=1.0. A gamma error is a
      curvature-in-S error: the induced delta error is smooth in S and only
      weakly dependent on v/tau. With an ISOTROPIC field, d(eta)/dv is as large
      as d(eta)/dS, and because the range-normalized variance coordinate moves
      ~4x more per daily step than the spot coordinate (measured on the
      confirmatory paths: mean |dz_v| ~ 0.038 vs |dz_S| ~ 0.009), the isotropic
      field injects per-step delta JITTER from variance motion — turnover that
      has nothing to do with a gamma error, making field t_ex EXCEED iid t_ex,
      the opposite of the design's premise. Shrinking the v/tau bandwidth makes
      the field a near-pure curvature-in-S error, restoring the smoothness the
      contract relies on.
  (2) The iid comparator is matched to the field's SPATIAL amplitude (no
      sqrt(dt) factor). The original sqrt(mean(v)*dt) scale is ~0.02 at daily
      dt, which suppresses iid turnover by construction and makes the "iid
      overstates cost" contrast vacuous at the contract frequency. Matching the
      delta-error magnitude isolates the one thing that should differ:
      persistence in time.
  Neither change touches the gate's CVaR95 spread reading or the go/no-go — it
  only makes the field a faithful gamma-error surrogate and the iid arm a fair
  contrast.

FULL-SIZE GATE IS HUMAN-LAUNCHED (after the pilot; never run from tests):
    python gate_headroom.py --mode field --out-dir results/gate
    python gate_headroom.py --mode iid   --out-dir results/gate_iid
Single pilot-calibrated point, once the pilot fit exists:
    python gate_headroom.py --sigma-gamma <abs sigma_gamma from pilot fit>
Smoke sizing for local checks: add --n-paths 128 --n-seeds 1.
"""
from __future__ import annotations

import copy
import os

import numpy as np

import Hedging_backtest as hb
import providers as pv
from oracle import HestonParams
from providers import HestonCFProvider

_STREAM_FIELD = 11          # rng stream id of the frozen RFF draw (spec-pinned)
_STREAM_IID = 12            # rng stream id of the iid strawman draws

_N_FEATURES = 256           # Kf random Fourier features
# ANISOTROPIC bandwidth (S, v, tau) in range-normalized units: S at the spec's
# bw=1.0, v/tau shrunk 10x so the field is a near-pure curvature-in-S error
# (a gamma error). See the module docstring SPEC BUG-FIX note.
_BANDWIDTH = (1.0, 0.1, 0.1)
_DELTA_CLIP = (-0.05, 1.05)
_SIGMA_REL_DEFAULT = (0.1, 0.2, 0.4, 0.8)

# ---------------------------------------------------------------------------
# geometry helpers (contract-derived; no YAML edits, read-only consumption)
# ---------------------------------------------------------------------------


def _grid_ranges(cfg: dict) -> dict:
    """Range-normalization box of the corruption field: contract grid S and
    tau ranges; v uses the training-hypercube v0 range (spec clause — path
    variance has no contract grid axis of its own)."""
    bm = cfg["benchmark"]
    g = bm["grid"]
    v0 = bm["training_parameterization"]["sampling"]["ranges"]["v0"]
    return {"S": (float(g["S"]["min"]), float(g["S"]["max"])),
            "v": (float(v0[0]), float(v0[1])),
            "tau": (float(g["tau"]["min"]), float(g["tau"]["max"]))}


def _trim_to_combined_cell(cfg: dict) -> dict:
    """In-memory sweep trim to the contract confirmatory geometry (combined
    perturbation, magnitude 1.0, no cross-model sweeps) — mirrors
    test_hedging_backtest._trim_to_one_cell; the contract file is untouched."""
    mis = cfg["benchmark"]["hedging_simulation"]["misspecification"]
    mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
    mis["cross_model"] = []
    cfg["engine"]["misspecification"]["magnitudes"] = [1.0]
    return cfg


def confirmatory_dgp(cfg: dict) -> HestonParams:
    """Hedge-side DGP of the confirmatory cell: baseline train regime pushed
    along the combined perturbation at magnitude 1.0 (xi 0.45, rho -0.80)."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]["misspecification"]
    base = hb.SimParams.from_regime(bm["regimes"][hs["train_params"]],
                                    bm["grid"]["r"], bm["grid"]["q"])
    hp = hb.perturb_params(base, "combined", 1.0,
                           eng["misspecification"]["directions"])
    return HestonParams(kappa=hp.kappa, theta=hp.theta, xi=hp.xi, rho=hp.rho,
                        v0=hp.v0)


def reference_state_cloud(cfg: dict, n_states: int = 2000,
                          n_paths: int = 256, seed: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, v, tau) reference cloud via providers.sample_qe_states at the
    combined-perturbation DGP — the state distribution the engine will feed
    the providers on the confirmatory cell. Deterministic in `seed` (default:
    the contract global seed), so the field calibration is reproducible."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    inst = bm["hedging_simulation"]["instrument"]
    T_prime = float(eng["horizon"]["T_prime"])
    n_steps = int(round(T_prime * eng["rebalancing"]["frequency_per_year"]))
    if seed is None:
        seed = int(bm["meta"]["global_seed"])
    return pv.sample_qe_states(confirmatory_dgp(cfg), bm["grid"]["r"],
                               bm["grid"]["q"], float(inst["S0"]),
                               float(inst["tau0"]), T_prime, n_steps, n_paths,
                               int(seed), eng["simulation"]["psi_c"], n_states)


def gamma_rms(provider: HestonCFProvider, states, K: float) -> float:
    """rms(Gamma_oracle) over the reference cloud — the scale that converts
    the relative sigma ladder into absolute sigma_gamma targets."""
    S, v, tau = (np.asarray(a, float).ravel() for a in states)
    g = np.empty(S.size)
    for t in np.unique(tau):
        m = tau == t
        g[m] = np.asarray(provider.evaluate(S[m], v[m], float(t), K)["gamma"],
                          float)
    return float(np.sqrt(np.mean(g ** 2)))

# ---------------------------------------------------------------------------
# corrupted oracle provider
# ---------------------------------------------------------------------------


class NoisyOracleProvider:
    """Oracle wrapper returning the base provider's price/gamma (and any other
    Greeks) UNCHANGED and delta_noisy = clip(delta + eta, -0.05, 1.05).

    The frozen RFF field eta(z) = amp * sqrt(2/Kf) * sum_k a_k cos(w_k . z +
    b_k) is built in BOTH modes on the range-normalized z = (S, v, tau), with
    Kf = 256, w_k[:, d] ~ N(0, (2*pi*bw_d)^2) for the ANISOTROPIC bandwidth
    bw = (1.0, 0.1, 0.1) (S-dominated; see class/module notes), a_k ~ N(0, 1),
    b_k ~ U[0, 2*pi) — all drawn ONCE from np.random.default_rng([seed, 11])
    and never redrawn. `amp` is calibrated so the std over `ref_states` of the
    ANALYTIC d(eta)/dS (closed form from the RFF sum, no autodiff) equals
    sigma_gamma_target.

    mode="field" (PRIMARY): delta_noisy = clip(delta + eta). A delta error
    consistent with a gamma error of the requested size, smooth in state and
    persistent in time. evaluate() is a pure function of (S, v, tau): same
    inputs -> bit-identical output.

    mode="iid" (CONTRAST ONLY): delta + eps with eps ~ N(0, sigma_delta^2)
    drawn i.i.d. per evaluate() call from stream [seed, 12], sigma_delta = the
    std of the field's delta error eta over the reference cloud — the SAME
    spatial amplitude as the field, redrawn every call. Isolates temporal
    structure (identical magnitude, zero persistence); reported separately
    only to bracket the field mode.
    """

    def __init__(self, base, sigma_gamma_target: float, seed: int,
                 ranges: dict, ref_states, mode: str = "field",
                 n_features: int = _N_FEATURES,
                 bandwidth: tuple = _BANDWIDTH) -> None:
        if mode not in ("field", "iid"):
            raise ValueError(f"unknown corruption mode {mode!r}")
        self.base = base
        self.mode = mode
        self.sigma_gamma_target = float(sigma_gamma_target)
        self.seed = int(seed)
        self.ranges = {k: (float(lo), float(hi))
                       for k, (lo, hi) in ranges.items()}
        # the frozen RFF field is built in BOTH modes: field mode evaluates it,
        # iid mode matches its spatial amplitude
        rng = np.random.default_rng([self.seed, _STREAM_FIELD])
        self._nf = int(n_features)
        bw = np.broadcast_to(np.asarray(bandwidth, float), (3,))
        self._w = rng.standard_normal((self._nf, 3)) * (2.0 * np.pi * bw)
        self._a = rng.standard_normal(self._nf)
        self._b = rng.uniform(0.0, 2.0 * np.pi, self._nf)
        self.amp = 1.0                          # unit amp for calibration pass
        unit_std = float(np.std(self.eta_dS(*ref_states)))
        self.amp = (self.sigma_gamma_target / unit_std
                    if self.sigma_gamma_target != 0.0 else 0.0)
        if mode == "iid":
            self.sigma_delta = float(np.std(self.eta(*ref_states)))
            self._rng = np.random.default_rng([self.seed, _STREAM_IID])

    def _z(self, S, v, tau) -> tuple[np.ndarray, tuple]:
        """Range-normalized [n, 3] state matrix plus the broadcast shape."""
        S, v, tau = np.broadcast_arrays(np.asarray(S, float),
                                        np.asarray(v, float),
                                        np.asarray(tau, float))
        cols = [(arr.ravel() - self.ranges[nm][0])
                / (self.ranges[nm][1] - self.ranges[nm][0])
                for nm, arr in (("S", S), ("v", v), ("tau", tau))]
        return np.stack(cols, axis=1), S.shape

    def eta(self, S, v, tau) -> np.ndarray:
        """Frozen smooth corruption field (field mode)."""
        z, shape = self._z(S, v, tau)
        phase = z @ self._w.T + self._b
        return (self.amp * np.sqrt(2.0 / self._nf)
                * (np.cos(phase) @ self._a)).reshape(shape)

    def eta_dS(self, S, v, tau) -> np.ndarray:
        """ANALYTIC d(eta)/dS from the RFF sum (chain rule through the S
        normalization; no autodiff) — the calibration target statistic."""
        z, shape = self._z(S, v, tau)
        lo, hi = self.ranges["S"]
        phase = z @ self._w.T + self._b
        coef = self._a * self._w[:, 0] / (hi - lo)
        return (-self.amp * np.sqrt(2.0 / self._nf)
                * (np.sin(phase) @ coef)).reshape(shape)

    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float,
                 K: float) -> dict:
        out = dict(self.base.evaluate(S, v, tau, K))
        delta = np.asarray(out["delta"], float)
        err = (self.eta(S, v, tau) if self.mode == "field"
               else self._rng.normal(0.0, self.sigma_delta, delta.shape))
        out["delta"] = np.clip(delta + err, *_DELTA_CLIP)
        return out

# ---------------------------------------------------------------------------
# gate driver
# ---------------------------------------------------------------------------


def run_gate(cfg: dict, sigma_rel_list: tuple = _SIGMA_REL_DEFAULT,
             seed_list: list | None = None, mode: str = "field",
             out_dir: str = "results/gate",
             sigma_gamma_abs: float | None = None,
             n_cloud_states: int = 2000, n_cloud_paths: int = 256,
             spread_threshold_rel: float | None = None) -> dict:
    """Run the oracle-headroom gate on the confirmatory cell and write
    headroom.csv + headroom_report.md into `out_dir`.

    Parameters
    ----------
    cfg : dict
        resolve_config output; deep-copied, then trimmed in memory to the
        confirmatory geometry (combined perturbation, magnitude 1.0, baseline
        train regime, T' = 0.17, daily grid; n_paths/seeds as configured).
    sigma_rel_list : tuple of float
        Relative sigma ladder; sigma_gamma_target = sigma_rel *
        rms(Gamma_oracle) with the rms measured on the reference state cloud.
    seed_list : list of int, optional
        Overrides cfg["derived"]["seeds"] (CLI --n-seeds).
    mode : {"field", "iid"}
        Corruption model; "iid" is the contrast-only strawman.
    sigma_gamma_abs : float, optional
        Absolute pilot-calibrated sigma_gamma (CLI --sigma-gamma); when given
        the sweep is replaced by this single "pilot" point.
    spread_threshold_rel : float, optional
        The pre-registered relative CVaR95 spread the DECISION section flags on.
        Defaults to the CONTRACT (oracle_headroom_gate.spread_threshold_rel) —
        never a Python literal (audit G1/C1).

    Returns
    -------
    dict with keys records (per sigma/tc/seed), summary (per sigma/tc),
    decision (per tc), spread_threshold_rel, rms_gamma, csv_path, report_path,
    rows (engine rows).
    """
    cfg = _trim_to_combined_cell(copy.deepcopy(cfg))
    if spread_threshold_rel is None:
        spread_threshold_rel = hb.contract_thresholds(cfg)["gate_spread_threshold_rel"]
    spread_threshold_rel = float(spread_threshold_rel)
    if seed_list is not None:
        cfg["derived"]["seeds"] = [int(s) for s in seed_list]
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    K = float(hs["instrument"]["K"])
    regime = bm["regimes"][hs["misspecification"]["train_params"]]
    theta_train = HestonParams(kappa=regime["kappa"], theta=regime["theta"],
                               xi=regime["xi"], rho=regime["rho"],
                               v0=regime["v0"])
    base = HestonCFProvider(theta_train, r, q)
    ref = reference_state_cloud(cfg, n_states=n_cloud_states,
                                n_paths=n_cloud_paths)
    rms = gamma_rms(base, ref, K)
    ranges = _grid_ranges(cfg)
    gate_seed = int(bm["meta"]["global_seed"])   # ONE frozen field per gate

    if sigma_gamma_abs is not None:
        arms = [("pilot", float(sigma_gamma_abs) / rms, float(sigma_gamma_abs))]
    else:
        arms = sorted(((f"s{sr:g}", float(sr), float(sr) * rms)
                       for sr in sigma_rel_list), key=lambda a: a[1])
    oracle_name = eng.get("oracle_provider_name", "oracle")
    providers = {oracle_name: base}
    for label, _sr, sa in arms:
        providers[f"noisy_{label}"] = NoisyOracleProvider(
            base, sa, gate_seed, ranges, ref, mode=mode)

    rows = hb.run_headline(cfg, providers,
                           out_dir=os.path.join(out_dir, "engine"))

    by = {(row["method"], row["tc"], row["seed"]): row for row in rows}
    tiers = list(hs["transaction_costs"]["tiers"])
    seeds = cfg["derived"]["seeds"]
    records = []
    for label, sr, sa in arms:
        for tc in tiers:
            for seed in seeds:
                ro = by[(oracle_name, tc, seed)]
                rn = by[(f"noisy_{label}", tc, seed)]
                co, cn = float(ro["cvar"]), float(rn["cvar"])
                ci_lo = float(rn["pnl_vs_oracle_ci_lo"])
                records.append({
                    "mode": mode, "arm": label, "sigma_rel": sr,
                    "sigma_gamma_abs": sa, "tc": tc, "seed": seed,
                    "cvar_oracle": co, "cvar_noisy": cn,
                    "spread_rel": ((cn - co) / co if co != 0.0
                                   else float("nan")),
                    "cvar_diff": float(rn["pnl_vs_oracle_cvar_diff"]),
                    "ci_lo": ci_lo,
                    "ci_hi": float(rn["pnl_vs_oracle_ci_hi"]),
                    "ci_excludes_zero": ci_lo > 0.0,
                    "t_ex": float(rn["t_ex"]),
                    "n_paths": int(rn["n_paths"])})
    summary = []
    for label, sr, sa in arms:
        for tc in tiers:
            grp = [rec for rec in records
                   if rec["arm"] == label and rec["tc"] == tc]
            sp = np.array([g["spread_rel"] for g in grp], float)
            summary.append({
                "mode": mode, "arm": label, "sigma_rel": sr,
                "sigma_gamma_abs": sa, "tc": tc, "n_seeds": len(grp),
                "spread_rel_mean": float(np.mean(sp)),
                "spread_rel_seed_std": (float(np.std(sp, ddof=1))
                                        if len(grp) > 1 else 0.0),
                "ci_excludes_zero_frac": float(np.mean(
                    [g["ci_excludes_zero"] for g in grp])),
                "t_ex_mean": float(np.mean([g["t_ex"] for g in grp]))})
    # smallest sigma_rel whose mean spread clears the pre-registered relative
    # threshold (contract oracle_headroom_gate.spread_threshold_rel) WITH every
    # seed's paired CI excluding 0
    decision = {}
    for tc in tiers:
        decision[tc] = next(
            (s for s in summary if s["tc"] == tc
             and s["spread_rel_mean"] >= spread_threshold_rel
             and s["ci_excludes_zero_frac"] == 1.0), None)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = hb.write_rows_csv(records, os.path.join(out_dir, "headroom.csv"))
    report_path = _write_report(
        os.path.join(out_dir, "headroom_report.md"), cfg, mode, rms, arms,
        tiers, seeds, summary, decision, spread_threshold_rel)
    return {"records": records, "summary": summary, "decision": decision,
            "spread_threshold_rel": spread_threshold_rel,
            "rms_gamma": rms, "csv_path": csv_path,
            "report_path": report_path, "rows": rows}


def _write_report(path: str, cfg: dict, mode: str, rms: float, arms: list,
                  tiers: list, seeds: list, summary: list,
                  decision: dict, spread_threshold_rel: float) -> str:
    """headroom_report.md: run header, per-(sigma, tc) seed-mean table, and
    the DECISION section (explicitly a HUMAN go/no-go)."""
    eng = cfg["engine"]
    lines = [
        "# Oracle-headroom gate report",
        "",
        "Contract `oracle_headroom_gate` (runs_before: all_training). The",
        "spread between the oracle hedge and the delta-corrupted oracle hedge",
        "is the CEILING on any effect the project can show on the primary",
        "metric (misspecified delta-only CVaR95).",
        "",
        f"- corruption mode: **{mode}**"
        + ("  (PRIMARY: frozen smooth RFF field)" if mode == "field" else
           "  (CONTRAST-ONLY strawman: i.i.d. per-call delta noise inflates "
           "turnover and OVERSTATES the cost channel; shown only to bracket "
           "the field mode)"),
        f"- cell: combined perturbation, magnitude 1.0, baseline train regime,"
        f" T' = {eng['horizon']['T_prime']},"
        f" freq = {eng['rebalancing']['frequency_per_year']}/yr,"
        f" n_paths = {eng['simulation']['n_paths']}, seeds = {list(seeds)}",
        f"- rms(Gamma_oracle) on the reference cloud: **{rms:.6g}**",
        "- absolute sigma_gamma targets: "
        + ", ".join(f"{label}: {sa:.6g}" for label, _sr, sa in arms),
        "",
        "## Spread over seeds (spread_rel = (cvar_noisy - cvar_oracle) / "
        "cvar_oracle)",
        "",
        "| arm | sigma_rel | sigma_gamma | tc | spread_rel mean | seed std |"
        " CI-excl-0 frac | t_ex mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['arm']} | {s['sigma_rel']:.4g} | "
            f"{s['sigma_gamma_abs']:.4g} | {s['tc']} | "
            f"{s['spread_rel_mean']:+.4f} | {s['spread_rel_seed_std']:.4f} | "
            f"{s['ci_excludes_zero_frac']:.2f} | {s['t_ex_mean']:.4f} |")
    lines += [
        "",
        "## DECISION (per tc tier)",
        "",
        f"Smallest sigma_rel with mean spread_rel >= {spread_threshold_rel} (the",
        "pre-registered relative CVaR95 threshold, contract",
        "`oracle_headroom_gate.spread_threshold_rel`) AND the paired per-path",
        "bootstrap 95% CI excluding 0 in every seed:",
        "",
    ]
    for tc in tiers:
        hit = decision[tc]
        lines.append(
            f"- tc = {tc}: "
            + (f"sigma_rel = {hit['sigma_rel']:.4g} "
               f"(sigma_gamma = {hit['sigma_gamma_abs']:.4g})" if hit
               else "NONE — no swept noise level clears the threshold"))
    lines += [
        "",
        "The go/no-go is a HUMAN decision, not this script's (contract",
        "`oracle_headroom_gate.decision_rule`). If plausible pilot-fit",
        "sigma_gamma sits below the passing level, RETUNE before M4 — levers",
        "in order: (1) rebalancing frequency, (2) misspecification severity.",
        "The gate runs BEFORE all training (runs_before: all_training).",
        "",
        "Full-size gate command (human-launched):",
        "",
        "    python gate_headroom.py --mode field --out-dir results/gate",
        "",
        "Pilot-calibrated point once the pilot fit exists:",
        "",
        "    python gate_headroom.py --sigma-gamma <abs sigma_gamma from"
        " pilot fit>",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Oracle-headroom gate (contract oracle_headroom_gate); "
                    "human-launched — see module docstring.")
    ap.add_argument("--mode", choices=("field", "iid"), default="field",
                    help="corruption model (field = primary, iid = "
                         "contrast-only strawman)")
    ap.add_argument("--sigma-rel", type=float, nargs="+",
                    default=list(_SIGMA_REL_DEFAULT),
                    help="pre-pilot relative sigma sweep")
    ap.add_argument("--sigma-gamma", type=float, default=None,
                    help="ABSOLUTE pilot-calibrated sigma_gamma; replaces the "
                         "sweep with the single pilot point")
    ap.add_argument("--n-paths", type=int, default=None,
                    help="override engine simulation.n_paths")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="use only the first N derived seeds")
    ap.add_argument("--out-dir", default="results/gate")
    args = ap.parse_args(argv)
    cfg = hb.resolve_config()
    if args.n_paths is not None:
        cfg["engine"]["simulation"]["n_paths"] = int(args.n_paths)
    seed_list = (cfg["derived"]["seeds"][:args.n_seeds]
                 if args.n_seeds is not None else None)
    res = run_gate(cfg, sigma_rel_list=tuple(args.sigma_rel),
                   seed_list=seed_list, mode=args.mode, out_dir=args.out_dir,
                   sigma_gamma_abs=args.sigma_gamma)
    print(f"rms(Gamma_oracle) = {res['rms_gamma']:.6g}")
    print(f"wrote {res['csv_path']}")
    print(f"wrote {res['report_path']}")


if __name__ == "__main__":
    main()
