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

SIGMA LADDER (contract `oracle_headroom_gate.sigma_rel_ladder`, AM2-3a): the
swept ladder is the CONTRACT's, never a module literal — `decision` rungs are
the only ones the DECISION scan may fire on, `diagnostic` rungs are swept,
written and plotted but excluded from that scan (at ~78% clipped they sit
outside `region_of_validity` and their spread is not a reading of a gamma error
of the labelled size).

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
import warnings

import numpy as np

import Hedging_backtest as hb
import providers as pv
from oracle import HestonParams
from providers import HestonCFProvider

_STREAM_FIELD = 11          # rng stream id of the frozen RFF draw (spec-pinned)
_STREAM_IID = 12            # rng stream id of the iid strawman draws
_STREAM_EFF = 13            # rng stream of the effective-sigma MEASUREMENT draws
                            # (iid mode only; a separate stream so measuring can
                            # never advance the strawman's hedging draws)

_N_FEATURES = 256           # Kf random Fourier features
# ANISOTROPIC bandwidth (S, v, tau) in range-normalized units: S at the spec's
# bw=1.0, v/tau shrunk 10x so the field is a near-pure curvature-in-S error
# (a gamma error). See the module docstring SPEC BUG-FIX note.
_BANDWIDTH = (1.0, 0.1, 0.1)
_DELTA_CLIP = (-0.05, 1.05)
# NO _SIGMA_REL_DEFAULT: the ladder is the CONTRACT's
# (oracle_headroom_gate.sigma_rel_ladder, amendment AM2-3a) and is resolved by
# _resolve_ladder. The old literal (0.1, 0.2, 0.4, 0.8) swept a rung — 0.8 —
# that the amendment DELETED and treated 0.4 as a decision rung, which the
# amendment demoted to diagnostic-only.

# audit G2: what a non-zero clipped_frac means, stated once and reused by the
# report and the run_gate result so a reader cannot meet the number without it.
_CLIPPED_NOTE = (
    "`clipped_frac` is the fraction of delta evaluations on which the "
    f"[{_DELTA_CLIP[0]}, {_DELTA_CLIP[1]}] delta clip BOUND. The field amplitude "
    "is calibrated on the UNCLIPPED field, so wherever the clip binds the "
    "DELIVERED gamma error is smaller than the sigma_gamma this row is labelled "
    "with: the spread is understated and the gate is conservative (the safe "
    "direction for a go/no-go, but the sigma axis is then not the axis it is "
    "labelled with). A value of 0 means the sigma axis is exact. "
    "MEASURED CAVEAT (fix batch 3): that 'smaller, so conservative' reading is "
    "NOT uniform. Where the clip binds, the corrupted hedger is FLAT in S, so "
    "its gamma error there is the oracle's own -Gamma; across the contract's "
    "DECISION rungs the DELIVERED gamma scale comes out LARGER than the nominal "
    "label (up to ~1.5x), and only falls below it once the clip binds nearly "
    "everywhere. Read the direction off `sigma_gamma_effective` below — do not "
    "assume it.")

# AM2-3b: what the two DELIVERED sigma columns mean, stated once and reused by
# the report so the nominal/effective distinction cannot be read past.
_EFFECTIVE_NOTE = (
    "`sigma_gamma` is NOMINAL: the field is calibrated to it BEFORE the "
    f"[{_DELTA_CLIP[0]}, {_DELTA_CLIP[1]}] delta clip. `sigma_gamma_effective` "
    "(gamma units: std of d/dS of the delivered post-clip delta error) and "
    "`sigma_delta_effective` (delta units: std of that error) are what was "
    "actually DELIVERED. sigma_gamma_pilot is a gamma rmse, so the pilot point "
    "is compared against `sigma_gamma_effective` — comparing it against the "
    "delta-error std would be a units error (contract "
    "`effective_sigma_reporting.compare_pilot_against`).")

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


def _base_delta_cloud(base, states, K: float) -> dict:
    """UNCORRUPTED delta AND gamma on the reference cloud (one pass, shared by
    every arm — the base provider is the same object for all of them).

    `effective_sigmas` needs delta to find where the clip binds and gamma because
    d/dS of the delivered error is -Gamma exactly where it does."""
    S, v, tau = (np.asarray(a, float).ravel() for a in states)
    out = {"delta": np.empty(S.size), "gamma": np.empty(S.size)}
    for t in np.unique(tau):
        m = tau == t
        got = base.evaluate(S[m], v[m], float(t), K)
        out["delta"][m] = np.asarray(got["delta"], float)
        out["gamma"][m] = np.asarray(got["gamma"], float)
    return out


def effective_sigmas(provider: "NoisyOracleProvider", states,
                     base_cloud: dict) -> dict:
    """The two DELIVERED (post-clip) corruption scales the contract declares
    (`oracle_headroom_gate.effective_sigma_reporting`, AM2-3b).

    `sigma_rel` / `sigma_gamma_target` label the field BEFORE the clip; wherever
    `_DELTA_CLIP` binds, less corruption is delivered than the label claims. Both
    quantities are measured on the DELIVERED delta error
    `err(z) = clip(delta(z) + eta(z)) - delta(z)` over the reference cloud:

    - `sigma_delta_effective` — std(err). Delta units. The direct measure of how
      much of the intended corruption the clip removed.
    - `sigma_gamma_effective` — std(d(err)/dS). GAMMA units, the post-clip
      counterpart of the calibration statistic std(d(eta)/dS), and the ONLY one
      of the two commensurable with `sigma_gamma_pilot` (also a gamma rmse) —
      hence the contract's `compare_pilot_against: sigma_gamma_effective`.

    d(err)/dS is taken ANALYTICALLY and almost everywhere, not by finite
    difference: where the clip is slack the delivered error IS eta, so the
    derivative is the analytic `eta_dS`; where it binds, err = bound - delta and
    the derivative is exactly -Gamma. A finite difference instead straddles the
    clip boundary on an O(h) set of states and turns each kink into a spike
    ~eta/h, which INFLATES the statistic above its nominal (measured: +11% at
    sigma_rel = 0.1, h = 1e-3 of the S range) — the opposite of the "the clip
    removed corruption" reading the contract asks this number to support.

    In iid mode the delta error is redrawn per call, so it has no spatial
    derivative at all: `sigma_gamma_effective` is NaN and the reason travels with
    it in `note`. The measurement draws from `_STREAM_EFF`, never from the
    strawman's own hedging stream.
    """
    S, v, tau = (np.asarray(a, float).ravel() for a in states)
    lo, hi = _DELTA_CLIP
    delta, gamma = base_cloud["delta"], base_cloud["gamma"]

    if provider.mode == "field":
        err_raw = provider.eta(S, v, tau).ravel()
        err_raw_dS = provider.eta_dS(S, v, tau).ravel()
    else:
        rng = np.random.default_rng([provider.seed, _STREAM_EFF])
        err_raw = rng.normal(0.0, provider.sigma_delta, delta.shape)
        err_raw_dS = None
    raw = delta + err_raw
    bound = (raw < lo) | (raw > hi)
    delivered = np.clip(raw, lo, hi) - delta

    out = {"sigma_delta_effective": float(np.std(delivered)),
           "sigma_gamma_target": provider.sigma_gamma_target,
           "clipped_frac_reference_cloud": float(np.mean(bound))}
    if err_raw_dS is None:
        out["sigma_gamma_effective"] = float("nan")
        out["note"] = ("iid mode: the delta error is redrawn every call, so the "
                       "delivered error has no d/dS — sigma_gamma_effective is "
                       "undefined (NaN), not zero")
    else:
        out["sigma_gamma_effective"] = float(
            np.std(np.where(bound, -gamma, err_raw_dS)))
        out["note"] = ""
    return out

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
        # audit G2: `amp` is calibrated on the UNCLIPPED field and _DELTA_CLIP is
        # applied afterwards, so wherever the clip binds the DELIVERED gamma error
        # is smaller than sigma_gamma_target and the measured spread is understated
        # (conservative, but the sigma axis is then not the axis it is labelled
        # with). The clip is unchanged; these counters make a binding clip visible.
        self._n_clipped = 0
        self._n_delta = 0

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

    @property
    def clipped_fraction(self) -> float:
        """Fraction of delta evaluations _DELTA_CLIP actually bound on, over the
        provider's whole life (NaN before the first evaluate). A non-zero value
        means the delivered corruption was WEAKER than `sigma_gamma_target` on
        that fraction of states, so the arm's spread is understated (audit G2).
        Bookkeeping only — `evaluate` stays a pure function of state."""
        if self._n_delta == 0:
            return float("nan")
        return self._n_clipped / self._n_delta

    def evaluate(self, S: np.ndarray, v: np.ndarray, tau: float,
                 K: float) -> dict:
        out = dict(self.base.evaluate(S, v, tau, K))
        delta = np.asarray(out["delta"], float)
        err = (self.eta(S, v, tau) if self.mode == "field"
               else self._rng.normal(0.0, self.sigma_delta, delta.shape))
        raw = delta + err
        lo, hi = _DELTA_CLIP
        self._n_clipped += int(np.count_nonzero((raw < lo) | (raw > hi)))
        self._n_delta += int(np.asarray(raw).size)
        out["delta"] = np.clip(raw, lo, hi)
        return out

# ---------------------------------------------------------------------------
# sigma ladder (contract AM2-3a) — decision rungs vs labelled diagnostic rungs
# ---------------------------------------------------------------------------


def _resolve_ladder(cfg: dict, sigma_rel_list, sigma_rel_diagnostic
                    ) -> tuple[list, str]:
    """[(sigma_rel, decision_eligible), ...] in ascending sigma, plus the source.

    `sigma_rel_list is None` (the DEFAULT, and what the CLI passes unless a human
    overrides it) => the CONTRACT ladder: `oracle_headroom_gate.sigma_rel_ladder`
    `decision` rungs are decision-eligible, `diagnostic` rungs are swept, reported
    and plotted but NEVER an input to the DECISION scan (amendment AM2-3a: 0.40 is
    ~78% clipped, outside `region_of_validity`, and 0.80 is dropped entirely).

    An explicit `sigma_rel_list` is an OPERATOR OVERRIDE and has left the contract
    ladder: those rungs are decision-eligible and `sigma_rel_diagnostic` (default
    empty) carries any the caller wants labelled diagnostic. Passing a rung the
    CONTRACT calls diagnostic warns loudly rather than silently re-promoting it.
    A rung named in BOTH lists is diagnostic (the conservative reading).
    """
    th = hb.contract_thresholds(cfg)
    if sigma_rel_list is None:
        decision = tuple(th["gate_sigma_rel_decision"])
        diagnostic = tuple(th["gate_sigma_rel_diagnostic"]
                           if sigma_rel_diagnostic is None
                           else (float(s) for s in sigma_rel_diagnostic))
        source = "contract"
    else:
        decision = tuple(float(s) for s in sigma_rel_list)
        diagnostic = tuple(float(s) for s in (sigma_rel_diagnostic or ()))
        source = "override"
        contract_diag = th["gate_sigma_rel_diagnostic"]
        strays = [s for s in decision
                  if any(abs(s - d) <= 1e-12 for d in contract_diag)]
        if strays:
            warnings.warn(
                f"gate ladder OVERRIDE: {strays} are declared DIAGNOSTIC-ONLY by "
                f"oracle_headroom_gate.sigma_rel_ladder.diagnostic but were passed "
                f"as swept decision rungs; the DECISION scan may fire on a rung the "
                f"contract demoted. Pass them via sigma_rel_diagnostic instead.",
                RuntimeWarning)
    eligible: dict[float, bool] = {}
    for sr in decision:
        eligible[float(sr)] = True
    for sr in diagnostic:
        eligible[float(sr)] = False                 # diagnostic wins a collision
    entries = sorted(eligible.items())
    return entries, source


# ---------------------------------------------------------------------------
# gate driver
# ---------------------------------------------------------------------------


def run_gate(cfg: dict, sigma_rel_list: tuple | None = None,
             seed_list: list | None = None, mode: str = "field",
             out_dir: str = "results/gate",
             sigma_gamma_abs: float | None = None,
             n_cloud_states: int = 2000, n_cloud_paths: int = 256,
             spread_threshold_rel: float | None = None,
             sigma_rel_diagnostic: tuple | None = None) -> dict:
    """Run the oracle-headroom gate on the confirmatory cell and write
    headroom.csv + headroom_report.md into `out_dir`.

    Parameters
    ----------
    cfg : dict
        resolve_config output; deep-copied, then trimmed in memory to the
        confirmatory geometry (combined perturbation, magnitude 1.0, baseline
        train regime, T' = 0.17, daily grid; n_paths/seeds as configured).
    sigma_rel_list : tuple of float, optional
        Relative sigma ladder; sigma_gamma_target = sigma_rel *
        rms(Gamma_oracle) with the rms measured on the reference state cloud.
        DEFAULT (None) = the CONTRACT ladder `oracle_headroom_gate.
        sigma_rel_ladder.decision` + `.diagnostic` (AM2-3a) — never a Python
        literal. Passing a list is an operator override (see _resolve_ladder).
    sigma_rel_diagnostic : tuple of float, optional
        Rungs swept and reported but NEVER an input to the DECISION scan.
        Defaults to the contract's `.diagnostic` rungs when the ladder itself
        is the contract's, else empty.
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
    decision (per tc), ladder ({decision, diagnostic, source}),
    spread_threshold_rel, rms_gamma, csv_path, report_path, rows (engine rows).
    """
    cfg = _trim_to_combined_cell(copy.deepcopy(cfg))
    entries, ladder_source = _resolve_ladder(cfg, sigma_rel_list,
                                             sigma_rel_diagnostic)
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

    # arms carry decision eligibility from here on: a DIAGNOSTIC rung is swept,
    # written and plotted exactly like a decision rung and is excluded ONLY from
    # the decision scan (AM2-3a).
    if sigma_gamma_abs is not None:
        arms = [("pilot", float(sigma_gamma_abs) / rms, float(sigma_gamma_abs),
                 True)]
        ladder = {"decision": (float(sigma_gamma_abs) / rms,),
                  "diagnostic": (), "source": "pilot"}
    else:
        arms = [(f"s{sr:g}", float(sr), float(sr) * rms, bool(elig))
                for sr, elig in entries]
        ladder = {"decision": tuple(sr for sr, e in entries if e),
                  "diagnostic": tuple(sr for sr, e in entries if not e),
                  "source": ladder_source}
    oracle_name = eng.get("oracle_provider_name", "oracle")
    providers = {oracle_name: base}
    noisy_by_arm: dict[str, NoisyOracleProvider] = {}
    for label, _sr, sa, _elig in arms:
        noisy_by_arm[label] = NoisyOracleProvider(
            base, sa, gate_seed, ranges, ref, mode=mode)
        providers[f"noisy_{label}"] = noisy_by_arm[label]

    # DELIVERED (post-clip) corruption scales, per arm — AM2-3b. One base-delta
    # pass on the reference cloud is shared by every arm.
    base_cloud = _base_delta_cloud(base, ref, K)
    eff_by_arm = {label: effective_sigmas(noisy_by_arm[label], ref, base_cloud)
                  for label, _sr, _sa, _e in arms}

    rows = hb.run_headline(cfg, providers,
                           out_dir=os.path.join(out_dir, "engine"))

    by = {(row["method"], row["tc"], row["seed"]): row for row in rows}
    tiers = list(hs["transaction_costs"]["tiers"])
    seeds = cfg["derived"]["seeds"]
    records = []
    for label, sr, sa, elig in arms:
        for tc in tiers:
            for seed in seeds:
                ro = by[(oracle_name, tc, seed)]
                rn = by[(f"noisy_{label}", tc, seed)]
                co, cn = float(ro["cvar"]), float(rn["cvar"])
                ci_lo = float(rn["pnl_vs_oracle_ci_lo"])
                records.append({
                    "mode": mode, "arm": label, "sigma_rel": sr,
                    "sigma_gamma_abs": sa,
                    "rung_role": "decision" if elig else "diagnostic",
                    "decision_eligible": bool(elig),
                    # NOMINAL sigma above, DELIVERED post-clip sigmas here — the
                    # contract requires both on every row (AM2-3b).
                    "sigma_delta_effective":
                        eff_by_arm[label]["sigma_delta_effective"],
                    "sigma_gamma_effective":
                        eff_by_arm[label]["sigma_gamma_effective"],
                    "tc": tc, "seed": seed,
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
    for label, sr, sa, elig in arms:
        for tc in tiers:
            grp = [rec for rec in records
                   if rec["arm"] == label and rec["tc"] == tc]
            sp = np.array([g["spread_rel"] for g in grp], float)
            summary.append({
                "mode": mode, "arm": label, "sigma_rel": sr,
                "sigma_gamma_abs": sa,
                "rung_role": "decision" if elig else "diagnostic",
                "decision_eligible": bool(elig),
                "sigma_delta_effective":
                    eff_by_arm[label]["sigma_delta_effective"],
                "sigma_gamma_effective":
                    eff_by_arm[label]["sigma_gamma_effective"],
                "tc": tc, "n_seeds": len(grp),
                "spread_rel_mean": float(np.mean(sp)),
                "spread_rel_seed_std": (float(np.std(sp, ddof=1))
                                        if len(grp) > 1 else 0.0),
                "ci_excludes_zero_frac": float(np.mean(
                    [g["ci_excludes_zero"] for g in grp])),
                "t_ex_mean": float(np.mean([g["t_ex"] for g in grp])),
                # per ARM over the whole run, repeated on each tc row: positions
                # are built once per method and reused across tiers, so there is
                # no per-tc clipping to report (audit G2).
                "clipped_frac": noisy_by_arm[label].clipped_fraction})
    # smallest DECISION-ELIGIBLE sigma_rel whose mean spread clears the
    # pre-registered relative threshold (contract
    # oracle_headroom_gate.spread_threshold_rel) WITH every seed's paired CI
    # excluding 0. Diagnostic rungs are skipped HERE, not upstream: they are
    # still swept, written and plotted (AM2-3a).
    decision = {}
    for tc in tiers:
        decision[tc] = next(
            (s for s in summary if s["tc"] == tc and s["decision_eligible"]
             and s["spread_rel_mean"] >= spread_threshold_rel
             and s["ci_excludes_zero_frac"] == 1.0), None)

    pilot_comparison = (None if sigma_gamma_abs is None else
                        _pilot_comparison(cfg, float(sigma_gamma_abs),
                                          eff_by_arm["pilot"],
                                          noisy_by_arm["pilot"].clipped_fraction))

    os.makedirs(out_dir, exist_ok=True)
    csv_path = hb.write_rows_csv(records, os.path.join(out_dir, "headroom.csv"))
    report_path = _write_report(
        os.path.join(out_dir, "headroom_report.md"), cfg, mode, rms, arms,
        tiers, seeds, summary, decision, spread_threshold_rel, ladder,
        pilot_comparison)
    return {"records": records, "summary": summary, "decision": decision,
            "ladder": ladder, "effective_sigmas": eff_by_arm,
            "pilot_comparison": pilot_comparison,
            "spread_threshold_rel": spread_threshold_rel,
            "rms_gamma": rms, "csv_path": csv_path,
            "report_path": report_path, "rows": rows,
            "clipped_frac": {label: p.clipped_fraction
                             for label, p in noisy_by_arm.items()},
            "clipped_fraction_note": _CLIPPED_NOTE}


def _pilot_comparison(cfg: dict, sigma_gamma_abs: float, eff: dict,
                      clipped_frac: float) -> dict:
    """Compare the pilot's sigma_gamma against the quantity the CONTRACT names
    (`effective_sigma_reporting.compare_pilot_against`, AM2-3b).

    sigma_gamma_pilot is a GAMMA rmse; `sigma_gamma_effective` is the post-clip
    gamma scale, and it is the only one of the two effective quantities in the
    same units. The field name is read from the contract rather than assumed, so
    a human who re-decides that key moves this comparison with it."""
    field = hb.contract_thresholds(cfg)["gate_compare_pilot_against"]
    return {"compare_against": field,
            "nominal": float(sigma_gamma_abs),
            "effective": float(eff[field]),
            "sigma_delta_effective": float(eff["sigma_delta_effective"]),
            "sigma_gamma_effective": float(eff["sigma_gamma_effective"]),
            "clipped_frac": float(clipped_frac)}


def _write_report(path: str, cfg: dict, mode: str, rms: float, arms: list,
                  tiers: list, seeds: list, summary: list,
                  decision: dict, spread_threshold_rel: float,
                  ladder: dict, pilot_comparison: dict | None = None) -> str:
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
        + ", ".join(f"{label}: {sa:.6g}" for label, _sr, sa, _e in arms),
        f"- sigma ladder ({ladder['source']}, contract "
        "`oracle_headroom_gate.sigma_rel_ladder`): decision rungs "
        + (", ".join(f"{x:g}" for x in ladder["decision"]) or "none")
        + "; DIAGNOSTIC-ONLY rungs "
        + (", ".join(f"{x:g}" for x in ladder["diagnostic"]) or "none")
        + " — swept, reported and plotted, NEVER an input to the DECISION scan"
        + ("" if ladder["source"] == "contract" else
           "  **(LADDER OVERRIDDEN on the command line — this is NOT the "
           "contract ladder)**"),
        "",
        "## Spread over seeds (spread_rel = (cvar_noisy - cvar_oracle) / "
        "cvar_oracle)",
        "",
        "| arm | role | sigma_rel | sigma_gamma (NOMINAL) |"
        " sigma_gamma_effective | sigma_delta_effective | tc |"
        " spread_rel mean | seed std | CI-excl-0 frac | t_ex mean |"
        " clipped_frac |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        role = s["rung_role"].upper() if not s["decision_eligible"] else "decision"
        lines.append(
            f"| {s['arm']} | {role} | {s['sigma_rel']:.4g} | "
            f"{s['sigma_gamma_abs']:.4g} | "
            f"{s['sigma_gamma_effective']:.4g} | "
            f"{s['sigma_delta_effective']:.4g} | {s['tc']} | "
            f"{s['spread_rel_mean']:+.4f} | {s['spread_rel_seed_std']:.4f} | "
            f"{s['ci_excludes_zero_frac']:.2f} | {s['t_ex_mean']:.4f} | "
            f"{s.get('clipped_frac', float('nan')):.4f} |")
    lines += ["", _EFFECTIVE_NOTE, "", _CLIPPED_NOTE]
    if pilot_comparison is not None:
        pc = pilot_comparison
        lines += [
            "", "## Pilot point vs the DELIVERED corruption", "",
            f"- sigma_gamma_pilot (NOMINAL target): **{pc['nominal']:.6g}**",
            f"- compared against `{pc['compare_against']}` (contract "
            "`oracle_headroom_gate.effective_sigma_reporting."
            f"compare_pilot_against`): **{pc['effective']:.6g}**",
            f"- sigma_delta_effective (delta units, NOT the comparison target): "
            f"{pc['sigma_delta_effective']:.6g}",
            f"- clipped_frac on the pilot arm: {pc['clipped_frac']:.4f}",
        ]
    lines += [
        "",
        "## DECISION (per tc tier)",
        "",
        f"Smallest DECISION-ELIGIBLE sigma_rel with mean spread_rel >= "
        f"{spread_threshold_rel} (the",
        "pre-registered relative CVaR95 threshold, contract",
        "`oracle_headroom_gate.spread_threshold_rel`) AND the paired per-path",
        "bootstrap 95% CI excluding 0 in every seed. DIAGNOSTIC rungs ("
        + (", ".join(f"{x:g}" for x in ladder["diagnostic"]) or "none")
        + ") are EXCLUDED from this scan by the contract, however large their",
        "spread: above `region_of_validity.clipped_frac_max` the spread is no",
        "longer a monotone reading of a gamma error of the labelled size.",
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
        "Pilot-calibrated point once the pilot fit exists — read the float from",
        "the runlog rather than retyping it (`train.py --pilot` also prints a",
        "deliberately-reproduced PRE-FIX value, which must never reach the gate):",
        "",
        "    python gate_headroom.py --sigma-gamma-from-runlog <run>/runlog.json",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path

# ---------------------------------------------------------------------------
# pilot -> gate handoff
# ---------------------------------------------------------------------------


def sigma_gamma_pilot_from_runlog(path: str) -> float:
    """ABSOLUTE sigma_gamma_pilot out of a `train.py --pilot` runlog.json.

    The handoff used to be a human copy-paste off a two-line stdout whose FIRST
    line was the deliberately-reproduced pre-fix value — computed on an
    unconverged model, so typically LARGER, which would make the gate look more
    favourable than it is (audit T1). `sigma_gamma_pilot` is the best-step,
    gamma_ref value; this reads exactly that key and refuses anything else.
    """
    import json

    with open(path) as fh:
        log = json.load(fh)
    if "sigma_gamma_pilot" not in log:
        raise KeyError(
            f"{path!r} carries no 'sigma_gamma_pilot' — it is written only by "
            "`python train.py --pilot`. Do NOT substitute "
            "'sigma_gamma_pilot_prefix_bug': that is the known-wrong value.")
    sigma = float(log["sigma_gamma_pilot"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"{path!r} carries a non-usable sigma_gamma_pilot={sigma!r}")
    return sigma


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
    ap.add_argument("--sigma-rel", type=float, nargs="+", default=None,
                    help="OVERRIDE the pre-pilot relative sigma sweep; the "
                         "default is the CONTRACT ladder "
                         "(oracle_headroom_gate.sigma_rel_ladder.decision + "
                         ".diagnostic, AM2-3a)")
    ap.add_argument("--sigma-rel-diagnostic", type=float, nargs="+",
                    default=None,
                    help="rungs swept and reported but NEVER an input to the "
                         "DECISION scan (only meaningful with --sigma-rel; the "
                         "contract ladder brings its own)")
    ap.add_argument("--sigma-gamma", type=float, default=None,
                    help="ABSOLUTE pilot-calibrated sigma_gamma; replaces the "
                         "sweep with the single pilot point")
    ap.add_argument("--sigma-gamma-from-runlog", default=None,
                    help="PREFERRED over --sigma-gamma: path to a train.py "
                         "--pilot runlog.json; reads sigma_gamma_pilot from it, "
                         "so the pilot -> gate handoff is not a hand-typed float "
                         "off a multi-line stdout (audit T1)")
    ap.add_argument("--n-paths", type=int, default=None,
                    help="override engine simulation.n_paths")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="use only the first N derived seeds")
    ap.add_argument("--out-dir", default="results/gate")
    args = ap.parse_args(argv)
    sigma_gamma_abs = args.sigma_gamma
    if args.sigma_gamma_from_runlog is not None:
        if sigma_gamma_abs is not None:
            ap.error("pass --sigma-gamma OR --sigma-gamma-from-runlog, not both")
        sigma_gamma_abs = sigma_gamma_pilot_from_runlog(args.sigma_gamma_from_runlog)
        print(f"sigma_gamma_pilot read from {args.sigma_gamma_from_runlog}: "
              f"{sigma_gamma_abs:.6g}")
    cfg = hb.resolve_config()
    if args.n_paths is not None:
        cfg["engine"]["simulation"]["n_paths"] = int(args.n_paths)
    seed_list = (cfg["derived"]["seeds"][:args.n_seeds]
                 if args.n_seeds is not None else None)
    res = run_gate(cfg,
                   sigma_rel_list=(None if args.sigma_rel is None
                                   else tuple(args.sigma_rel)),
                   sigma_rel_diagnostic=(None if args.sigma_rel_diagnostic is None
                                         else tuple(args.sigma_rel_diagnostic)),
                   seed_list=seed_list, mode=args.mode, out_dir=args.out_dir,
                   sigma_gamma_abs=sigma_gamma_abs)
    print(f"rms(Gamma_oracle) = {res['rms_gamma']:.6g}")
    lad = res.get("ladder")
    if lad:
        print(f"sigma ladder ({lad['source']}): decision="
              f"{list(lad['decision'])}, diagnostic="
              f"{list(lad['diagnostic'])} (never a decision input)")
    print(f"wrote {res['csv_path']}")
    print(f"wrote {res['report_path']}")


if __name__ == "__main__":
    main()
