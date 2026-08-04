"""Tests for the oracle-headroom gate (gate_headroom.py).

Fast by construction (n_paths <= 128, 1 seed, bootstrap_B <= 25); NEVER runs
the full-size gate — that is a human-launched CLI after the pilot fit exists
(see gate_headroom module docstring). Covers: field determinism, analytic
d(eta)/dS calibration, the field-smoother-than-iid turnover property (the
design motivation, at the contract daily frequency), spread monotonicity in
sigma, report/csv emission, and the sigma=0 oracle-reproduction sanity.
Runnable via pytest or `python test_gate_headroom.py`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import Hedging_backtest as hb
import gate_headroom as gh
from oracle import HestonParams
from providers import HestonCFProvider

_DIR = Path(__file__).resolve().parent


def _cfg(n_paths=96, freq=252, n_boot=25, n_seeds=1):
    """Small resolved config for the gate tests (contract daily frequency by
    default; the field-vs-iid turnover property is anchored there)."""
    cfg = hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                            str(_DIR / "hedging_config.yaml"))
    eng = cfg["engine"]
    eng["simulation"]["n_paths"] = n_paths
    eng["rebalancing"]["frequency_per_year"] = freq
    eng["risk"]["bootstrap_B"] = n_boot
    cfg["derived"]["seeds"] = cfg["derived"]["seeds"][:n_seeds]
    return cfg


# module-level shared pieces (built once; the tests only read them)
_CFG = _cfg()
_BM = _CFG["benchmark"]
_REG = _BM["regimes"]["baseline"]
_THETA_TRAIN = HestonParams(kappa=_REG["kappa"], theta=_REG["theta"],
                            xi=_REG["xi"], rho=_REG["rho"], v0=_REG["v0"])
_BASE = HestonCFProvider(_THETA_TRAIN, _BM["grid"]["r"], _BM["grid"]["q"])
_RANGES = gh._grid_ranges(_CFG)
_CLOUD = gh.reference_state_cloud(_CFG, n_states=600, n_paths=96)
_RMS = gh.gamma_rms(_BASE, _CLOUD, 100.0)


def _probe():
    """A small (S, v, tau) probe grid off the reference cloud for pure-field
    determinism checks."""
    S = np.linspace(80.0, 120.0, 7)
    v = np.linspace(0.02, 0.10, 7)
    tau = 0.08
    return S, v, tau


def test_field_determinism():
    """Same (seed, mode, sigma) -> bit-identical field; different seed differs;
    evaluate() is a pure function of state (no advance between calls)."""
    S, v, tau = _probe()
    p1 = gh.NoisyOracleProvider(_BASE, 0.4 * _RMS, 5, _RANGES, _CLOUD, "field")
    p1b = gh.NoisyOracleProvider(_BASE, 0.4 * _RMS, 5, _RANGES, _CLOUD, "field")
    p2 = gh.NoisyOracleProvider(_BASE, 0.4 * _RMS, 6, _RANGES, _CLOUD, "field")
    e1 = p1.evaluate(S, v, tau, 100.0)["delta"]
    assert np.array_equal(e1, p1b.evaluate(S, v, tau, 100.0)["delta"])
    assert np.array_equal(e1, p1.evaluate(S, v, tau, 100.0)["delta"])  # pure
    assert not np.array_equal(e1, p2.evaluate(S, v, tau, 100.0)["delta"])
    # price and gamma pass through the wrapper unchanged
    got, ref = p1.evaluate(S, v, tau, 100.0), _BASE.evaluate(S, v, tau, 100.0)
    assert np.array_equal(got["price"], ref["price"])
    assert np.array_equal(got["gamma"], ref["gamma"])


def test_calibration_matches_target():
    """Measured std of the ANALYTIC d(eta)/dS on the reference cloud is within
    2% of sigma_gamma_target, for two targets."""
    for sr in (0.2, 0.6):
        target = sr * _RMS
        p = gh.NoisyOracleProvider(_BASE, target, 7, _RANGES, _CLOUD, "field")
        measured = float(np.std(p.eta_dS(*_CLOUD)))
        assert abs(measured - target) <= 0.02 * target


def test_field_tex_below_iid():
    """The design motivation: at the contract daily frequency, the smooth
    persistent field generates LESS excess turnover than iid delta noise of the
    SAME spatial amplitude (matched sigma_delta). Both t_ex are finite. Anchored
    at daily rebalancing, where the margin is large and robust across seeds;
    the property is marginal only at coarse (few-step) frequencies."""
    cfg = gh._trim_to_combined_cell(_cfg(n_paths=128, freq=252))
    target = 0.4 * _RMS
    for seed in (42, 5):
        pf = gh.NoisyOracleProvider(_BASE, target, seed, _RANGES, _CLOUD,
                                    "field")
        pi = gh.NoisyOracleProvider(_BASE, target, seed, _RANGES, _CLOUD,
                                    "iid")
        rows = hb.run_headline(cfg, {"oracle": _BASE, "field": pf, "iid": pi})
        by = {(r_["method"], r_["tc"]): r_ for r_ in rows}
        t_ex_field = by[("field", 0.01)]["t_ex"]
        t_ex_iid = by[("iid", 0.01)]["t_ex"]
        assert np.isfinite(t_ex_field) and np.isfinite(t_ex_iid)
        assert t_ex_field < t_ex_iid


def test_spread_monotone_in_sigma():
    """spread_rel (mean over seeds) is non-decreasing in sigma at tc=0.02;
    one small inversion tolerated at tiny n_paths."""
    cfg = _cfg(n_paths=128, freq=252)
    out = tempfile.mkdtemp()
    res = gh.run_gate(cfg, sigma_rel_list=(0.1, 0.2, 0.4, 0.8), mode="field",
                      out_dir=out, n_cloud_states=600, n_cloud_paths=96)
    at_tc = sorted((s for s in res["summary"] if s["tc"] == 0.02),
                   key=lambda s: s["sigma_rel"])
    spreads = [s["spread_rel_mean"] for s in at_tc]
    inversions = sum(1 for a, b in zip(spreads, spreads[1:]) if b < a - 1e-9)
    assert inversions <= 1, f"spread not monotone in sigma: {spreads}"
    assert spreads[-1] > spreads[0]     # net increase across the ladder


def test_report_and_csv_written():
    """run_gate emits headroom.csv + headroom_report.md with the expected
    records and a DECISION section."""
    cfg = _cfg(n_paths=96, freq=252)
    out = tempfile.mkdtemp()
    res = gh.run_gate(cfg, sigma_rel_list=(0.1, 0.4), mode="field",
                      out_dir=out, n_cloud_states=500, n_cloud_paths=96)
    assert Path(res["csv_path"]).exists()
    assert Path(res["report_path"]).exists()
    assert res["records"] and all(np.isfinite(r["spread_rel"])
                                  for r in res["records"])
    text = Path(res["report_path"]).read_text()
    assert "DECISION" in text and "HUMAN decision" in text
    assert "python gate_headroom.py" in text


def test_sigma_zero_reproduces_oracle():
    """amp=0 (sigma_gamma_target=0) reproduces the oracle rows exactly: field
    corruption vanishes, and the engine's paired diff vs oracle is exactly 0
    with zero excess turnover."""
    S, v, tau = _probe()
    p0 = gh.NoisyOracleProvider(_BASE, 0.0, 5, _RANGES, _CLOUD, "field")
    assert p0.amp == 0.0
    got, ref = p0.evaluate(S, v, tau, 100.0), _BASE.evaluate(S, v, tau, 100.0)
    assert np.array_equal(got["delta"], ref["delta"])
    # iid at sigma=0 also degenerates to the oracle (matched amplitude is 0)
    pi0 = gh.NoisyOracleProvider(_BASE, 0.0, 5, _RANGES, _CLOUD, "iid")
    assert pi0.sigma_delta == 0.0
    cfg = gh._trim_to_combined_cell(_cfg(n_paths=64, freq=252, n_boot=10))
    rows = hb.run_headline(cfg, {"oracle": _BASE, "noisy_z": p0})
    noisy = [r_ for r_ in rows if r_["method"] == "noisy_z"]
    assert noisy and all(r_["pnl_vs_oracle_cvar_diff"] == 0.0 for r_ in noisy)
    assert all(r_["t_ex"] == 0.0 for r_ in noisy)


def test_binding_delta_clip_is_reported():
    """G2: `amp` is calibrated on the UNCLIPPED field, and the [-0.05, 1.05]
    clip is applied afterwards. Where it binds, the delivered gamma error is
    smaller than sigma_gamma_target, so the measured spread is understated and
    the gate is conservative — safe direction, wrong axis label. The clip is
    NOT changed; it is made visible.

    Deep-ITM/OTM states sit at delta ~ 1 and ~ 0, exactly where the bounds are.
    """
    S = np.concatenate([np.linspace(40.0, 60.0, 40),        # deep OTM: delta ~ 0
                        np.linspace(160.0, 200.0, 40)])     # deep ITM: delta ~ 1
    v = np.full(S.shape, 0.06)
    tau = 0.10

    p = gh.NoisyOracleProvider(_BASE, 3.0 * _RMS, 5, _RANGES, _CLOUD, "field")
    assert p.clipped_fraction != p.clipped_fraction          # NaN before any eval
    p.evaluate(S, v, tau, 100.0)
    assert p.clipped_fraction > 0.0, (
        "a binding clip must be visible: the delivered corruption is weaker "
        "than the sigma_gamma the gate table is labelled with")
    assert 0.0 < p.clipped_fraction <= 1.0

    # sigma = 0 delivers exactly the oracle -> nothing can clip
    p0 = gh.NoisyOracleProvider(_BASE, 0.0, 5, _RANGES, _CLOUD, "field")
    p0.evaluate(S, v, tau, 100.0)
    assert p0.clipped_fraction == 0.0


def test_clipped_fraction_reaches_the_summary_and_report():
    """The gate's own summary rows and headroom_report.md must carry it — the
    audit's one unverifiable item was 'nothing measures or reports the clipped
    fraction'."""
    cfg = _cfg(n_paths=96, freq=252)
    out = tempfile.mkdtemp()
    res = gh.run_gate(cfg, sigma_rel_list=(0.1, 0.4), mode="field",
                      out_dir=out, n_cloud_states=500, n_cloud_paths=96)
    assert all("clipped_frac" in s for s in res["summary"])
    assert all(0.0 <= s["clipped_frac"] <= 1.0 for s in res["summary"])
    # per ARM, not per tier: positions are built once per method, never per tc
    by_arm = {}
    for s in res["summary"]:
        by_arm.setdefault(s["arm"], set()).add(s["clipped_frac"])
    assert all(len(v) == 1 for v in by_arm.values())
    assert "clipped_frac" in res["clipped_fraction_note"]
    text = Path(res["report_path"]).read_text()
    assert "clipped" in text


def test_sigma_ladder_comes_from_the_contract():
    """AM2-3a: the swept ladder is `oracle_headroom_gate.sigma_rel_ladder`, not a
    Python literal. The module must carry NO default tuple, and the resolver must
    return the contract's decision + diagnostic rungs with their eligibility."""
    assert not hasattr(gh, "_SIGMA_REL_DEFAULT"), (
        "the sigma ladder is a contract key (AM2-3a); no Python literal may "
        "shadow it")
    lad = _BM["oracle_headroom_gate"]["sigma_rel_ladder"]
    entries, source = gh._resolve_ladder(_CFG, None, None)
    assert source == "contract"
    assert [sr for sr, elig in entries if elig] == [float(x) for x in lad["decision"]]
    assert [sr for sr, elig in entries if not elig] == [float(x) for x in lad["diagnostic"]]
    # the deleted rung really is gone from what a default invocation executes
    assert 0.8 not in [sr for sr, _ in entries]


def test_diagnostic_rung_cannot_fire_the_decision_scan():
    """AM2-3a's whole point: a rung the contract declares DIAGNOSTIC must never be
    selected by the DECISION scan, even when it clears the spread threshold and
    every seed's CI excludes 0 — which the huge-sigma rung below does.

    The ladder is trimmed IN MEMORY (the contract file is untouched) exactly as
    _trim_to_combined_cell trims the sweep geometry."""
    cfg = _cfg(n_paths=96, freq=252)
    cfg["benchmark"]["oracle_headroom_gate"]["sigma_rel_ladder"] = {
        "decision": [0.05], "diagnostic": [0.8]}
    out = tempfile.mkdtemp()
    res = gh.run_gate(cfg, mode="field", out_dir=out, n_cloud_states=400,
                      n_cloud_paths=96, spread_threshold_rel=-1e9)

    # the executed ladder IS the contract's (0.2/0.4 of the old literal are absent)
    assert {s["arm"] for s in res["summary"]} == {"s0.05", "s0.8"}
    assert res["ladder"] == {"decision": (0.05,), "diagnostic": (0.8,),
                             "source": "contract"}

    diag = [s for s in res["summary"] if s["arm"] == "s0.8"]
    assert diag and all(s["decision_eligible"] is False for s in diag)
    assert all(s["rung_role"] == "diagnostic" for s in diag)
    for s in diag:                      # it WOULD have fired the scan's predicate
        assert (s["spread_rel_mean"] >= res["spread_threshold_rel"]
                and s["ci_excludes_zero_frac"] == 1.0)
    for tc, hit in res["decision"].items():
        assert hit is None or hit["arm"] != "s0.8", (
            f"tc={tc}: a DIAGNOSTIC rung fired the decision scan")

    # ... and the label is visible where a human reads it
    csv_text = Path(res["csv_path"]).read_text()
    assert "rung_role" in csv_text.splitlines()[0] and "diagnostic" in csv_text
    report = Path(res["report_path"]).read_text()
    assert "DIAGNOSTIC" in report and "0.8" in report


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"PASS {fn}")
