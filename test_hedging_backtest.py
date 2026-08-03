"""Tests: PnL-decomposition-sums-to-total, no-look-ahead, Bates lambda_j=0
Heston recovery, seed determinism, CVaR unit check, premium-convention
override, T' terminal-mark pricers (Bates CF / Merton series), expiry bridge,
contract-target lock, tiny end-to-end smoke.
Runnable via pytest or `python test_hedging_backtest.py`."""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import norm

import Hedging_backtest as hb
import oracle

_DIR = Path(__file__).resolve().parent


class BSProvider:
    """Black-Scholes (price, Greeks) test double with sigma = sqrt(v);
    exercises the GreekProvider interface without touching oracle/models.
    v floored at 1e-4 (QE paths can hit v = 0 exactly) to keep Greeks finite,
    as the provider contract requires."""

    def __init__(self, r: float, q: float):
        self.r, self.q = r, q

    def evaluate(self, S, v, tau, K):
        S, v = np.asarray(S, float), np.maximum(np.asarray(v, float), 1e-4)
        srt = np.sqrt(v * tau)
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * v) * tau) / srt
        d2 = d1 - srt
        dq, dr = np.exp(-self.q * tau), np.exp(-self.r * tau)
        return {"price": S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2),
                "delta": dq * norm.cdf(d1),
                "gamma": dq * norm.pdf(d1) / (S * srt)}


class ShiftedBSProvider(BSProvider):
    """BSProvider clone that overprices by +0.5 with IDENTICAL Greeks — the
    premium-convention probe: without the oracle-premium override this arm
    banks a uniform 0.5*e^{rT'} PnL shift while hedging identically."""

    def evaluate(self, S, v, tau, K):
        out = super().evaluate(S, v, tau, K)
        return {**out, "price": out["price"] + 0.5}


def _cfg(n_paths=128, freq=52, n_boot=40, n_seeds=1):
    cfg = hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                            str(_DIR / "hedging_config.yaml"))
    eng = cfg["engine"]
    eng["simulation"]["n_paths"] = n_paths
    eng["rebalancing"]["frequency_per_year"] = freq
    eng["risk"]["bootstrap_B"] = n_boot
    eng["misspecification"]["magnitudes"] = [0.0, 1.0]
    cfg["derived"]["seeds"] = cfg["derived"]["seeds"][:n_seeds]
    return cfg


def _trim_to_one_cell(cfg, direction="xi_up", magnitude=1.0):
    """In-memory sweep trim (contract file untouched): one perturbation cell,
    no cross-model sweeps — keeps engine-level tests fast."""
    mis = cfg["benchmark"]["hedging_simulation"]["misspecification"]
    mis["perturbations"] = {direction: mis["perturbations"][direction]}
    mis["cross_model"] = []
    cfg["engine"]["misspecification"]["magnitudes"] = [magnitude]
    return cfg


def _setup(cfg, seed=42):
    """Simulate paths to full expiry spec['T'] for settlement-mechanics tests
    (the pipeline itself simulates to horizon.T_prime; see smoke tests)."""
    bm, eng = cfg["benchmark"], cfg["engine"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    p = hb.SimParams.from_regime(bm["regimes"]["baseline"], r, q)
    spec = eng["contract"]
    n_steps = int(round(spec["T"] * eng["rebalancing"]["frequency_per_year"]))
    times, S, v = hb.simulate_heston_qe(p, spec["S0"], spec["T"], n_steps,
                                        eng["simulation"]["n_paths"], seed,
                                        eng["simulation"]["psi_c"])
    return p, spec, r, q, times, S, v


def _intrinsic(S, K):
    return np.maximum(S[:, -1] - K, 0.0)


def test_pnl_decomposition_sums_to_total():
    cfg = _cfg()
    _, spec, r, q, times, S, v = _setup(cfg)
    prov = BSProvider(r, q)
    tiers = cfg["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"]
    liab = _intrinsic(S, spec["K"])
    pos, prem = hb.delta_positions(S, v, times, prov, spec["K"], spec["T"])
    for tc in tiers:
        res = hb.settle_delta(S, times, pos, r, q, tc, prem, liab)
        assert np.allclose(res.pnl_total, res.pnl_directional - res.tc_paid_fv,
                           rtol=0.0, atol=1e-8)
        if tc == 0.0:
            assert np.all(res.tc_paid_fv == 0.0)
            assert np.array_equal(res.pnl_total, res.pnl_directional)
    ho = cfg["engine"]["hedge_option"]
    u, n_opt, ph, prem2 = hb.dg_positions(S, v, times, prov, spec["K"],
                                          spec["T"], spec["S0"], 2 * spec["T"],
                                          ho["min_tau"], ho["gamma_floor"])
    for tc in tiers:
        res = hb.settle_delta_gamma(S, ph, times, u, n_opt, r, q, tc, prem2,
                                    liab)
        assert np.allclose(res.pnl_total, res.pnl_directional - res.tc_paid_fv,
                           rtol=0.0, atol=1e-8)


def test_no_look_ahead():
    cfg = _cfg()
    _, spec, r, q, times, S, v = _setup(cfg)
    prov = BSProvider(r, q)
    k = (S.shape[1] - 1) // 2
    S2, v2 = S.copy(), v.copy()
    S2[:, k + 1:] *= 1.07
    v2[:, k + 1:] *= 1.30
    pos1, _ = hb.delta_positions(S, v, times, prov, spec["K"], spec["T"])
    pos2, _ = hb.delta_positions(S2, v2, times, prov, spec["K"], spec["T"])
    assert np.array_equal(pos1[:, :k + 1], pos2[:, :k + 1])
    assert not np.allclose(pos1[:, k + 1], pos2[:, k + 1])
    ho = cfg["engine"]["hedge_option"]
    u1, n1, _, _ = hb.dg_positions(S, v, times, prov, spec["K"], spec["T"],
                                   spec["S0"], 2 * spec["T"],
                                   ho["min_tau"], ho["gamma_floor"])
    u2, n2, _, _ = hb.dg_positions(S2, v2, times, prov, spec["K"], spec["T"],
                                   spec["S0"], 2 * spec["T"],
                                   ho["min_tau"], ho["gamma_floor"])
    assert np.array_equal(u1[:, :k + 1], u2[:, :k + 1])
    assert np.array_equal(n1[:, :k + 1], n2[:, :k + 1])


def test_bates_lambda0_recovers_heston():
    cfg = _cfg()
    p, spec, _, _, _, S, v = _setup(cfg)
    eng = cfg["engine"]
    n_steps = int(round(spec["T"] * eng["rebalancing"]["frequency_per_year"]))
    args = (spec["S0"], spec["T"], n_steps, eng["simulation"]["n_paths"], 42,
            eng["simulation"]["psi_c"])
    p0 = dataclasses.replace(p, lambda_j=0.0, mu_j=-0.10, sigma_j=0.10)
    _, S0_, v0_ = hb.simulate_heston_qe(p0, *args)
    assert np.array_equal(S, S0_) and np.array_equal(v, v0_)
    pj = dataclasses.replace(p, lambda_j=0.25, mu_j=-0.10, sigma_j=0.10)
    _, Sj, vj = hb.simulate_heston_qe(pj, *args)
    assert not np.array_equal(S, Sj)
    assert np.array_equal(v, vj)          # jumps touch S only


def test_seed_determinism():
    cfg = _cfg()

    def run(seed):
        _, spec, r, q, times, S, v = _setup(cfg, seed=seed)
        prov = BSProvider(r, q)
        pos, prem = hb.delta_positions(S, v, times, prov, spec["K"], spec["T"])
        res = hb.settle_delta(S, times, pos, r, q, 0.01, prem,
                              _intrinsic(S, spec["K"]))
        se = hb.bootstrap_cvar_se(res.pnl_total, 0.95, 40, seed)
        return res.pnl_total, se

    pnl_a, se_a = run(42)
    pnl_b, se_b = run(42)
    assert np.array_equal(pnl_a, pnl_b) and se_a == se_b
    pnl_c, _ = run(43)
    assert not np.array_equal(pnl_a, pnl_c)


def test_cvar_unit():
    pnl = -np.arange(1.0, 101.0)          # losses 1..100
    assert hb.cvar(pnl, 0.95) == 98.0     # mean of worst 5: 96..100
    se1 = hb.bootstrap_cvar_se(pnl, 0.95, 100, 7)
    se2 = hb.bootstrap_cvar_se(pnl, 0.95, 100, 7)
    assert se1 == se2 > 0.0


def test_premium_override_neutralizes_price_shift():
    """FIX 1: two clones differing ONLY by a +0.5 price shift (same Greeks)
    give identical pnl_total under premium_override and a uniform
    0.5*e^{rT'} shift without it."""
    cfg = _cfg(n_paths=64, freq=26)
    _, spec, r, q, times, S, v = _setup(cfg)
    base, shifted = BSProvider(r, q), ShiftedBSProvider(r, q)
    liab = _intrinsic(S, spec["K"])
    pos_a, prem_a = hb.delta_positions(S, v, times, base, spec["K"], spec["T"])
    pos_b, prem_b = hb.delta_positions(S, v, times, shifted, spec["K"],
                                       spec["T"])
    assert np.array_equal(pos_a, pos_b)              # Greeks identical
    assert np.allclose(prem_b - prem_a, 0.5)
    res_a = hb.settle_delta(S, times, pos_a, r, q, 0.01, prem_a, liab)
    res_b = hb.settle_delta(S, times, pos_b, r, q, 0.01, prem_b, liab)
    shift = res_b.pnl_total - res_a.pnl_total
    assert np.allclose(shift, 0.5 * np.exp(r * times[-1]))   # the artefact
    pos_bo, prem_bo = hb.delta_positions(S, v, times, shifted, spec["K"],
                                         spec["T"], premium_override=prem_a)
    assert np.array_equal(pos_bo, pos_b) and np.array_equal(prem_bo, prem_a)
    res_bo = hb.settle_delta(S, times, pos_bo, r, q, 0.01, prem_bo, liab)
    assert np.array_equal(res_bo.pnl_total, res_a.pnl_total)
    # engine level: with the oracle provider present, the overpricing clone's
    # rows are IDENTICAL to the oracle's (same paths, same premium, same deltas)
    cfg2 = _trim_to_one_cell(_cfg(n_paths=32, freq=26, n_boot=10))
    rows = hb.run_headline(cfg2, {"oracle": base, "shifted": shifted})
    assert rows and all(row["premium_convention_ok"] is True for row in rows)
    by = {(row["method"], row["tc"]): row for row in rows}
    for tc in (0.0, 0.01, 0.02):
        assert by[("shifted", tc)]["mean_pnl"] == by[("oracle", tc)]["mean_pnl"]
        assert by[("shifted", tc)]["cvar"] == by[("oracle", tc)]["cvar"]


def test_terminal_mark_bates_lambda0_equals_heston_bitwise():
    """FIX 2: Bates DGP with lambda_j = 0 must mark the liability bit-for-bit
    like the pure-Heston DGP on the same liquidation states."""
    cfg = _cfg(n_paths=32, freq=26)
    p, spec, _, _, _, S, v = _setup(cfg)
    tau = spec["T"] - cfg["engine"]["horizon"]["T_prime"]
    bates0 = dataclasses.replace(p, lambda_j=0.0, mu_j=-0.10, sigma_j=0.10)
    m_h = hb.heston_bates_terminal_mark(S[:, -1], v[:, -1], tau, spec["K"], p)
    m_b = hb.heston_bates_terminal_mark(S[:, -1], v[:, -1], tau, spec["K"],
                                        bates0)
    assert np.array_equal(m_h, m_b)
    assert np.all(np.isfinite(m_h)) and np.all(m_h >= 0.0)
    bates = dataclasses.replace(p, lambda_j=0.25, mu_j=-0.10, sigma_j=0.10)
    m_j = hb.heston_bates_terminal_mark(S[:, -1], v[:, -1], tau, spec["K"],
                                        bates)
    assert not np.array_equal(m_h, m_j)   # jumps must actually reprice


def test_bates_price_cf_validations():
    """oracle.bates_price_cf: lambda_j=0 reproduces heston_greeks_cf price to
    1e-8 rel; per-point v0 vectorization matches scalar-param evaluation;
    xi -> 0 with v0 = theta = sigma^2 reproduces the Merton closed form."""
    r, q = 0.02, 0.0
    p = oracle.HestonParams(kappa=2.0, theta=0.04, xi=0.30, rho=-0.50, v0=0.04)
    S = np.tile([90.0, 100.0, 110.0], 3)
    K = np.full_like(S, 100.0)
    T = np.repeat([0.08, 0.25, 1.0], 3)
    ref = oracle.heston_greeks_cf(S, K, T, p, r, q).price
    got = oracle.bates_price_cf(S, K, T, p, r, q,
                                lambda_j=0.0, mu_j=-0.10, sigma_j=0.10)
    assert float(np.max(np.abs(got - ref) / np.abs(ref))) < 1e-8
    # per-point v0: vectorized call == one scalar-param call per point
    v0s = np.array([0.01, 0.03, 0.05, 0.08, 0.12])
    S5 = np.array([85.0, 95.0, 100.0, 105.0, 120.0])
    vec = oracle.bates_price_cf(S5, 100.0, 0.25, p, r, q,
                                lambda_j=0.25, mu_j=-0.10, sigma_j=0.10,
                                v0=v0s)
    for i in range(5):
        pi = oracle.HestonParams(p.kappa, p.theta, p.xi, p.rho, float(v0s[i]))
        ref_i = float(oracle.bates_price_cf(S5[i], 100.0, 0.25, pi, r, q,
                                            lambda_j=0.25, mu_j=-0.10,
                                            sigma_j=0.10))
        assert abs(vec[i] - ref_i) / ref_i < 1e-8
    # Merton limit: xi -> 0 with v0 = theta = sigma^2 is GBM + jumps
    pm = oracle.HestonParams(kappa=2.0, theta=0.04, xi=1e-4, rho=0.0, v0=0.04)
    S3 = np.array([90.0, 100.0, 110.0])
    bp = oracle.bates_price_cf(S3, 100.0, 0.5, pm, r, q,
                               lambda_j=0.25, mu_j=-0.10, sigma_j=0.10)
    mp = oracle.merton_price(S3, 100.0, 0.5, 0.20, 0.25, -0.10, 0.10, r, q)
    assert float(np.max(np.abs(bp - mp) / mp)) < 1e-3


def test_merton_price_vs_mc():
    """oracle.merton_price against a Monte-Carlo estimate from the engine's own
    Merton simulator (exact in distribution per step)."""
    r, q = 0.02, 0.0
    mp = {"sigma": 0.20, "lambda_j": 0.25, "mu_j": -0.10, "sigma_j": 0.10}
    _, S, _ = hb.simulate_merton(mp, 100.0, 0.25, 8, 200_000, 7, r, q)
    disc = np.exp(-r * 0.25) * np.maximum(S[:, -1] - 100.0, 0.0)
    mc, se = float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(disc.size))
    ref = float(oracle.merton_price(100.0, 100.0, 0.25, mp["sigma"],
                                    mp["lambda_j"], mp["mu_j"], mp["sigma_j"],
                                    r, q))
    assert abs(ref - mc) < 4.0 * se
    assert float(oracle.merton_price(100.0, 100.0, 0.25, 0.20, 0.0, 0.0, 0.0,
                                     r, q)) > 0.0   # lambda_j=0 -> BS branch


def test_expiry_bridge_reduces_to_old_behavior():
    """FIX 2 regression bridge: T_prime == tau0 with an intrinsic terminal mark
    reproduces the pre-P5 expiry-settlement numbers exactly."""
    cfg = _trim_to_one_cell(_cfg(n_paths=48, freq=26, n_boot=10))
    eng = cfg["engine"]
    eng["horizon"]["T_prime"] = eng["contract"]["T"]     # T' == tau0
    bm = cfg["benchmark"]
    r, q = bm["grid"]["r"], bm["grid"]["q"]
    prov = BSProvider(r, q)
    rows = hb.run_headline(cfg, {"bs": prov})
    row = next(row for row in rows if row["method"] == "bs"
               and row["tc"] == 0.01)
    assert row["premium_convention_ok"] is False        # no oracle provider
    # old behavior, replicated by hand: expiry grid, intrinsic payoff,
    # per-provider premium
    spec = eng["contract"]
    base = hb.SimParams.from_regime(bm["regimes"]["baseline"], r, q)
    hp = hb.perturb_params(base, "xi_up", 1.0,
                           eng["misspecification"]["directions"])
    n_steps = int(round(spec["T"] * eng["rebalancing"]["frequency_per_year"]))
    times, S, v = hb.simulate_heston_qe(hp, spec["S0"], spec["T"], n_steps,
                                        eng["simulation"]["n_paths"],
                                        cfg["derived"]["seeds"][0],
                                        eng["simulation"]["psi_c"])
    pos, prem = hb.delta_positions(S, v, times, prov, spec["K"], spec["T"])
    res = hb.settle_delta(S, times, pos, r, q, 0.01, prem,
                          _intrinsic(S, spec["K"]))
    assert row["mean_pnl"] == float(res.pnl_total.mean())
    assert row["cvar"] == hb.cvar(res.pnl_total, row["cvar_level"])


def test_resolve_config_rejects_contract_drift():
    """FIX 3: a shift_at_m1 that no longer lands on the contract perturbation
    target must fail resolve_config loudly."""
    real_load = hb._load_yaml

    def drifted(path):
        d = real_load(path)
        if "hedging_config" in str(path):
            d["misspecification"]["directions"]["xi_up"]["shift_at_m1"] = 0.10
        return d

    hb._load_yaml = drifted
    try:
        raised = False
        try:
            hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                              str(_DIR / "hedging_config.yaml"))
        except AssertionError as err:
            raised = True
            assert "drifted" in str(err)
        assert raised, "resolve_config accepted a drifted shift_at_m1"
    finally:
        hb._load_yaml = real_load
    # the real pair must still pass, composite legs included
    cfg = hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                            str(_DIR / "hedging_config.yaml"))
    assert cfg["engine"]["misspecification"]["directions"]["combined"]["legs"] \
        == ["xi_up", "rho_down"]


def test_run_headline_smoke():
    cfg = _cfg(n_paths=64, freq=26, n_boot=25)
    cfg["engine"]["smoothing_baseline"]["applies_to"] = ["bs"]
    r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
    out_dir = tempfile.mkdtemp()
    rows = hb.run_headline(cfg, {"bs": BSProvider(r, q)}, out_dir=out_dir)
    assert rows and all(np.isfinite(row["mean_pnl"]) and
                        np.isfinite(row["cvar"]) for row in rows)
    for row in rows:
        assert abs(row["mean_pnl"] -
                   (row["directional_component"] - row["tc_component"])) < 1e-8
        if row["tc"] == 0.0:
            assert row["tc_component"] == 0.0
        assert row["premium_convention_ok"] is False   # no oracle provider
    key = lambda row: (row["sweep"], row["direction"], row["magnitude"],
                       row["lambda_j"], row["sigma_j"], row["method"])
    by_setting = {}
    for row in rows:
        by_setting.setdefault(key(row), []).append(row)
    for grp in by_setting.values():
        d_vals = {row["directional_component"] for row in grp}
        assert len(d_vals) == 1           # directional leg independent of tc
        tcs = sorted(grp, key=lambda row: row["tc"])
        assert all(a["tc_component"] <= b["tc_component"]
                   for a, b in zip(tcs, tcs[1:]))
    raw = [row for row in rows if row["method"] == "bs"]
    smo = [row for row in rows if row["method"] == "bs_smoothed"]
    assert smo and np.mean([row["turnover"] for row in smo]) <= \
        np.mean([row["turnover"] for row in raw])
    for name in ("headline_delta_only_per_seed.csv",
                 "headline_delta_only_agg.csv", "resolved_config.yaml"):
        assert (Path(out_dir) / name).exists()


def test_run_secondary_smoke():
    cfg = _cfg(n_paths=64, freq=26, n_boot=25)
    r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
    out_dir = tempfile.mkdtemp()
    rows = hb.run_secondary_delta_gamma(cfg, {"bs": BSProvider(r, q)},
                                        out_dir=out_dir)
    assert rows and all(np.isfinite(row["cvar"]) for row in rows)
    for row in rows:
        assert abs(row["mean_pnl"] -
                   (row["directional_component"] - row["tc_component"])) < 1e-8
        assert row["turnover"] >= 0.0     # option-leg turnover
    assert (Path(out_dir) / "secondary_delta_gamma_per_seed.csv").exists()


def test_secondary_hedge_maturity_both_sides():
    metric_keys = ("mean_pnl", "cvar", "turnover", "tc_component",
                   "directional_component")
    for factor in (1.5, 0.75):            # T_h > T, and T_h = 0.1875 < T = 0.25
        cfg = _cfg(n_paths=64, freq=26, n_boot=25)
        eng = cfg["engine"]
        eng["hedge_option"]["maturity_factor"] = factor
        _, spec, r, q, times, S, v = _setup(cfg)
        prov = BSProvider(r, q)
        T_h = factor * spec["T"]
        u, n_opt, ph, prem = hb.dg_positions(
            S, v, times, prov, spec["K"], spec["T"], spec["S0"], T_h,
            eng["hedge_option"]["min_tau"], eng["hedge_option"]["gamma_floor"])
        assert np.isfinite(u).all() and np.isfinite(n_opt).all() \
            and np.isfinite(ph).all()
        expired = np.asarray([T_h - t_next <= 0.0 for t_next in times[1:]])
        assert np.all(n_opt[:, expired] == 0.0)   # never held past own expiry
        res = hb.settle_delta_gamma(S, ph, times, u, n_opt, r, q, 0.02, prem,
                                    _intrinsic(S, spec["K"]))
        assert np.isfinite(res.pnl_total).all()
        rows = hb.run_secondary_delta_gamma(cfg, {"bs": prov})
        assert rows and all(np.isfinite(row[k]) for row in rows
                            for k in metric_keys)


def test_resolve_config_exposes_p5_staging_and_tc_tiers():
    cfg = hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                            str(_DIR / "hedging_config.yaml"))
    eng = cfg["engine"]
    assert eng["horizon"]["T_prime"] == 0.17
    dirs = eng["misspecification"]["directions"]
    assert dirs["xi_up"]["shift_at_m1"] == 0.15
    assert dirs["rho_down"]["shift_at_m1"] == -0.30
    assert dirs["combined"]["legs"] == ["xi_up", "rho_down"]
    tiers = cfg["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"]
    assert tiers == [0.0, 0.01, 0.02]


def test_paired_bootstrap_cvar_diff():
    """FIX 1: constant-shift pairs give the exact diff with a degenerate CI
    (the common path term cancels replicate by replicate); zero shift gives a
    CI containing 0; results are deterministic in the seed."""
    rng = np.random.default_rng(0)
    pnl_b = rng.normal(0.0, 1.0, 4000)
    pnl_a = pnl_b + 2.0                   # cvar(a) = cvar(b) - 2 exactly
    st = hb.paired_bootstrap_cvar_diff(pnl_a, pnl_b, 0.95, 200, 7)
    assert np.isclose(st["diff"], -2.0)
    assert st["ci_hi"] < 0.0              # large shift: CI excludes 0
    # pairing cancels the common term EXACTLY: every replicate diff is -2
    assert abs(st["ci_hi"] - st["ci_lo"]) < 1e-9
    cb, ca = hb.cvar(pnl_b, 0.95), hb.cvar(pnl_a, 0.95)
    assert np.isclose(st["rel_improvement"], (cb - ca) / cb)
    # zero shift: diff exactly 0 and CI contains 0
    st0 = hb.paired_bootstrap_cvar_diff(pnl_b, pnl_b.copy(), 0.95, 200, 7)
    assert st0["diff"] == 0.0
    assert st0["ci_lo"] <= 0.0 <= st0["ci_hi"]
    # determinism from seed (non-degenerate pair so the CI has width)
    pnl_c = pnl_b + 1.0 + rng.normal(0.0, 0.5, 4000)
    s1 = hb.paired_bootstrap_cvar_diff(pnl_c, pnl_b, 0.95, 200, 7)
    s2 = hb.paired_bootstrap_cvar_diff(pnl_c, pnl_b, 0.95, 200, 7)
    assert s1 == s2
    s3 = hb.paired_bootstrap_cvar_diff(pnl_c, pnl_b, 0.95, 200, 8)
    assert s3["diff"] == s1["diff"]       # point estimate is seed-free
    assert (s3["ci_lo"], s3["ci_hi"]) != (s1["ci_lo"], s1["ci_hi"])


def test_paired_columns_blank_on_reference_rows():
    """FIX 1 engine level: vs_baseline columns blank on the baseline's own
    rows and numeric elsewhere (same for vs_oracle); all blank when neither
    reference provider is present; hidden _gap columns never reach the CSVs."""
    cfg = _trim_to_one_cell(_cfg(n_paths=32, freq=26, n_boot=50))
    r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
    out_dir = tempfile.mkdtemp()
    rows = hb.run_headline(cfg, {"oracle": BSProvider(r, q),
                                 "standard_pinn": ShiftedBSProvider(r, q)},
                           out_dir=out_dir)
    assert {row["method"] for row in rows} == \
        {"oracle", "standard_pinn", "standard_pinn_smoothed"}
    for row in rows:
        assert row["t_ex_definition"] == "sum_incl_endpoints"
        if row["method"] == "standard_pinn":          # the baseline itself
            assert row["pnl_vs_baseline_cvar_diff"] == ""
            assert row["pnl_vs_baseline_ci_lo"] == ""
            assert row["pnl_vs_baseline_ci_hi"] == ""
            # identical Greeks to the oracle -> exactly zero paired diff
            assert row["pnl_vs_oracle_cvar_diff"] == 0.0
            assert row["t_ex"] == 0.0                 # same trades as oracle
        elif row["method"] == "oracle":
            assert row["pnl_vs_oracle_cvar_diff"] == ""
            assert row["pnl_vs_oracle_ci_lo"] == ""
            assert row["pnl_vs_oracle_ci_hi"] == ""
            assert isinstance(row["pnl_vs_baseline_cvar_diff"], float)
            assert row["t_ex"] == 0.0                 # oracle vs itself
        else:                                         # smoothed variant
            for prefix in ("pnl_vs_baseline", "pnl_vs_oracle"):
                d = row[f"{prefix}_cvar_diff"]
                lo, hi = row[f"{prefix}_ci_lo"], row[f"{prefix}_ci_hi"]
                assert isinstance(d, float)
                assert lo <= d <= hi
    # falsifier-relevant columns present in the per-seed CSV; hidden ones not
    header = (Path(out_dir) / "headline_delta_only_per_seed.csv") \
        .read_text().splitlines()[0]
    for col in ("t_ex", "t_ex_definition", "pnl_vs_baseline_cvar_diff",
                "pnl_vs_oracle_ci_hi"):
        assert col in header
    assert "_gap_num" not in header and "_cvar_oracle" not in header
    agg_header = (Path(out_dir) / "headline_delta_only_agg.csv") \
        .read_text().splitlines()[0]
    assert "gap_closed_mean" in agg_header
    assert "gap_closed_mean_of_ratios" in agg_header
    # neither reference present -> every paired column blank
    cfg2 = _trim_to_one_cell(_cfg(n_paths=16, freq=26, n_boot=5))
    rows2 = hb.run_headline(cfg2, {"bs": BSProvider(r, q)})
    assert all(row["pnl_vs_baseline_cvar_diff"] == "" and
               row["pnl_vs_oracle_cvar_diff"] == "" for row in rows2)


def _mk_gap_row(method, cvar_val, seed):
    """Minimal per-seed row for gap_closed aggregation tests."""
    return {"sweep": "perturbation", "direction": "xi_up", "magnitude": 1.0,
            "lambda_j": "", "sigma_j": "", "in_model": False,
            "method": method, "tc": 0.01, "seed": seed,
            "mean_pnl": 0.0, "cvar": cvar_val, "cvar_level": 0.95,
            "cvar_boot_se": 0.1, "turnover": 0.1, "tc_component": 0.0,
            "directional_component": 0.0, "n_paths": 8,
            "t_ex": 0.0, "t_ex_definition": "sum_incl_endpoints",
            "premium_convention_ok": True}


def test_gap_closed_ratio_of_means_bounded():
    """FIX 2: a seed with a near-zero baseline-oracle gap blows up the legacy
    mean-of-ratios aggregate but leaves the ratio-of-mean-gaps bounded; and
    the gap_floor_frac floor (fraction of |cvar_oracle|) blanks per-seed
    ratios that the old absolute 1e-6 floor let explode."""
    rows = []
    for seed, (c_base, c_orac, c_sob) in enumerate(
            [(10.0, 5.0, 6.0),            # healthy gap: ratio 0.8
             (5.06, 5.0, 4.0)]):          # near-zero gap: ratio ~ 17.7
        rows += [_mk_gap_row("standard_pinn", c_base, seed),
                 _mk_gap_row("oracle", c_orac, seed),
                 _mk_gap_row("sobolev", c_sob, seed)]
    rows = hb.add_gap_closed(rows, "standard_pinn", "oracle", 0.01)
    per_seed = {(r_["method"], r_["seed"]): r_ for r_ in rows}
    assert np.isclose(per_seed[("sobolev", 0)]["gap_closed"], 0.8)
    assert per_seed[("sobolev", 1)]["gap_closed"] > 15.0   # the pathology
    agg = hb.aggregate_over_seeds(rows, 0.01)
    sob = next(r_ for r_ in agg if r_["method"] == "sobolev")
    assert sob["n_seeds"] == 2
    assert sob["gap_closed_mean_of_ratios"] > 5.0          # legacy blows up
    assert 0.9 < sob["gap_closed_mean"] < 1.1              # ratio of means
    # denominator below the scale-aware floor -> per-seed ratio undefined
    rows_f = [_mk_gap_row("standard_pinn", 5.02, 0),
              _mk_gap_row("oracle", 5.0, 0),
              _mk_gap_row("sobolev", 4.0, 0)]
    rows_f = hb.add_gap_closed(rows_f, "standard_pinn", "oracle", 0.01)
    assert all(r_["gap_closed"] == "" for r_ in rows_f)    # 0.02 < 0.05 floor
    agg_f = hb.aggregate_over_seeds(rows_f, 0.01)
    assert all(r_["gap_closed_mean"] == "" and
               r_["gap_closed_mean_of_ratios"] == "" for r_ in agg_f)


def test_total_traded_t_ex_hand_example():
    """FIX 3: hand-computed 2-path example including the inception trade and
    final unwind — the quantity tc is charged on at unit price weighting."""
    pos = np.array([[0.5, 0.7, 0.4],
                    [0.2, 0.2, 0.6]])
    # path0: 0.5 + (0.2 + 0.3) + 0.4 = 1.4 ; path1: 0.2 + (0.0 + 0.4) + 0.6 = 1.2
    assert np.allclose(hb._total_traded(pos), [1.4, 1.2])
    oracle_pos = np.array([[0.5, 0.6, 0.5],
                           [0.2, 0.3, 0.4]])
    # oracle: 0.5+0.1+0.1+0.5 = 1.2 ; 0.2+0.1+0.1+0.4 = 0.8
    t_ex = hb._total_traded(pos) - hb._total_traded(oracle_pos)
    assert np.allclose(t_ex, [0.2, 0.4])
    # `turnover` (continuity column) keeps the old per-step-mean definition
    assert np.allclose(hb._turnover(pos), [0.25, 0.2])


def test_smooth_positions_pure_band():
    """FIX 4: pure no-trade band — position changes iff the raw delta's
    excursion from the held position exceeds the band, and then jumps TO the
    raw delta; ema_alpha is ignored (backward-compatible signature)."""
    pos = np.array([[0.50, 0.51, 0.55, 0.54, 0.60]])
    out = hb.smooth_positions(pos, None, 0.02)
    assert np.allclose(out, [[0.50, 0.50, 0.55, 0.55, 0.60]])
    assert np.array_equal(out, hb.smooth_positions(pos, 0.3, 0.02))
    rng = np.random.default_rng(3)
    raw = np.cumsum(rng.normal(0.0, 0.01, (4, 60)), axis=1)
    band = 0.02
    out = hb.smooth_positions(raw, None, band)
    assert np.array_equal(out[:, 0], raw[:, 0])
    moved = out[:, 1:] != out[:, :-1]
    assert moved.any() and not moved.all()
    # moved iff excursion > band; every move lands exactly on the raw delta
    exceeded = np.abs(raw[:, 1:] - out[:, :-1]) > band
    assert np.array_equal(moved, exceeded)
    assert np.array_equal(out[:, 1:][moved], raw[:, 1:][moved])


def test_persist_pnl_npz():
    """FIX 1 flag: risk.persist_pnl=false (default) writes nothing;
    true writes one npz per (cell, seed) keyed method__tc<tier>."""
    r_q = _cfg(n_paths=16, freq=26, n_boot=5)
    r, q = r_q["benchmark"]["grid"]["r"], r_q["benchmark"]["grid"]["q"]
    cfg = _trim_to_one_cell(r_q)
    out_dir = tempfile.mkdtemp()
    hb.run_headline(cfg, {"oracle": BSProvider(r, q)}, out_dir=out_dir)
    assert not (Path(out_dir) / "pnl_headline_delta_only").exists()
    cfg2 = _trim_to_one_cell(_cfg(n_paths=16, freq=26, n_boot=5))
    cfg2["engine"]["risk"]["persist_pnl"] = True
    out_dir2 = tempfile.mkdtemp()
    rows = hb.run_headline(cfg2, {"oracle": BSProvider(r, q)},
                           out_dir=out_dir2)
    files = sorted((Path(out_dir2) / "pnl_headline_delta_only").glob("*.npz"))
    assert len(files) == 1                # one trimmed cell, one seed
    data = np.load(files[0])
    tiers = cfg2["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"]
    assert set(data.files) == {f"oracle__tc{tc}" for tc in tiers}
    for tc in tiers:
        pnl = data[f"oracle__tc{tc}"]
        assert pnl.shape == (16,) and np.all(np.isfinite(pnl))
        row = next(r_ for r_ in rows if r_["tc"] == tc)
        assert row["mean_pnl"] == float(pnl.mean())   # CSV row <-> persisted PnL


def test_realized_dt_matches_the_contract():
    """H1/Q1: `dt` is DERIVED, not declared — T'=0.17 does not divide evenly into
    daily steps. The grid the engine actually lays down must equal the contract's
    amended rebalancing.{n_steps, dt_realized}; a disagreement means the code and
    the pre-registration have drifted apart."""
    cfg = hb.resolve_config(str(_DIR / "heston_benchmark_v6.yaml"),
                            str(_DIR / "hedging_config.yaml"))
    bm, eng = cfg["benchmark"], cfg["engine"]
    hs = bm["hedging_simulation"]
    reb = hs["rebalancing"]
    T_prime = float(hs["horizon"]["T_prime"])
    freq = eng["rebalancing"]["frequency_per_year"]
    assert freq == reb["frequency_per_year"]        # engine mirrors the contract's target

    n_steps = int(round(T_prime * freq))            # Hedging_backtest._run_sweep
    assert n_steps == int(reb["n_steps"])
    assert abs(T_prime / n_steps - float(reb["dt_realized"])) < 1e-6

    # the simulator's own grid, not just the arithmetic
    p = hb.SimParams.from_regime(bm["regimes"]["baseline"], bm["grid"]["r"],
                                 bm["grid"]["q"])
    times, S, _v = hb.simulate_heston_qe(p, 100.0, T_prime, n_steps, 8, 0,
                                         eng["simulation"]["psi_c"])
    steps = np.diff(times)
    assert steps.size == n_steps and S.shape[1] == n_steps + 1
    assert np.allclose(steps, float(reb["dt_realized"]), atol=1e-6)
    assert abs(times[-1] - T_prime) < 1e-12         # the horizon is exact, dt is not


class _ZeroWatchProvider(BSProvider):
    """BSProvider that records whether the engine ever handed it an EXACT v = 0
    (the QE exponential branch's atom at zero) and refuses non-finite output."""

    def __init__(self, r, q):
        super().__init__(r, q)
        self.saw_zero = False
        self.n_states = 0

    def evaluate(self, S, v, tau, K):
        va = np.asarray(v, float)
        self.saw_zero = self.saw_zero or bool((va == 0.0).any())
        self.n_states += va.size
        out = super().evaluate(S, v, tau, K)
        assert all(np.all(np.isfinite(out[k])) for k in out), "provider went non-finite"
        return out


def test_qe_exponential_branch_atom_reaches_the_providers():
    """The QE exponential branch has an atom at v = 0 (`np.where(ue <= pp, 0.0, ...)`)
    and the GreekProvider contract says outputs must be finite for all v >= 0. The
    baseline regime (Feller 1.78) produces NO exact zeros, so no engine test exercised
    it. This drives the contract's own Feller-stressed anchor instead."""
    cfg = _cfg(n_paths=128, freq=252, n_boot=10)
    bm = cfg["benchmark"]
    stressed = bm["regimes"]["feller_violating_volvol"]
    kappa, theta, xi = stressed["kappa"], stressed["theta"], stressed["xi"]
    assert 2 * kappa * theta / xi ** 2 < 0.5              # the regime IS Feller-violating
    # in-memory only (the contract file is untouched): hedge the stressed regime
    bm["regimes"]["baseline"] = dict(stressed)
    cfg = _trim_to_one_cell(cfg, direction="xi_up", magnitude=1.0)

    r, q = bm["grid"]["r"], bm["grid"]["q"]
    prov = _ZeroWatchProvider(r, q)
    rows = hb.run_headline(cfg, {"oracle": prov})

    assert prov.n_states > 0
    assert prov.saw_zero, "no exact v == 0 reached the provider — branch not exercised"
    assert rows and all(np.isfinite(float(r_["cvar"])) for r_ in rows)
    assert all(np.isfinite(float(r_["mean_pnl"])) for r_ in rows)


def test_delta_overlay_plot():
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    cfg = _cfg(n_paths=64, freq=26)
    r, q = cfg["benchmark"]["grid"]["r"], cfg["benchmark"]["grid"]["q"]
    out = Path(tempfile.mkdtemp()) / "overlay.png"
    hb.run_delta_overlay(cfg, {"bs": BSProvider(r, q)}, str(out),
                         direction="xi_up", magnitude=1.0)
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        globals()[fn]()
        print(f"PASS {fn}")
