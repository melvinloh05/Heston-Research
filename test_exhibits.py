"""test_exhibits.py — exhibit regeneration on tiny synthetic results/ fixtures.

Every exhibit function must (1) run on a minimal results/ tree and write a
nonempty PNG + CSV, (2) regenerate BIT-STABLY for fixed inputs (we hash the
backing CSV, never the PNG), and (3) raise with the EXACT expected path when a
required input file is missing. The overlay wrapper subsets providers and
delegates to the engine diagnostic (monkeypatched — no simulation in tests).
"""
from __future__ import annotations

import csv
import hashlib
import os

import pytest

import exhibits
import Hedging_backtest as hb


# ---------------------------------------------------------------------------
# tiny synthetic results/ fixtures (schemas track P11 / P12 / P13)
# ---------------------------------------------------------------------------

_HEDGE_METHODS = {
    # method: (cvar@misspec, gap_closed, t_ex, tc_comp, dir_comp, cvar_diff_vs_base)
    "oracle":                 (0.40, 1.00, 0.00, 0.05, 0.02, ""),
    "standard_pinn":          (1.00, 0.00, 0.30, 0.08, -0.01, ""),
    "standard_pinn_smoothed": (0.92, 0.13, 0.10, 0.04, -0.02, -0.08),
    "rung1":                  (0.85, 0.25, 0.28, 0.075, 0.00, -0.15),
    "rung2":                  (0.70, 0.50, 0.24, 0.07, 0.01, -0.30),
    "rung3":                  (0.62, 0.63, 0.20, 0.065, 0.015, -0.38),
    "sans_pde":               (0.78, 0.37, 0.26, 0.072, 0.005, -0.22),
}

_AGG_COLS = ["sweep", "direction", "magnitude", "lambda_j", "sigma_j", "in_model",
             "method", "tc", "cvar_level", "n_seeds", "cvar_mean", "cvar_seed_std",
             "mean_pnl_mean", "turnover_mean", "turnover_seed_std",
             "tc_component_mean", "directional_component_mean", "t_ex_mean",
             "t_ex_seed_std", "gap_closed_mean", "gap_closed_seed_std",
             "pnl_vs_baseline_cvar_diff_mean", "cvar_boot_se_mean"]


def _agg_row(method, magnitude, tc, blanks=()):
    """One agg row. `blanks` names columns to leave EMPTY for this method, the
    on-disk state the engine really produces (delta-gamma rows carry t_ex="",
    every arm does when the oracle provider is absent)."""
    cv, gc, te, tcc, dc, diff = _HEDGE_METHODS[method]
    # misspec (m=1.0) is the headline; in-model (m=0.0) is a milder, shrunk copy
    scale = 1.0 if magnitude >= 0.5 else 0.4
    cv_m = cv * scale + tc * 2.0                        # cost lifts CVaR with tc
    row = {"sweep": "perturbation", "direction": "combined", "magnitude": magnitude,
           "lambda_j": "", "sigma_j": "", "in_model": magnitude == 0.0,
           "method": method, "tc": tc, "cvar_level": 0.95, "n_seeds": 5,
           "cvar_mean": round(cv_m, 6), "cvar_seed_std": 0.03,
           "mean_pnl_mean": -0.1, "turnover_mean": 0.5 + te, "turnover_seed_std": 0.02,
           "tc_component_mean": tcc, "directional_component_mean": dc,
           "t_ex_mean": te, "t_ex_seed_std": 0.01,
           "gap_closed_mean": gc, "gap_closed_seed_std": 0.05,
           "pnl_vs_baseline_cvar_diff_mean": diff, "cvar_boot_se_mean": 0.02}
    for col in blanks:
        row[col] = ""
    return row


def _write_hedging_agg(path, blanks=None):
    """`blanks`: {method: (column, ...)} to leave empty; default populates all."""
    blanks = blanks or {}
    rows = []
    for magnitude in (0.0, 1.0):
        for tc in (0.0, 0.01, 0.02):
            for method in _HEDGE_METHODS:
                rows.append(_agg_row(method, magnitude, tc,
                                     blanks=blanks.get(method, ())))
    _dictwrite(path, _AGG_COLS, rows)
    return path


_GREEK_COLS = ["regime", "slice", "arm", "greek", "n_seeds",
               "rmse_mean", "rmse_std", "rel_rmse_mean", "rel_rmse_std",
               "reduction_vs_standard_pinn_mean", "reduction_vs_standard_pinn_std",
               "price_parity_mean", "price_parity_std"]

_GREEK_RED = {"gamma": 0.22, "vega": 0.31, "vanna": 0.18, "delta": 0.05, "price": 0.0}


def _write_greek_agg(path):
    rows = []
    for regime in ("near_feller", "strong_neg_corr"):
        for arm in ("rung3", "rung2"):
            for greek, red in _GREEK_RED.items():
                bump = 0.0 if arm == "rung3" else -0.05
                rows.append({"regime": regime, "slice": "full", "arm": arm,
                             "greek": greek, "n_seeds": 5,
                             "rmse_mean": 0.01, "rmse_std": 0.001,
                             "rel_rmse_mean": 0.1, "rel_rmse_std": 0.01,
                             "reduction_vs_standard_pinn_mean": round(red + bump, 4),
                             "reduction_vs_standard_pinn_std": 0.04,
                             "price_parity_mean": 0.03, "price_parity_std": 0.005})
    _dictwrite(path, _GREEK_COLS, rows)
    return path


_DOSE_COLS = ["arm", "label_source", "gamma_label_noise_sigma", "label_error",
              "cvar_mean", "cvar_seed_std", "n_seeds", "isotonic_fit", "is_reference"]

# (arm, source, sigma, label_error, cvar, iso_fit, is_reference)
_DOSE = [
    ("sigma_000", "true_consensus", 0.00, 0.02, 0.62, 0.62, False),
    ("sigma_010", "true_consensus", 0.10, 0.12, 0.66, 0.66, False),
    ("sigma_025", "true_consensus", 0.25, 0.28, 0.71, 0.71, False),
    ("sigma_050", "true_consensus", 0.50, 0.55, 0.80, 0.80, False),
    ("bs_gamma",  "bs_gamma",       "",   0.18, 0.68, 0.685, False),
    ("shuffled",  "shuffled",       "",   0.95, 0.99, 0.99, False),
    ("gradient_penalty_only", "", "", "", 0.90, "", True),
]


def _write_dose(path):
    rows = [{"arm": a, "label_source": s, "gamma_label_noise_sigma": sig,
             "label_error": err, "cvar_mean": cv, "cvar_seed_std": 0.03,
             "n_seeds": 5, "isotonic_fit": iso, "is_reference": ref}
            for a, s, sig, err, cv, iso, ref in _DOSE]
    _dictwrite(path, _DOSE_COLS, rows)
    return path


def _dictwrite(path, cols, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def results(tmp_path):
    """A minimal results/ tree with P11/P12/P13 CSVs; returns the path dict."""
    root = tmp_path / "results"
    return {
        "hedging_agg": _write_hedging_agg(str(root / exhibits.HEDGING_AGG)),
        "greek_agg": _write_greek_agg(str(root / exhibits.GREEK_AGG)),
        "dose": _write_dose(str(root / "tables" / exhibits.DOSE_CSV)),
        "root": str(root),
    }


def _nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# each exhibit writes a nonempty png + csv
# ---------------------------------------------------------------------------

def test_e1_writes_png_and_csv(results, tmp_path):
    out = exhibits.exhibit_e1(results["hedging_agg"], results["greek_agg"],
                              str(tmp_path / "out"))
    assert _nonempty(out["png"]) and _nonempty(out["csv"])
    rows = exhibits.read_csv(out["csv"])
    sections = {r["section"] for r in rows}
    assert {"ladder", "factorial", "greek"} <= sections
    # the price-only x PDE-ablated corner is not a trained arm -> blank method
    fac = [r for r in rows if r["section"] == "factorial"]
    assert any(r["method"] == "" and r["cvar_mean"] == "" for r in fac)
    # rung3 gamma/vega reductions carried through from P11
    greek = [r for r in rows if r["section"] == "greek" and r["greek"] == "vega"]
    assert greek and all(exhibits._num(r["reduction_mean"]) is not None for r in greek)


def test_e2_writes_png_and_csv(results, tmp_path):
    out = exhibits.exhibit_e2(results["hedging_agg"], str(tmp_path / "out"))
    assert _nonempty(out["png"]) and _nonempty(out["csv"])
    rows = exhibits.read_csv(out["csv"])
    panels = {r["panel"] for r in rows}
    assert {"cvar_vs_tc", "t_ex", "inset_2x2"} <= panels
    # 2x2 has all four corners
    quad = {(r["regime"], r["tc"]) for r in rows if r["panel"] == "inset_2x2"}
    assert quad == {("in-model", "tc=0"), ("in-model", "tc>0"),
                    ("misspec", "tc=0"), ("misspec", "tc>0")}


def test_e3_writes_png_cb_and_csv(results, tmp_path):
    out = exhibits.exhibit_e3(results["dose"], str(tmp_path / "out"))
    assert _nonempty(out["png"]) and _nonempty(out["png_cb"]) and _nonempty(out["csv"])
    assert out["png"] != out["png_cb"]
    rows = exhibits.read_csv(out["csv"])
    roles = {r["role"] for r in rows}
    assert {"point", "bs_gamma", "shuffled", "reference"} <= roles
    # shuffled sits at the right edge (largest label error among on-axis points)
    on_axis = [r for r in rows if r["role"] != "reference"]
    xmax = max(on_axis, key=lambda r: exhibits._num(r["label_error"]))
    assert xmax["arm"] == "shuffled"


def test_e4_writes_png_and_csv(results, tmp_path):
    out = exhibits.exhibit_e4(results["hedging_agg"], results["greek_agg"],
                              str(tmp_path / "out"))
    assert _nonempty(out["png"]) and _nonempty(out["csv"])
    rows = exhibits.read_csv(out["csv"])
    panels = {r["panel"] for r in rows}
    assert {"decomp", "vanna", "scatter"} <= panels
    # band (smoothed) and Sobolev (rung3) both present in the scatter panel
    scat = {r["method"] for r in rows if r["panel"] == "scatter"}
    assert "standard_pinn_smoothed" in scat and "rung3" in scat


# ---------------------------------------------------------------------------
# X1 — a MISSING cell must not be DRAWN as a real 0.0
#
# The backing CSVs are already right (_fmt maps None -> ""); it is the PNG that
# lied. T_ex = 0 is the contract's cost-channel confirming evidence ("this arm
# trades exactly like the oracle"), so a blank cell drawn at zero is an
# affirmative finding invented out of an absence. matplotlib omits NaN bars.
# ---------------------------------------------------------------------------

def _capture_bars(monkeypatch):
    """Record every Axes.bar height the exhibit actually draws."""
    import matplotlib.axes

    drawn: list[list[float]] = []
    orig = matplotlib.axes.Axes.bar

    def spy(self, x, height, *a, **kw):
        drawn.append([float(h) for h in height])
        return orig(self, x, height, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy)
    return drawn


def _is_nan(v):
    return v != v


def test_e2_missing_t_ex_is_not_drawn_as_zero(tmp_path, monkeypatch):
    agg = _write_hedging_agg(str(tmp_path / "results" / exhibits.HEDGING_AGG),
                             blanks={"rung1": ("t_ex_mean", "t_ex_seed_std")})
    drawn = _capture_bars(monkeypatch)
    exhibits.exhibit_e2(agg, str(tmp_path / "out"))

    tex_methods = [m for m in exhibits.MECH_ARMS if m != exhibits.ORACLE_ARM]
    i = tex_methods.index("rung1")
    heights = [h for h in drawn if len(h) == len(tex_methods)]
    assert heights, "E2 drew no T_ex bar row"
    bars = heights[0]
    assert _is_nan(bars[i]), (
        f"blank t_ex_mean drawn at {bars[i]!r}; a missing cell must be absent, "
        "not a bar at the cost channel's confirming value")
    # the populated arms are untouched
    assert bars[tex_methods.index("rung3")] == _HEDGE_METHODS["rung3"][2]
    # and the CSV, which was always correct, still records the blank
    rows = exhibits.read_csv(os.path.join(str(tmp_path / "out"), "e2_mechanism.csv"))
    tex = {r["method"]: r["value"] for r in rows if r["panel"] == "t_ex"}
    assert tex["rung1"] == ""


def test_e4_missing_gap_closed_is_not_drawn_as_zero(tmp_path, monkeypatch):
    arms = ("standard_pinn", "rung1", "rung2", "rung3", "sans_pde")
    agg = _write_hedging_agg(
        str(tmp_path / "results" / exhibits.HEDGING_AGG),
        blanks={"rung2": ("gap_closed_mean", "tc_component_mean",
                          "directional_component_mean")})
    greek = _write_greek_agg(str(tmp_path / "results" / exhibits.GREEK_AGG))
    drawn = _capture_bars(monkeypatch)
    exhibits.exhibit_e4(agg, greek, str(tmp_path / "out"))

    i = arms.index("rung2")
    rows = [h for h in drawn if len(h) == len(arms)]
    assert len(rows) >= 3, "E4 drew fewer bar rows than (a) + (b)'s two stacks"
    gap, dirs, costs = rows[0], rows[1], rows[2]
    assert _is_nan(gap[i]), (
        f"blank gap_closed_mean drawn at {gap[i]!r}; 0.0 reads as 'closed none "
        "of the oracle gap', an affirmative negative result")
    assert _is_nan(dirs[i]) and _is_nan(costs[i])
    assert gap[arms.index("rung3")] == _HEDGE_METHODS["rung3"][1]


def test_e4_missing_vanna_reduction_is_not_drawn_as_zero(tmp_path, monkeypatch):
    agg = _write_hedging_agg(str(tmp_path / "results" / exhibits.HEDGING_AGG))
    greek_path = str(tmp_path / "results" / exhibits.GREEK_AGG)
    _write_greek_agg(greek_path)
    # drop rung3's vanna row for one regime entirely (the arm was not evaluated)
    kept = [r for r in exhibits.read_csv(greek_path)
            if not (r["arm"] == "rung3" and r["greek"] == "vanna"
                    and r["regime"] == "near_feller")]
    _dictwrite(greek_path, _GREEK_COLS, kept)

    drawn = _capture_bars(monkeypatch)
    exhibits.exhibit_e4(agg, greek_path, str(tmp_path / "out"))
    pairs = [h for h in drawn if len(h) == 2]
    assert pairs, "E4 drew no vanna bar row"
    assert _is_nan(pairs[0][0]) and not _is_nan(pairs[0][1])


# ---------------------------------------------------------------------------
# bit-stable regeneration (hash the csv, not the png)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["e1", "e2", "e3", "e4"])
def test_regeneration_is_bit_stable(results, tmp_path, name):
    def _run(sub):
        d = str(tmp_path / sub)
        if name == "e1":
            return exhibits.exhibit_e1(results["hedging_agg"], results["greek_agg"], d)
        if name == "e2":
            return exhibits.exhibit_e2(results["hedging_agg"], d)
        if name == "e3":
            return exhibits.exhibit_e3(results["dose"], d)
        return exhibits.exhibit_e4(results["hedging_agg"], results["greek_agg"], d)

    a, b = _run("a"), _run("b")
    assert _sha(a["csv"]) == _sha(b["csv"])


# ---------------------------------------------------------------------------
# missing required input raises with the EXACT expected path
# ---------------------------------------------------------------------------

def test_e1_missing_hedging_raises_with_path(results, tmp_path):
    missing = str(tmp_path / "nope" / "absent_agg.csv")
    with pytest.raises(FileNotFoundError) as exc:
        exhibits.exhibit_e1(missing, results["greek_agg"], str(tmp_path / "out"))
    assert missing in str(exc.value)


def test_e1_missing_greek_raises_with_path(results, tmp_path):
    missing = str(tmp_path / "nope" / "absent_greek.csv")
    with pytest.raises(FileNotFoundError) as exc:
        exhibits.exhibit_e1(results["hedging_agg"], missing, str(tmp_path / "out"))
    assert missing in str(exc.value)


def test_e2_missing_hedging_raises_with_path(tmp_path):
    missing = str(tmp_path / "nope" / "absent_agg.csv")
    with pytest.raises(FileNotFoundError) as exc:
        exhibits.exhibit_e2(missing, str(tmp_path / "out"))
    assert missing in str(exc.value)


def test_e3_missing_dose_raises_with_path(tmp_path):
    missing = str(tmp_path / "nope" / "absent_dose.csv")
    with pytest.raises(FileNotFoundError) as exc:
        exhibits.exhibit_e3(missing, str(tmp_path / "out"))
    assert missing in str(exc.value)


def test_e4_missing_greek_raises_with_path(results, tmp_path):
    missing = str(tmp_path / "nope" / "absent_greek.csv")
    with pytest.raises(FileNotFoundError) as exc:
        exhibits.exhibit_e4(results["hedging_agg"], missing, str(tmp_path / "out"))
    assert missing in str(exc.value)


# ---------------------------------------------------------------------------
# overlay wrapper subsets providers and delegates (no simulation in tests)
# ---------------------------------------------------------------------------

def test_overlay_subsets_and_delegates(tmp_path, monkeypatch):
    captured = {}

    def fake_overlay(cfg, providers, out_path, direction=None, magnitude=0.0, seed=None):
        captured.update(providers=providers, direction=direction,
                        magnitude=magnitude, seed=seed)
        with open(out_path, "w") as fh:
            fh.write("stub")
        return out_path

    monkeypatch.setattr(hb, "run_delta_overlay", fake_overlay)
    providers = {"oracle": 1, "standard_pinn": 2, "rung1": 3, "rung3": 4, "extra": 5}
    out = str(tmp_path / "overlay.png")
    ret = exhibits.overlay_delta_paths({"cfg": True}, providers, out)

    assert ret == out and _nonempty(out)
    # only the three requested-and-present methods, in canonical colour order
    assert list(captured["providers"].keys()) == ["oracle", "standard_pinn", "rung3"]
    assert captured["direction"] == "combined" and captured["magnitude"] == 1.0


def test_overlay_missing_all_methods_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(hb, "run_delta_overlay",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(KeyError):
        exhibits.overlay_delta_paths({}, {"foo": 1}, str(tmp_path / "o.png"))


# ---------------------------------------------------------------------------
# CLI-level regenerate_all wiring
# ---------------------------------------------------------------------------

def test_regenerate_all(results, tmp_path):
    out = exhibits.regenerate_all(results["root"], str(tmp_path / "out"))
    assert set(out) == {"e1", "e2", "e3", "e4"}
    for paths in out.values():
        assert _nonempty(paths["png"]) and _nonempty(paths["csv"])
