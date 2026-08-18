"""analyze_results.py — P13 verdict + adjudication layer on the FROZEN run outputs.

Read-only over P12's per-seed CSVs, persisted per-path PnL npz banks and run
ledger, plus P11's OOD Greek tables. NO engine/model edits: every statistic here
is recomputed from artifacts the hedging engine and the Greek-eval layer already
wrote. The pre-registered thresholds (CLAUDE.md) are turned into typed verdicts;
the mechanism 2x2 / TC-sweep / T_ex adjudication is ASSEMBLED and DESCRIBED, never
decided beyond the pre-registered readings.

Two machineries:

  paired_ci_from_npz   the confirmatory statistic on persisted PnL. Reports BOTH
                       (i) per-seed paired CVaR-diff CIs (reusing the engine's
                       paired_bootstrap_cvar_diff, CRN pairing WITHIN a seed) then
                       mean +/- seed-std, and (ii) the POOLED-STRATIFIED number:
                       loss concatenated across seeds, bootstrap resampling path
                       indices WITHIN each seed block (pairing only holds inside a
                       seed), percentile CI on the pooled CVaR difference.
  the verdict funcs    one per contract threshold; each returns
                       {threshold_id, cell, statistic, ci_lo, ci_hi, verdict, notes}.

npz layout (Hedging_backtest._eval_methods with risk.persist_pnl): one file per
(cell, seed), name = Hedging_backtest._cell_slug(tag, seed) + '.npz', keys
'{arm}__tc{tc}' -> per-path total PnL on that cell's CRN paths. Slugs carry the
tag fields (direction/magnitude/sweep/...), so a cell is selected by matching the
parsed slug (see _slug_matches).
"""
from __future__ import annotations

import csv
import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np

import Hedging_backtest as hb

# ---------------------------------------------------------------------------
# artifact naming (must track Hedging_backtest / run_hedging)
# ---------------------------------------------------------------------------
_OUT_NAME = "headline_delta_only"                       # run_hedging._OUT_NAME
PER_SEED_CSV = f"{_OUT_NAME}_per_seed.csv"
AGG_CSV = f"{_OUT_NAME}_agg.csv"
PNL_DIR = f"pnl_{_OUT_NAME}"                             # per-cell PnL npz directory

# confirmatory cell = combined perturbation at magnitude 1.0 (misspec) / 0.0 (in-model);
# the cell SELECTOR, locked to the contract's confirmatory_cell by
# test_contract_thresholds.test_analyze_results_confirmatory_cell_filter_is_the_contract_cell
_MISSPEC_FILTER = {"sweep": "perturbation", "direction": "combined", "magnitude": 1.0}
_INMODEL_FILTER = {"sweep": "perturbation", "direction": "combined", "magnitude": 0.0}

_STREAM_POOLED = 7                                       # pooled-bootstrap rng stream
_STREAM_SPEARMAN = 8                                     # dose seed-bootstrap rng stream
_DIR = Path(__file__).resolve().parent
_TH_CACHE: dict = {}

# dose-response ladder (engine names == pinn_config arm names for these)
DOSE_ARMS = ("sigma_000", "sigma_010", "sigma_025", "sigma_050",
             "bs_gamma", "shuffled", "gradient_penalty_only")
_GRADPEN = "gradient_penalty_only"

VERDICT_COLS = ["threshold_id", "cell", "n_seeds", "statistic", "ci_lo", "ci_hi",
                "verdict", "notes"]
DOSE_COLS = ["arm", "label_source", "gamma_label_noise_sigma", "label_error",
             "cvar_mean", "cvar_seed_std", "n_seeds", "n_seeds_common",
             "isotonic_fit", "is_reference"]


# ---------------------------------------------------------------------------
# contract thresholds (audit A1/C1: this module used to own 11 re-typed literals)
# ---------------------------------------------------------------------------

def _default_thresholds(contract_path=None, engine_path=None) -> dict:
    """The contract's pre-registered numerics (hb.contract_thresholds), cached.

    Every verdict function takes its thresholds as ARGUMENTS; when the caller
    supplies none it falls back to this — the contract — never to a module-level
    literal. Paths default to the copies next to this module so the fallback does
    not depend on the working directory.
    """
    key = (str(contract_path or _DIR / "heston_benchmark_v6.yaml"),
           str(engine_path or _DIR / "hedging_config.yaml"))
    if key not in _TH_CACHE:
        _TH_CACHE[key] = hb.contract_thresholds(hb.resolve_config(*key))
    return dict(_TH_CACHE[key])


def _pick(value, th: dict, key: str):
    """`value` when the caller supplied one, else the contract's `key`."""
    return th[key] if value is None else value


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _num(v):
    """float(v) or None for blanks / non-numeric ('' stays None, never raises)."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce(v: str):
    """CSV cell -> int/float/str ('' stays blank), matching run_hedging._coerce."""
    if v == "":
        return ""
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(path) -> list[dict]:
    """Typed rows from a metric CSV (blank cells stay ''); [] when missing."""
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _finite_mean_std(vals) -> tuple[float, float, int]:
    """(mean, sample-std ddof=1, n) over finite entries; std=0 for n==1, nan/nan/0 empty."""
    a = np.asarray([x for x in (_num(v) for v in vals) if x is not None], float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0), int(a.size)


def _normal_ci(mean: float, std: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approx seed CI mean +/- z*std/sqrt(n) (the seed-to-seed CI convention)."""
    if not math.isfinite(mean) or n <= 0:
        return float("nan"), float("nan")
    se = (std / math.sqrt(n)) if (math.isfinite(std) and n > 0) else float("nan")
    if not math.isfinite(se):
        return float("nan"), float("nan")
    return mean - z * se, mean + z * se


def _excludes_zero(lo: float, hi: float) -> bool:
    """True iff the CI [lo, hi] lies strictly on one side of 0 (both finite)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo > 0.0 or hi < 0.0


def _covers_zero(lo: float, hi: float) -> bool:
    return math.isfinite(lo) and math.isfinite(hi) and lo <= 0.0 <= hi


def _tc_key(tc) -> str:
    """PnL-npz tc token, matching Hedging_backtest f'{arm}__tc{tc}' with float tiers."""
    if isinstance(tc, str):
        return tc
    return f"{float(tc)}"


def _pnl_key(arm: str, tc) -> str:
    return f"{arm}__tc{_tc_key(tc)}"


def _parse_slug(name: str) -> dict:
    """Slug (or file name) -> {field: str}; 'seed' cast to int. Inverse of _cell_slug."""
    stem = os.path.basename(str(name))
    if stem.endswith(".npz"):
        stem = stem[:-4]
    out: dict = {}
    for tok in stem.split("__"):
        if "=" not in tok:
            continue
        k, val = tok.split("=", 1)
        out[k] = val
    if "seed" in out:
        try:
            out["seed"] = int(out["seed"])
        except ValueError:
            pass
    return out


def _slug_matches(parsed: dict, slug_filter) -> bool:
    """Match a parsed slug against None / a callable / a {field: value} dict.

    Dict values compare as strings (slug fields are strings): a float filter value
    1.0 matches the slug token 'magnitude=1.0'. A field absent from the slug never
    matches a non-empty requested value.
    """
    if slug_filter is None:
        return True
    if callable(slug_filter):
        return bool(slug_filter(parsed))
    for k, want in slug_filter.items():
        got = parsed.get(k)
        if got is None or str(got) != str(want):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. paired machinery on persisted PnL
# ---------------------------------------------------------------------------

def _gather_cell_arrays(cell_dir, arm_a: str, arm_b: str, tc, slug_filter
                        ) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(seed, pnl_a, pnl_b) per matching cell npz, sorted by seed.

    Every '*.npz' in `cell_dir` whose parsed slug passes `slug_filter` AND carries
    both '{arm}__tc{tc}' keys is loaded. Files missing a key (e.g. the arm was not
    hedged on that cell) are skipped. The seed is read from the slug.
    """
    key_a, key_b = _pnl_key(arm_a, tc), _pnl_key(arm_b, tc)
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for path in sorted(glob.glob(os.path.join(str(cell_dir), "*.npz"))):
        parsed = _parse_slug(path)
        if not _slug_matches(parsed, slug_filter):
            continue
        with np.load(path, allow_pickle=False) as d:
            if key_a not in d.files or key_b not in d.files:
                continue
            a = np.asarray(d[key_a], float)
            b = np.asarray(d[key_b], float)
        seed = parsed.get("seed")
        seed = int(seed) if isinstance(seed, int) else len(blocks)
        blocks.append((seed, a, b))
    blocks.sort(key=lambda t: t[0])
    return blocks


def _pooled_stratified(blocks, level: float, n_boot: int, seed: int) -> dict:
    """POOLED-STRATIFIED paired CVaR diff (the confirmatory number).

    Point estimate: CVaR of the loss POOLED across seed blocks (arm a minus arm b).
    Bootstrap: each replicate resamples path indices WITHIN each seed block and
    applies the SAME indices to both arms of that block (CRN pairing only holds
    inside a seed), re-pools, and recomputes the CVaR difference; percentile CI.
    """
    a_all = np.concatenate([a for _, a, _ in blocks])
    b_all = np.concatenate([b for _, _, b in blocks])
    cvar_a, cvar_b = hb.cvar(a_all, level), hb.cvar(b_all, level)
    rng = np.random.default_rng([int(seed), _STREAM_POOLED])
    diffs = np.empty(int(n_boot))
    for k in range(int(n_boot)):
        pa, pb = [], []
        for _s, a, b in blocks:
            idx = rng.integers(0, a.size, a.size)
            pa.append(a[idx])
            pb.append(b[idx])
        diffs[k] = hb.cvar(np.concatenate(pa), level) - hb.cvar(np.concatenate(pb), level)
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    return {"cvar_a": float(cvar_a), "cvar_b": float(cvar_b),
            "diff": float(cvar_a - cvar_b),
            # ONE definition, shared with the engine (audit Q4): |cvar_b| in the
            # denominator, so a negative baseline CVaR cannot invert the sign of
            # a real improvement and turn a pass into a fail.
            "rel_improvement": hb.rel_improvement(cvar_a, cvar_b),
            "ci_lo": lo, "ci_hi": hi,
            "n_paths": int(a_all.size), "n_seeds": len(blocks)}


def paired_ci_from_npz(cell_dir, arm_a: str, arm_b: str, tc,
                       level: float | None = None, n_boot: int | None = None,
                       seed: int | None = None, *, slug_filter=None,
                       thresholds: dict | None = None) -> dict:
    """Paired CVaR-difference of two arms on the SAME CRN paths, from persisted PnL.

    Loads arm_a/arm_b per-path PnL per seed block (one npz per (cell, seed)) and
    reports BOTH readings the contract asks for:

      per_seed          list of the engine's own paired_bootstrap_cvar_diff per
                        seed (paired WITHIN the seed, seeded by that seed so the
                        numbers reproduce the engine's per-seed CSV);
      per_seed_summary  mean +/- seed-std of the per-seed rel-improvement/diff;
      pooled            the POOLED-STRATIFIED confirmatory statistic
                        (_pooled_stratified): CVaR of the pooled loss and a
                        percentile CI on the pooled diff, bootstrap resampling
                        path indices within each seed block.

    `slug_filter` selects the cell among many npz in one directory (dict of
    slug-field -> value, or a callable on the parsed slug). Raises if no cell
    npz carries both arms at `tc`. level/n_boot/seed default to the contract
    (cvar_convention.level, risk.bootstrap_B, meta.global_seed).
    """
    th = _default_thresholds() if thresholds is None else thresholds
    level = _pick(level, th, "cvar_level")
    n_boot = _pick(n_boot, th, "n_boot")
    seed = _pick(seed, th, "global_seed")
    blocks = _gather_cell_arrays(cell_dir, arm_a, arm_b, tc, slug_filter)
    if not blocks:
        raise FileNotFoundError(
            f"no PnL npz in {cell_dir!r} carrying both {arm_a!r} and {arm_b!r} at "
            f"tc={tc} (filter={slug_filter}); nothing to pair")

    per_seed = []
    for s, a, b in blocks:
        st = hb.paired_bootstrap_cvar_diff(a, b, level, n_boot, s)
        per_seed.append({"seed": s, "cvar_a": hb.cvar(a, level),
                         "cvar_b": hb.cvar(b, level), **st})
    rel_m, rel_s, _ = _finite_mean_std([r["rel_improvement"] for r in per_seed])
    dif_m, dif_s, _ = _finite_mean_std([r["diff"] for r in per_seed])
    summary = {"rel_improvement_mean": rel_m, "rel_improvement_seed_std": rel_s,
               "diff_mean": dif_m, "diff_seed_std": dif_s}
    return {"arm_a": arm_a, "arm_b": arm_b, "tc": float(tc) if _num(tc) is not None else tc,
            "level": level, "n_seeds": len(blocks),
            "per_seed": per_seed, "per_seed_summary": summary,
            "pooled": _pooled_stratified(blocks, level, n_boot, seed)}


# ---------------------------------------------------------------------------
# verdict record helper
# ---------------------------------------------------------------------------

def _verdict(threshold_id: str, cell: str, statistic, ci, verdict: str,
             notes: str, n_seeds=None) -> dict:
    """One verdict record. `n_seeds` is a STRUCTURED column (audit A3): the seed
    count a verdict was computed on must be readable without parsing `notes`."""
    lo, hi = (ci if ci is not None else (float("nan"), float("nan")))
    return {"threshold_id": threshold_id, "cell": cell,
            "n_seeds": "" if n_seeds is None else int(n_seeds),
            "statistic": float(statistic) if statistic is not None else float("nan"),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "verdict": verdict, "notes": notes}


#: exception types that are how a genuinely MISSING artifact presents itself.
#: FileNotFoundError is raised by `paired_ci_from_npz` when no cell npz carries
#: the arms, and by every artifact open. KeyError belongs here ONLY at the one
#: site where absence really does surface that way — `SobolevPINN.load_arm` on an
#: arm that is not in pinn_config.yaml (see `_measured_label_error`) — so it is
#: NOT listed globally: a KeyError from a renamed CSV column is a defect, not an
#: absence, and must not be laundered into "not evaluated".
_ABSENCE_EXC: tuple = (FileNotFoundError,)


def _failure_verdict(exc: BaseException, tid: str, cell: str) -> dict:
    """A verdict row for an analysis that could not produce one (audit A4).

    'null' is reserved for ABSENCE — the artifact is not there, nothing was
    evaluated, a legitimate and non-alarming state (`ood_greek_thresholds` uses
    the same word for it). Anything else — a renamed column, a shape mismatch, a
    failed import — is a real defect in a PRESENT artifact, and must not be
    indistinguishable from absence at a glance: `mechanism_memo` prints every
    verdict identically as **{verdict}**, so the distinction has to live in the
    value. Those get verdict='error'.
    """
    if isinstance(exc, _ABSENCE_EXC):
        return _verdict(tid, cell, float("nan"), None, "null",
                        f"not evaluated: {type(exc).__name__}: {exc}")
    return _verdict(tid, cell, float("nan"), None, "error",
                    f"ANALYSIS ERROR (artifact present but unusable): "
                    f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 2a. confirmatory pass + order attribution
# ---------------------------------------------------------------------------

def confirmatory_cell(pnl_dir, *, thresholds: dict | None = None,
                      tc: float | None = None, level: float | None = None,
                      n_boot: int | None = None, seed: int | None = None,
                      arm: str = "rung3", baseline: str | None = None,
                      rel_threshold: float | None = None,
                      n_seeds_required: int | None = None) -> dict:
    """Confirmatory pass: rung3 vs standard_pinn at (combined, m=1.0, tc=1%), 10
    seeds. PASS iff pooled-stratified rel improvement >= rel_threshold AND the
    pooled CI on the CVaR difference excludes 0 (on the improvement side) AND the
    cell carries the pre-registered seed count.

    The seed count is a CONDITION, not a footnote (audit A3): run_hedging is
    resumable, so a partially-complete confirmatory directory is a normal on-disk
    state and would otherwise yield a fully-formed 'pass' on an under-powered seed
    set. A short cell is 'null' (not evaluated), never 'fail' — nothing about the
    hypothesis has been tested.

    Every threshold defaults to the CONTRACT (`thresholds` / _default_thresholds),
    never to a module literal."""
    th = _default_thresholds() if thresholds is None else thresholds
    tc = _pick(tc, th, "confirmatory_tc")
    baseline = _pick(baseline, th, "baseline_arm")
    rel_threshold = _pick(rel_threshold, th, "confirmatory_rel_threshold")
    n_req = int(_pick(n_seeds_required, th, "seeds_confirmatory_cell"))
    res = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                             slug_filter=_MISSPEC_FILTER, thresholds=th)
    p = res["pooled"]
    rel = p["rel_improvement"]
    n_seeds = int(res["n_seeds"])
    ok_ci = _excludes_zero(p["ci_lo"], p["ci_hi"]) and p["ci_hi"] < 0.0
    ok_rel = math.isfinite(rel) and rel >= rel_threshold
    ok_seeds = n_seeds == n_req
    verdict = ("null" if not ok_seeds else "pass" if (ok_ci and ok_rel) else "fail")
    notes = (f"{n_seeds} seeds; pooled rel improvement={rel:.4f} "
             f"(threshold {rel_threshold:.2f}: {'met' if ok_rel else 'not met'}); "
             f"pooled CVaR diff CI=[{p['ci_lo']:.4g}, {p['ci_hi']:.4g}] "
             f"({'excludes' if ok_ci else 'includes'} 0); statistic column is the "
             f"pooled rel improvement, CI columns are the pooled CVaR-diff CI. "
             f"per-seed mean rel={res['per_seed_summary']['rel_improvement_mean']:.4f}"
             f"+/-{res['per_seed_summary']['rel_improvement_seed_std']:.4f}.")
    if not ok_seeds:
        notes = (f"NOT EVALUATED: {n_seeds} of the pre-registered {n_req} seeds "
                 f"present (meta.seeds_confirmatory_cell) — the confirmatory cell is "
                 f"incomplete, so no pass/fail is claimed. " + notes)
    return _verdict("confirmatory_cell", f"combined m=1.0 tc={tc}", rel,
                    (p["ci_lo"], p["ci_hi"]), verdict, notes, n_seeds=n_seeds)


def order_attribution(pnl_dir, *, thresholds: dict | None = None,
                      tc: float | None = None, level: float | None = None,
                      n_boot: int | None = None, seed: int | None = None,
                      arm: str = "rung2", baseline: str = "rung1") -> dict:
    """Order attribution: rung2 beats rung1 at the confirmatory cell, pooled CI
    excludes 0. The causal claim lives on this rung1->rung2 (add-Gamma) gap.
    fail => honest null (no evidence Gamma supervision helped over Delta-only)."""
    th = _default_thresholds() if thresholds is None else thresholds
    tc = _pick(tc, th, "confirmatory_tc")
    res = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                             slug_filter=_MISSPEC_FILTER, thresholds=th)
    p = res["pooled"]
    excludes = _excludes_zero(p["ci_lo"], p["ci_hi"])
    beats = excludes and p["ci_hi"] < 0.0
    verdict = "pass" if beats else "fail"
    # THREE outcomes, not two. The old `else` branch asserted "CI includes 0" for
    # every non-pass, which is FALSE when the CI excludes 0 on the HARMFUL side
    # (diff = arm - baseline, so ci_lo > 0 means rung2 is significantly WORSE).
    # That is a materially different finding from a null and must not be
    # mislabelled: a null says "no evidence Gamma helps", this says "evidence
    # Gamma hurts at this cell".
    if beats:
        note = "rung2 beats rung1, CI excludes 0"
    elif excludes:                       # CI wholly above 0 -> significant REVERSAL
        note = ("SIGNIFICANT REVERSAL, not a null: rung1->rung2 (add-Gamma) CI "
                "EXCLUDES 0 on the harmful side — Gamma supervision significantly "
                "WORSENS delta-only hedging versus delta supervision alone at this cell")
    else:
        note = ("honest null: rung1->rung2 (add-Gamma) CI includes 0 — no evidence the "
                "Gamma rung improves delta-only hedging over delta supervision alone")
    notes = (f"{res['n_seeds']} seeds; pooled CVaR diff={p['diff']:.4g} "
             f"CI=[{p['ci_lo']:.4g}, {p['ci_hi']:.4g}]; rel={p['rel_improvement']:.4f}. "
             + note)
    return _verdict("order_attribution", f"rung2-vs-rung1 combined m=1.0 tc={tc}",
                    p["diff"], (p["ci_lo"], p["ci_hi"]), verdict, notes,
                    n_seeds=res["n_seeds"])


# ---------------------------------------------------------------------------
# 2b. dose-response (measured gamma-label error axis)
# ---------------------------------------------------------------------------

def _measured_label_error(labels_npz, pinn_cfg_path, arm: str,
                          label_seed: int) -> tuple:
    """(||gamma_label - gamma_ref|| / ||gamma_ref||, label_source, sigma) for one arm.

    Reads the arm's PINNConfig (SobolevPINN.load_arm) and its batch from the frozen
    label artifact (make_labels.build_arm_labels). gamma_ref is the TRUE consensus
    shared by every arm; the arm's gamma label is oracle/bs_gamma/shuffled/none per
    its config. label_source 'none' (gradient_penalty_only) has no gamma label ->
    error is NaN and the arm is a reference line, not a dose point.
    """
    from SobolevPINN import load_arm
    from make_labels import build_arm_labels
    from pinn_provider import _ARM_DIR

    acfg = load_arm(str(pinn_cfg_path), _ARM_DIR.get(arm, arm))
    batch = build_arm_labels(str(labels_npz), acfg, seed=int(label_seed))
    gref = np.asarray(batch["gamma_ref"], float)
    if "gamma" not in batch:
        return float("nan"), acfg.label_source, float(acfg.gamma_label_noise_sigma)
    g = np.asarray(batch["gamma"], float)
    denom = float(np.linalg.norm(gref))
    err = float(np.linalg.norm(g - gref) / denom) if denom > 0 else float("nan")
    return err, acfg.label_source, float(acfg.gamma_label_noise_sigma)


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-decreasing fit of y (weights w)."""
    val: list[float] = []
    wt: list[float] = []
    cnt: list[int] = []
    for yi, wi in zip(y, w):
        cv, cw, cc = float(yi), float(wi), 1
        while val and val[-1] > cv:
            pv, pw, pc = val.pop(), wt.pop(), cnt.pop()
            cv, cw, cc = (pv * pw + cv * cw) / (pw + cw), pw + cw, pc + cc
        val.append(cv)
        wt.append(cw)
        cnt.append(cc)
    fit = np.empty(len(y))
    k = 0
    for v, c in zip(val, cnt):
        fit[k:k + c] = v
        k += c
    return fit


def isotonic_increasing(x, y) -> np.ndarray:
    """Isotonic (non-decreasing in x) fit of y, returned in the ORIGINAL order."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size == 0:
        return y.copy()
    order = np.argsort(x, kind="mergesort")
    fit_sorted = _pava(y[order], np.ones(x.size))
    fit = np.empty_like(fit_sorted)
    fit[order] = fit_sorted
    return fit


def _spearman_seed_bootstrap(xs, y_by_seed, seeds, n_boot: int, seed: int):
    """(rho, one-sided p, ci) for Spearman(label_error, seed-mean CVaR) across arms.

    Point estimate on the per-arm seed means; the p-value and CI come from
    resampling the seed set with replacement (the y's per-arm noise source) and
    recomputing rho. p = P(rho <= 0): the pre-registered 'flat = regularization
    null' probability. `xs` and each arm row of `y_by_seed` are aligned by arm.
    """
    from scipy.stats import spearmanr

    xs = np.asarray(xs, float)
    seeds = list(seeds)
    y_means = np.array([np.mean([y_by_seed[a][s] for s in seeds]) for a in range(len(xs))])
    with warnings.catch_warnings():                     # constant y -> undefined rho (nan)
        warnings.simplefilter("ignore")
        rho0 = spearmanr(xs, y_means).correlation
    if not math.isfinite(rho0) or n_boot <= 0 or len(seeds) < 2:
        return (float(rho0) if math.isfinite(rho0) else float("nan"),
                float("nan"), (float("nan"), float("nan")))
    # Q7: the documented [seed, _STREAM_*] convention, as everywhere else in this
    # module and the engine. Numerically inert — default_rng(42) and
    # default_rng([42, 8]) are different streams either way, and this bootstrap
    # resamples SEEDS while _STREAM_POOLED resamples paths — but the convention is
    # what makes stream collisions checkable by inspection.
    rng = np.random.default_rng([int(seed), _STREAM_SPEARMAN])
    rhos = np.empty(int(n_boot))
    with warnings.catch_warnings():                     # constant resample -> nan (filtered)
        warnings.simplefilter("ignore")
        for b in range(int(n_boot)):
            bs = rng.choice(seeds, len(seeds), replace=True)
            ym = np.array([np.mean([y_by_seed[a][s] for s in bs]) for a in range(len(xs))])
            rhos[b] = spearmanr(xs, ym).correlation
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return float(rho0), float("nan"), (float("nan"), float("nan"))
    p = float(np.mean(rhos <= 0.0))
    lo, hi = (float(x) for x in np.percentile(rhos, [2.5, 97.5]))
    return float(rho0), p, (lo, hi)


def dose_response(per_seed_csv, labels_npz, pinn_cfg_path, *,
                  thresholds: dict | None = None,
                  dose_arms=DOSE_ARMS, direction: str | None = None,
                  magnitude: float | None = None, tc: float | None = None,
                  reference_arm: str = _GRADPEN, label_seed: int | None = None,
                  n_boot: int | None = None, seed: int | None = None,
                  spearman_p_max: float | None = None):
    """Gamma-label-noise dose-response at the confirmatory cell.

    x  = MEASURED gamma-label error per arm, ||gamma_label - gamma_ref|| /
         ||gamma_ref|| from the frozen label artifact (build_arm_labels) — this
         puts shuffled/bs_gamma on the SAME axis as the sigma arms (documented:
         a single scalar per arm, not the noise sigma).
    y  = misspec CVaR95 at (direction, magnitude, tc) per arm, mean of the 'cvar'
         column over the seeds COMMON to every fitted arm — the same seed set the
         isotonic/Spearman fit uses, so the plotted point and the fitted line can
         never come from different subsets (audit A5). `n_seeds` is the arm's own
         count, `n_seeds_common` the fit set's; reference arms carry "" for the
         latter since they are off the fit.
    Fit: isotonic (CVaR non-decreasing in label error) + Spearman rank corr with
    a seed-bootstrap p. Verdict 'monotone' iff rho>0 and p<spearman_p_max, else
    'flat' (the pre-registered regularization null). gradient_penalty_only is the
    reference LINE (no gamma label -> not on the axis); the bs_gamma vs sigma_000
    gap is reported ('cheap labels suffice' reading).

    Returns (verdict_dict, rows) where rows are the per-arm DOSE_COLS records.
    The cell coordinates and `spearman_p_max` default to the CONTRACT
    (confirmatory_cell + acceptance_thresholds.dose_response.bootstrap_tail_prob_max;
    the latter is a one-sided BOOTSTRAP TAIL PROBABILITY, not a classical p-value).
    """
    th = _default_thresholds() if thresholds is None else thresholds
    direction = _pick(direction, th, "confirmatory_direction")
    magnitude = _pick(magnitude, th, "confirmatory_magnitude")
    tc = _pick(tc, th, "confirmatory_tc")
    label_seed = _pick(label_seed, th, "global_seed")
    seed = _pick(seed, th, "global_seed")
    n_boot = _pick(n_boot, th, "n_boot")
    spearman_p_max = _pick(spearman_p_max, th, "dose_bootstrap_tail_prob_max")
    csv_rows = read_csv(per_seed_csv)

    def _cell(arm):
        return [r for r in csv_rows
                if r.get("method") == arm
                and str(r.get("direction")) == str(direction)
                and _num(r.get("magnitude")) == float(magnitude)
                and _num(r.get("tc")) == float(tc)]

    rows: list[dict] = []
    fit_arms: list[str] = []
    xs: list[float] = []
    y_by_seed: list[dict] = []
    seed_sets: list[set] = []
    for arm in dose_arms:
        cell = _cell(arm)
        cvars = {int(r["seed"]): _num(r.get("cvar")) for r in cell
                 if _num(r.get("cvar")) is not None and r.get("seed") != ""}
        y_mean, y_std, n = _finite_mean_std(list(cvars.values()))
        try:
            err, src, sig = _measured_label_error(labels_npz, pinn_cfg_path, arm, label_seed)
        except (*_ABSENCE_EXC, KeyError) as exc:
            # ABSENCE only: the label artifact is missing (FileNotFoundError) or
            # the arm is not in pinn_config.yaml (load_arm's raw["arms"][arm] ->
            # KeyError). A corrupt-but-present artifact must NOT degrade to a
            # reference line — it propagates to the caller's error verdict.
            err, src, sig = float("nan"), f"unavailable ({type(exc).__name__})", float("nan")
        is_ref = (arm == reference_arm) or not math.isfinite(err)
        rows.append({"arm": arm, "label_source": src, "gamma_label_noise_sigma": sig,
                     "label_error": err, "cvar_mean": y_mean, "cvar_seed_std": y_std,
                     "n_seeds": n, "n_seeds_common": "",
                     "isotonic_fit": float("nan"), "is_reference": is_ref})
        if not is_ref and math.isfinite(err) and math.isfinite(y_mean) and cvars:
            fit_arms.append(arm)
            xs.append(err)
            y_by_seed.append(cvars)
            seed_sets.append(set(cvars))

    iso = np.array([])
    rho = p = float("nan")
    rho_ci = (float("nan"), float("nan"))
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []

    def _arm_ys(i) -> list:
        """Arm i's y values on the seed set the FIT uses (common when it exists)."""
        d = y_by_seed[i]
        return [d[s] for s in common] if common else list(d.values())

    # audit A5: cvar_mean is the point E3 PLOTS and the value bs_gap differences,
    # so it must come from the SAME seed set the isotonic/Spearman fit runs on.
    # Restricting the fit to `common` is right; reporting the mean over each arm's
    # OWN seeds left the plotted point and the fitted line on different subsets
    # whenever a partial resume made the sets diverge. `n_seeds` keeps the arm's
    # own count (the shortfall stays visible); `n_seeds_common` records the fit set.
    for i, a in enumerate(fit_arms):
        m, sd, n = _finite_mean_std(_arm_ys(i))
        for rec in rows:
            if rec["arm"] == a:
                rec["cvar_mean"], rec["cvar_seed_std"] = m, sd
                rec["n_seeds_common"] = n

    if len(fit_arms) >= 2:
        ys = [float(np.mean(_arm_ys(i))) for i in range(len(fit_arms))]
        iso = isotonic_increasing(xs, ys)
        for a, fv in zip(fit_arms, iso):
            for rec in rows:
                if rec["arm"] == a:
                    rec["isotonic_fit"] = float(fv)
        if len(common) >= 2:
            rho, p, rho_ci = _spearman_seed_bootstrap(
                xs, [{s: y_by_seed[i][s] for s in common} for i in range(len(fit_arms))],
                common, n_boot, seed)
        else:                                           # single seed: rank corr only
            from scipy.stats import spearmanr
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(xs, ys).correlation

    monotone = math.isfinite(rho) and rho > 0.0 and (
        not math.isfinite(p) or p < spearman_p_max)
    verdict = "monotone" if monotone else "flat"

    def _y(arm):
        for rec in rows:
            if rec["arm"] == arm:
                return rec["cvar_mean"]
        return float("nan")

    bs_gap = _y("bs_gamma") - _y("sigma_000")
    notes = (f"axis = measured ||gamma_label-gamma_ref||/||gamma_ref||; "
             f"Spearman rho={rho:.4f}"
             + (f", seed-bootstrap p(rho<=0)={p:.4f}" if math.isfinite(p) else "")
             + f"; {'monotone (CVaR rises with label error)' if monotone else 'flat = regularization null'}. "
             f"bs_gamma - sigma_000 CVaR gap={bs_gap:.4g} "
             f"('cheap labels suffice' if ~0). gradient_penalty_only drawn as "
             f"reference line (CVaR={_y(_GRADPEN):.4g}, off the label-error axis).")
    vd = _verdict("dose_response", f"{direction} m={magnitude} tc={tc}", rho,
                  rho_ci, verdict, notes)
    return vd, rows


# ---------------------------------------------------------------------------
# 2c. OOD Greek thresholds (join P11)
# ---------------------------------------------------------------------------

def ood_greek_thresholds(greek_agg_csv, *, thresholds: dict | None = None,
                         regimes=None, arm: str = "rung3",
                         gamma_red_threshold: float | None = None,
                         vega_red_threshold: float | None = None,
                         parity_tol: float | None = None,
                         binding: bool = True) -> dict:
    """OOD Greek pass: Gamma & Vega RMSE reduction vs standard_pinn >= 0.15 at
    price parity within 0.10, on near_feller and strong_neg_corr (full grid,
    seed mean). Joins P11's ood_param_greeks_agg.csv; CI is the seed-std normal
    CI on the Gamma reduction (the binding Greek). PASS iff every primary regime
    passes; fail => honest null.

    binding=False emits the SAME thresholds and CI machinery for a SECONDARY arm
    (e.g. rung2, the causal rung of the rung1->rung2 order claim) under a distinct
    threshold_id and a SECONDARY / non-binding note — the frozen contract binds
    rung3 only, so this row is informational and gates nothing.

    Regimes and all three thresholds default to the CONTRACT
    (splits.heldout_greek_and_hedging, acceptance_thresholds.{ood_gamma,ood_vega}
    _rmse_reduction_min, acceptance_thresholds.price_parity_within)."""
    th = _default_thresholds() if thresholds is None else thresholds
    regimes = _pick(regimes, th, "ood_regimes")
    gamma_red_threshold = _pick(gamma_red_threshold, th, "ood_gamma_reduction_min")
    vega_red_threshold = _pick(vega_red_threshold, th, "ood_vega_reduction_min")
    parity_tol = _pick(parity_tol, th, "price_parity_within")
    rows = read_csv(greek_agg_csv)
    lut = {(r.get("regime"), r.get("arm"), r.get("greek")): r
           for r in rows if r.get("slice") == "full"}
    per_regime: list[dict] = []
    gamma_ci = (float("nan"), float("nan"))
    binding_red = float("nan")
    for rg in regimes:
        gr = lut.get((rg, arm, "gamma"))
        vr = lut.get((rg, arm, "vega"))
        if gr is None or vr is None:
            per_regime.append({"regime": rg, "pass": False, "reason": "missing agg row"})
            continue
        g_red = _num(gr.get("reduction_vs_standard_pinn_mean"))
        g_std = _num(gr.get("reduction_vs_standard_pinn_std"))
        v_red = _num(vr.get("reduction_vs_standard_pinn_mean"))
        pp = _num(gr.get("price_parity_mean"))
        n = _num(gr.get("n_seeds")) or 0
        g_ok = g_red is not None and g_red >= gamma_red_threshold
        v_ok = v_red is not None and v_red >= vega_red_threshold
        p_ok = pp is not None and abs(pp) <= parity_tol
        ok = bool(g_ok and v_ok and p_ok)
        if binding_red is None or not math.isfinite(binding_red) or (
                g_red is not None and g_red < binding_red):
            binding_red = g_red if g_red is not None else binding_red
            gamma_ci = _normal_ci(g_red if g_red is not None else float("nan"),
                                  g_std if g_std is not None else float("nan"), int(n))
        per_regime.append({"regime": rg, "gamma_red": g_red, "vega_red": v_red,
                           "price_parity": pp, "pass": ok})
    # only regimes with actual agg rows are 'scored'; none present -> null (nothing
    # to evaluate), NOT fail (which is reserved for a present-but-below-threshold arm)
    scored = [r for r in per_regime if "gamma_red" in r]
    if not scored:
        verdict = "null"
    else:
        verdict = "pass" if all(r["pass"] for r in scored) else "fail"
    notes = "; ".join(
        f"{r['regime']}: " + (r.get("reason") or
        f"Gamma red={r.get('gamma_red')}, Vega red={r.get('vega_red')}, "
        f"price parity={r.get('price_parity')} -> {'ok' if r.get('pass') else 'no'}")
        for r in per_regime)
    tid = "ood_greek_thresholds" if binding else f"ood_greek_thresholds_{arm}_secondary"
    prefix = "" if binding else "SECONDARY / non-binding (contract binds rung3 only): "
    return _verdict(tid,
                    f"{arm} {'/'.join(regimes)} full-grid", binding_red, gamma_ci,
                    verdict, prefix + notes
                    + (" (fail => honest null)" if verdict == "fail" else ""))


# ---------------------------------------------------------------------------
# 2d. Sakuma-null consistency (in-model x tc=0; NOT pass/fail)
# ---------------------------------------------------------------------------

def sakuma_null_consistency(pnl_dir, *, thresholds: dict | None = None,
                            tc: float | None = None, level: float | None = None,
                            n_boot: int | None = None, seed: int | None = None,
                            arm: str = "rung3", baseline: str | None = None,
                            rel_tol: float | None = None) -> dict:
    """In-model (m=0), tc=0 rung3-vs-standard gap. CONSISTENCY CHECK, not pass/fail:
    'consistent' iff the pooled CI covers 0 OR |rel| < rel_tol (both readings of a
    negligible in-model gap, reproducing the Sakuma null); else 'flag'.

    tc and rel_tol default to the CONTRACT (the zero TC tier and
    acceptance_thresholds.sakuma_null_rel_tol)."""
    th = _default_thresholds() if thresholds is None else thresholds
    tc = th["tc_tiers"][0] if tc is None else tc
    baseline = _pick(baseline, th, "baseline_arm")
    rel_tol = _pick(rel_tol, th, "sakuma_null_rel_tol")
    res = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                             slug_filter=_INMODEL_FILTER, thresholds=th)
    p = res["pooled"]
    rel = p["rel_improvement"]
    consistent = _covers_zero(p["ci_lo"], p["ci_hi"]) or (
        math.isfinite(rel) and abs(rel) < rel_tol)
    verdict = "consistent" if consistent else "flag"
    notes = (f"{res['n_seeds']} seeds; pooled rel={rel:.4f}, CVaR diff CI="
             f"[{p['ci_lo']:.4g}, {p['ci_hi']:.4g}]. "
             f"{'CI covers 0 / |rel|<%.2f -> Sakuma null reproduced' % rel_tol if consistent else 'in-model gap present -> FLAG (unexpected under the null)'}")
    return _verdict("sakuma_null_consistency", f"combined m=0.0 tc={tc}", rel,
                    (p["ci_lo"], p["ci_hi"]), verdict, notes,
                    n_seeds=res["n_seeds"])


# ---------------------------------------------------------------------------
# 2e. mechanism adjudication (TC sweep x T_ex; 2x2; pre-registered readings ONLY)
# ---------------------------------------------------------------------------

def _mechanism_reading(gaps: list[dict], t_ex: dict) -> dict:
    """Pre-registered channel reading from the TC-swept gaps and the T_ex diff.

    gaps: per-tc {tc, diff, ci_lo, ci_hi} of rung3 - standard (loss units, <0 =
    rung3 better), sorted by tc. t_ex: {diff, ci_lo, ci_hi} of t_ex(rung3) -
    t_ex(standard) (excess-over-oracle turnover; <0 = rung3 trades less).

    Readings (CLAUDE.md 'Mechanism'):
      channel_i  (robustness): the tc=0 gap is present AND is an IMPROVEMENT
                 (CI excludes 0 on the improvement side, ci_hi < 0).
      channel_ii (cost): the gap WIDENS with tc IN THE IMPROVEMENT DIRECTION
                 (gap at the top tier is MORE negative, and its CI excludes 0 on
                 the improvement side) AND t_ex(rung3)-t_ex(std) < 0 with CI
                 excluding 0. t_ex unmoved (CI covers 0) KILLS channel_ii
                 regardless of PnL.
      decomposition: both present. no_channel: neither.

    `no_channel` is an ADJUDICATED outcome ("neither channel is present"), NOT the
    universal `null` ("not evaluated") — the two used to share the string `null`,
    which put a result and an absence in one column (contract
    acceptance_thresholds.verdict_vocabulary.outcome_values.mechanism_adjudication).

    BOTH channels require the gap to be an IMPROVEMENT (audit A2): an arm that is
    significantly WORSE at 0% TC, or one that degrades FASTER as costs rise, is the
    mirror image of the hypothesis and reads 'no_channel' — a pre-registered null is
    publishable, an affirmative mechanism claim built on a wrong-signed gap is not.
    """
    by_tc = {round(float(g["tc"]), 12): g for g in gaps}
    tcs = sorted(by_tc)
    g0 = by_tc[tcs[0]]
    gmax = by_tc[tcs[-1]]
    present_i = _excludes_zero(g0["ci_lo"], g0["ci_hi"]) and g0["ci_hi"] < 0.0
    widening = (_excludes_zero(gmax["ci_lo"], gmax["ci_hi"]) and gmax["ci_hi"] < 0.0
                and gmax["diff"] < g0["diff"] - 1e-12)
    t_ex_reduced = _excludes_zero(t_ex["ci_lo"], t_ex["ci_hi"]) and t_ex["diff"] < 0.0
    present_ii = bool(widening and t_ex_reduced)
    if present_i and present_ii:
        reading = "decomposition"
    elif present_i:
        reading = "channel_i"
    elif present_ii:
        reading = "channel_ii"
    else:
        reading = "no_channel"
    return {"reading": reading, "present_i": present_i, "present_ii": present_ii,
            "widening": bool(widening), "t_ex_reduced": bool(t_ex_reduced)}


def _slope(tcs, diffs) -> float:
    """d(gap)/d(tc) least-squares slope; NaN when < 2 finite points."""
    x = np.asarray(tcs, float)
    y = np.asarray(diffs, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def _t_ex_diff(per_seed_csv, *, direction, magnitude, tc,
               arm="rung3", baseline="standard_pinn") -> dict:
    """Seed-paired t_ex(arm) - t_ex(baseline) at a cell, with a seed-std CI.

    t_ex is tc-invariant (positions never depend on tc) so any tc row serves; a
    single tc is read for a clean per-seed pairing."""
    rows = read_csv(per_seed_csv)

    def _pick(method):
        out = {}
        for r in rows:
            if (r.get("method") == method
                    and str(r.get("direction")) == str(direction)
                    and _num(r.get("magnitude")) == float(magnitude)
                    and _num(r.get("tc")) == float(tc)):
                te = _num(r.get("t_ex"))
                if te is not None and r.get("seed") != "":
                    out[int(r["seed"])] = te
        return out

    a, b = _pick(arm), _pick(baseline)
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    mean, std, n = _finite_mean_std(diffs)
    lo, hi = _normal_ci(mean, std, n)
    return {"diff": mean, "seed_std": std, "n_seeds": n, "ci_lo": lo, "ci_hi": hi,
            "per_seed": {s: a[s] - b[s] for s in seeds}}


def mechanism_adjudication(pnl_dir, per_seed_csv, *, thresholds: dict | None = None,
                           tcs=None, level: float | None = None,
                           n_boot: int | None = None, seed: int | None = None,
                           arm: str = "rung3", baseline: str | None = None) -> dict:
    """Assemble the mechanism adjudication (NOT a choice beyond the pre-registered
    readings): rung3-standard gap over the TC sweep at misspec m=1.0 AND in-model
    m=0.0, the T_ex(rung3)-T_ex(std) excess-turnover diff, d(gap)/d(tc), the 2x2
    {in-model, misspec} x {tc=0, tc>0} gap view, and the _mechanism_reading.

    The TC sweep defaults to the CONTRACT's transaction_costs.tiers."""
    th = _default_thresholds() if thresholds is None else thresholds
    tcs = tuple(_pick(tcs, th, "tc_tiers"))
    baseline = _pick(baseline, th, "baseline_arm")

    def _gap(slug_filter, tc):
        p = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                               slug_filter=slug_filter, thresholds=th)["pooled"]
        return {"tc": float(tc), "diff": p["diff"], "rel": p["rel_improvement"],
                "ci_lo": p["ci_lo"], "ci_hi": p["ci_hi"]}

    misspec = [_gap(_MISSPEC_FILTER, tc) for tc in tcs]
    try:
        inmodel = [_gap(_INMODEL_FILTER, tc) for tc in tcs]
    except FileNotFoundError:
        inmodel = []
    t_ex = _t_ex_diff(per_seed_csv, direction=th["confirmatory_direction"],
                      magnitude=th["confirmatory_magnitude"],
                      tc=tcs[1] if len(tcs) > 1 else tcs[0],
                      arm=arm, baseline=baseline)
    reading = _mechanism_reading(misspec, t_ex)
    slope = _slope([g["tc"] for g in misspec], [g["diff"] for g in misspec])

    def _at(rows, tc):
        for g in rows:
            if abs(g["tc"] - tc) < 1e-12:
                return g
        return None
    tc0, tcpos = tcs[0], (tcs[1] if len(tcs) > 1 else tcs[0])
    two_by_two = {
        "misspec": {"tc0": _at(misspec, tc0), "tc_pos": _at(misspec, tcpos)},
        "in_model": {"tc0": _at(inmodel, tc0), "tc_pos": _at(inmodel, tcpos)},
    }
    g0 = _at(misspec, tc0)
    notes = (f"reading={reading['reading']} (present_i={reading['present_i']}, "
             f"present_ii={reading['present_ii']}, widening={reading['widening']}, "
             f"t_ex_reduced={reading['t_ex_reduced']}); d(gap)/d(tc)={slope:.4g}; "
             f"t_ex(rung3)-t_ex(std)={t_ex['diff']:.4g} CI=[{t_ex['ci_lo']:.4g}, "
             f"{t_ex['ci_hi']:.4g}] over {t_ex['n_seeds']} seeds.")
    vd = _verdict("mechanism_adjudication", "combined m=1.0 TC-sweep",
                  g0["diff"] if g0 else float("nan"),
                  (g0["ci_lo"], g0["ci_hi"]) if g0 else None,
                  reading["reading"], notes)
    vd.update({"misspec_gaps": misspec, "inmodel_gaps": inmodel, "t_ex": t_ex,
               "slope_dgap_dtc": slope, "two_by_two": two_by_two, "reading": reading})
    return vd


# ---------------------------------------------------------------------------
# 2f. Goldilocks (Bates severity sweep)
# ---------------------------------------------------------------------------

def goldilocks_bates(pnl_dir, per_seed_csv, *, thresholds: dict | None = None,
                     tc: float | None = None, level: float | None = None,
                     n_boot: int | None = None, seed: int | None = None,
                     arm: str = "rung3", baseline: str | None = None):
    """Per Bates (lambda_j, sigma_j) severity row: pooled rung3-standard gap + CI.
    Gap visibility is NON-MONOTONE in severity by pre-registration (goldilocks) —
    the sweep LOCATES the decision-relevant regime (CI excludes 0). 'nowhere
    decisive' is the pre-registered null, NOT a failure. Returns (verdict, rows)."""
    th = _default_thresholds() if thresholds is None else thresholds
    tc = _pick(tc, th, "confirmatory_tc")
    baseline = _pick(baseline, th, "baseline_arm")
    csv_rows = read_csv(per_seed_csv)
    combos = sorted({(r.get("lambda_j"), r.get("sigma_j")) for r in csv_rows
                     if str(r.get("sweep")) == "bates"
                     and r.get("lambda_j") not in (None, "")
                     and r.get("sigma_j") not in (None, "")})
    rows: list[dict] = []
    for lam, sj in combos:
        filt = {"sweep": "bates", "lambda_j": lam, "sigma_j": sj}
        try:
            p = paired_ci_from_npz(pnl_dir, arm, baseline, tc, level, n_boot, seed,
                                   slug_filter=filt, thresholds=th)["pooled"]
        except FileNotFoundError:
            continue
        decisive = _excludes_zero(p["ci_lo"], p["ci_hi"]) and p["ci_hi"] < 0.0
        rows.append({"lambda_j": lam, "sigma_j": sj, "diff": p["diff"],
                     "rel": p["rel_improvement"], "ci_lo": p["ci_lo"],
                     "ci_hi": p["ci_hi"], "decisive": decisive,
                     "n_seeds": p["n_seeds"]})
    decisive_rows = [r for r in rows if r["decisive"]]
    # rows present but none decisive is the pre-registered 'nowhere' ADJUDICATION;
    # NO rows at all is not an adjudication, it is NOT EVALUATED (the universal
    # `null`). The two used to share one string (contract verdict_vocabulary).
    verdict = ("decision_relevant_regime_located" if decisive_rows
               else "no_decisive_regime")
    if not rows:
        verdict = "null"
    best = min(rows, key=lambda r: r["diff"]) if rows else None
    where = ", ".join(f"(lam={r['lambda_j']}, sig={r['sigma_j']})" for r in decisive_rows)
    notes = ("NOT EVALUATED: no bates severity cells found; " if not rows else "")
    notes += (f"{len(rows)} severity cells at tc={tc}; decisive (CI excludes 0): "
             + (where if decisive_rows else "NONE — the pre-registered 'nowhere' "
                "adjudication `no_decisive_regime` (not a failure, and not the "
                "universal `null`; gap visibility is non-monotone by design)"))
    stat = best["diff"] if best else float("nan")
    ci = (best["ci_lo"], best["ci_hi"]) if best else None
    return _verdict("goldilocks_bates", f"bates severity sweep tc={tc}", stat, ci,
                    verdict, notes), rows


# ---------------------------------------------------------------------------
# 3. outputs (tables + memo)
# ---------------------------------------------------------------------------

def _fmt(v, nd: int = 4) -> str:
    x = _num(v)
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{nd}g}"


def write_verdicts_csv(verdicts: list[dict], path) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VERDICT_COLS, extrasaction="ignore")
        w.writeheader()
        for v in verdicts:
            w.writerow({k: v.get(k, "") for k in VERDICT_COLS})
    return str(path)


def write_dose_csv(rows: list[dict], path) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DOSE_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            clean = {}
            for c in DOSE_COLS:
                v = r.get(c, "")
                if isinstance(v, float) and not math.isfinite(v):
                    v = ""
                clean[c] = v
            w.writerow(clean)
    return str(path)


def _gap_row_md(g: dict | None) -> str:
    if g is None:
        return "| n/a | n/a | n/a |"
    return (f"| {_fmt(g['diff'])} | [{_fmt(g['ci_lo'])}, {_fmt(g['ci_hi'])}] | "
            f"{_fmt(g.get('rel'))} |")


def mechanism_memo(mech: dict, verdicts: list[dict], goldilocks_rows: list[dict],
                   dose: dict | None = None) -> str:
    """Descriptive mechanism-adjudication memo (markdown).

    Tables + a STRICTLY descriptive narrative. The only channel language is the
    pre-registered _mechanism_reading; the 'three candidate stories' section
    describes what happened / possible artifacts / surprises WITHOUT choosing a
    channel. The 2x2 whose kappa (in-model x cost corner) is the COST parameter is
    stated as such.
    """
    reading = mech.get("reading", {})
    L = ["# Mechanism adjudication memo", "",
         "Descriptive only. Channel language is limited to the pre-registered "
         "readings (CLAUDE.md 'Mechanism'); nothing below chooses a channel "
         "beyond `_mechanism_reading`.", "",
         "## Pre-registered verdicts", "",
         "| threshold | cell | statistic | CI | verdict |", "|---|---|---|---|---|"]
    for v in verdicts:
        L.append(f"| {v['threshold_id']} | {v['cell']} | {_fmt(v['statistic'])} | "
                 f"[{_fmt(v['ci_lo'])}, {_fmt(v['ci_hi'])}] | **{v['verdict']}** |")

    L += ["", "## rung3 - standard_pinn gap over the TC sweep (misspec m=1.0)", "",
          "loss units; negative = rung3 better (lower CVaR95).", "",
          "| tc | gap (CVaR diff) | pooled 95% CI | rel |", "|---|---|---|---|"]
    for g in mech.get("misspec_gaps", []):
        L.append(f"| {g['tc']} " + _gap_row_md(g))
    if mech.get("inmodel_gaps"):
        L += ["", "### in-model (m=0.0) gap over the TC sweep", "",
              "| tc | gap (CVaR diff) | pooled 95% CI | rel |", "|---|---|---|---|"]
        for g in mech["inmodel_gaps"]:
            L.append(f"| {g['tc']} " + _gap_row_md(g))

    tb = mech.get("two_by_two", {})
    L += ["", "## 2x2: {in-model, misspec} x {tc=0, tc>0}", "",
          "The kappa of the contract's 2x2 (the in-model x cost corner) IS the "
          "COST parameter — this table is the cost axis of the adjudication.", "",
          "| | tc=0 | tc>0 |", "|---|---|---|"]
    for label, key in (("misspec (m=1.0)", "misspec"), ("in-model (m=0.0)", "in_model")):
        cell = tb.get(key, {})
        c0, cp = cell.get("tc0"), cell.get("tc_pos")
        L.append(f"| {label} | {_fmt(c0['diff']) if c0 else 'n/a'} | "
                 f"{_fmt(cp['diff']) if cp else 'n/a'} |")

    tex = mech.get("t_ex", {})
    L += ["", "## Excess-turnover statistic T_ex", "",
          f"t_ex(rung3) - t_ex(standard_pinn) = {_fmt(tex.get('diff'))} "
          f"(seed 95% CI [{_fmt(tex.get('ci_lo'))}, {_fmt(tex.get('ci_hi'))}], "
          f"n={tex.get('n_seeds', 0)} seeds). "
          + ("T_ex CI excludes 0 (turnover moved)." if reading.get("t_ex_reduced")
             else "T_ex CI covers 0 (turnover unmoved -> cost channel cannot be "
                  "credited regardless of PnL)."),
          "", f"d(gap)/d(tc) slope = {_fmt(mech.get('slope_dgap_dtc'))}.",
          "", f"**Pre-registered reading: `{reading.get('reading', 'n/a')}`** "
          f"(present_i={reading.get('present_i')}, present_ii={reading.get('present_ii')}).",
          "  - channel_i  = robustness: gap present at tc=0 (CI excludes 0).",
          "  - channel_ii = cost: gap widens with tc AND T_ex reduced (CI excludes 0).",
          "  - decomposition = both; no_channel = neither (an ADJUDICATED "
          "reading, distinct from the universal `null` = not evaluated)."]

    if goldilocks_rows:
        L += ["", "## Goldilocks (Bates severity sweep)", "",
              "| lambda_j | sigma_j | gap | pooled 95% CI | decisive |",
              "|---|---|---|---|---|"]
        for r in goldilocks_rows:
            L.append(f"| {r['lambda_j']} | {r['sigma_j']} | {_fmt(r['diff'])} | "
                     f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
                     f"{'yes' if r['decisive'] else 'no'} |")

    # ---- three candidate stories (descriptive; NO channel choice) ----
    gaps = mech.get("misspec_gaps", [])
    g0 = gaps[0] if gaps else None
    gmax = gaps[-1] if gaps else None
    dose_note = ""
    if dose is not None:
        dose_note = (f" The dose-response verdict is `{dose['verdict']}` "
                     f"(Spearman statistic {_fmt(dose['statistic'])}).")
    L += ["", "## Three candidate stories", "",
          "Per the roadmap workflow note — described, not adjudicated.", "",
          "**What happened.** The pre-registered reading is "
          f"`{reading.get('reading', 'n/a')}`. The tc=0 gap is "
          f"{_fmt(g0['diff']) if g0 else 'n/a'} (CI "
          f"[{_fmt(g0['ci_lo']) if g0 else 'n/a'}, {_fmt(g0['ci_hi']) if g0 else 'n/a'}]); "
          f"at the top tc tier it is {_fmt(gmax['diff']) if gmax else 'n/a'}. "
          f"T_ex(rung3)-T_ex(std) = {_fmt(tex.get('diff'))}."
          + dose_note, "",
          "**What could be an artifact.** A gap that appears only once TC is "
          "charged, with T_ex covering 0, could be turnover-accounting noise "
          "rather than a robustness effect; a large in-model gap would suggest "
          "the misspecification contrast is contaminated; a monotone TC slope "
          "with tiny per-seed CIs but few seeds may be under-powered. Check the "
          "seed count and the in-model 2x2 corner before reading the cost axis.",
          "",
          "**What is surprising.** Note here any sign flip between the per-seed "
          "and pooled-stratified readings, any regime where the Bates gap is "
          "decisive while the combined-perturbation gap is not, and whether the "
          "T_ex direction agrees with the sign of the TC-slope. These are logged "
          "for human adjudication; the memo does not resolve them."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def run_analysis(confirmatory_dir, out_dir, *, full_dir=None, greek_agg_csv=None,
                 labels_npz=None, pinn_cfg_path="pinn_config.yaml",
                 level: float | None = None, n_boot: int | None = None,
                 seed: int | None = None, contract_path=None, engine_path=None,
                 thresholds: dict | None = None) -> dict:
    """Run every verdict on frozen artifacts and write the P13 outputs:
    results/tables/threshold_verdicts.csv, results/tables/dose_response.csv and
    mechanism_adjudication_memo.md (under out_dir). ABSENT inputs degrade to a
    'null'/blank verdict with a note rather than raising; an artifact that is
    present but UNUSABLE degrades to a distinct 'error' verdict, never to the
    'null' that means "not evaluated" (audit A4).

    The pre-registered thresholds are resolved ONCE here (hb.contract_thresholds
    over the resolved contract) and threaded into every verdict function — no
    verdict in this module reads a Python literal (audit A1/C1)."""
    th = (_default_thresholds(contract_path, engine_path) if thresholds is None
          else thresholds)
    conf = Path(confirmatory_dir)
    conf_pnl = conf / PNL_DIR
    conf_ps = conf / PER_SEED_CSV
    full = Path(full_dir) if full_dir else None
    full_pnl = (full / PNL_DIR) if full else None
    full_ps = (full / PER_SEED_CSV) if full else None
    tables = Path(out_dir) / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    verdicts: list[dict] = []

    def _guard(fn, tid, cell):
        try:
            return fn()
        except Exception as exc:
            return _failure_verdict(exc, tid, cell)

    verdicts.append(_guard(
        lambda: confirmatory_cell(conf_pnl, thresholds=th, level=level,
                                  n_boot=n_boot, seed=seed),
        "confirmatory_cell", "combined m=1.0 tc=0.01"))
    verdicts.append(_guard(
        lambda: order_attribution(conf_pnl, thresholds=th, level=level,
                                  n_boot=n_boot, seed=seed),
        "order_attribution", "rung2-vs-rung1 combined m=1.0 tc=0.01"))

    dose_vd = None
    dose_rows: list[dict] = []
    if full_ps and labels_npz:
        try:
            dose_vd, dose_rows = dose_response(full_ps, labels_npz, pinn_cfg_path,
                                               thresholds=th, n_boot=n_boot, seed=seed)
        except Exception as exc:
            dose_vd = _failure_verdict(exc, "dose_response", "combined m=1.0 tc=0.01")
    else:
        dose_vd = _verdict("dose_response", "combined m=1.0 tc=0.01", float("nan"),
                           None, "null", "not evaluated: full-sweep CSV / labels absent")
    verdicts.append(dose_vd)

    verdicts.append(_guard(
        lambda: ood_greek_thresholds(greek_agg_csv, thresholds=th),
        "ood_greek_thresholds", "rung3 near_feller/strong_neg_corr"))
    # SECONDARY (non-binding): the causal rung of the rung1->rung2 order claim,
    # same thresholds/CI as rung3; informational only — gates nothing.
    verdicts.append(_guard(
        lambda: ood_greek_thresholds(greek_agg_csv, thresholds=th, arm="rung2",
                                     binding=False),
        "ood_greek_thresholds_rung2_secondary",
        "rung2 near_feller/strong_neg_corr (SECONDARY)"))
    verdicts.append(_guard(
        lambda: sakuma_null_consistency(conf_pnl, thresholds=th, level=level,
                                        n_boot=n_boot, seed=seed),
        "sakuma_null_consistency", "combined m=0.0 tc=0.0"))

    mech = _guard(
        lambda: mechanism_adjudication(conf_pnl, conf_ps, thresholds=th, level=level,
                                       n_boot=n_boot, seed=seed),
        "mechanism_adjudication", "combined m=1.0 TC-sweep")
    verdicts.append({k: mech[k] for k in VERDICT_COLS})

    gold_vd = None
    gold_rows: list[dict] = []
    if full_pnl and full_ps:
        try:
            gold_vd, gold_rows = goldilocks_bates(full_pnl, full_ps, thresholds=th,
                                                  level=level, n_boot=n_boot, seed=seed)
        except Exception as exc:
            gold_vd = _failure_verdict(exc, "goldilocks_bates",
                                       "bates severity sweep tc=0.01")
    else:
        gold_vd = _verdict("goldilocks_bates", "bates severity sweep tc=0.01",
                           float("nan"), None, "null",
                           "not evaluated: full-sweep artifacts absent")
    verdicts.append(gold_vd)

    verdicts_path = write_verdicts_csv(verdicts, tables / "threshold_verdicts.csv")
    dose_path = write_dose_csv(dose_rows, tables / "dose_response.csv")
    memo = mechanism_memo(mech, verdicts, gold_rows, dose_vd)
    memo_path = Path(out_dir) / "mechanism_adjudication_memo.md"
    memo_path.write_text(memo)

    return {"verdicts": verdicts, "dose_rows": dose_rows, "goldilocks_rows": gold_rows,
            "mechanism": mech,
            "paths": {"threshold_verdicts": verdicts_path, "dose_response": dose_path,
                      "mechanism_memo": str(memo_path)}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--confirmatory-dir", required=True,
                    help="run_confirmatory out_dir/confirmatory")
    ap.add_argument("--full-dir", default=None, help="run_full_sweep out_dir/full_sweep")
    ap.add_argument("--greek-agg-csv", default=None, help="P11 ood_param_greeks_agg.csv")
    ap.add_argument("--labels-npz", default=None, help="frozen label artifact labels.npz")
    ap.add_argument("--pinn-cfg", default="pinn_config.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contract", default=None,
                    help="benchmark contract (default: heston_benchmark_v6.yaml beside this module)")
    ap.add_argument("--engine", default=None,
                    help="engine supplement (default: hedging_config.yaml beside this module)")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="override the contract's risk.bootstrap_B")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the contract's meta.global_seed")
    args = ap.parse_args(argv)
    res = run_analysis(args.confirmatory_dir, args.out_dir, full_dir=args.full_dir,
                       greek_agg_csv=args.greek_agg_csv, labels_npz=args.labels_npz,
                       pinn_cfg_path=args.pinn_cfg, n_boot=args.n_boot, seed=args.seed,
                       contract_path=args.contract, engine_path=args.engine)
    for v in res["verdicts"]:
        print(f"{v['threshold_id']:>24s}: {v['verdict']}")
    for k, p in res["paths"].items():
        print(f"wrote {k}: {p}")
    return res


if __name__ == "__main__":
    main()
