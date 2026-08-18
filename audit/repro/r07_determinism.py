"""R07 — determinism of two Tier-1 entry points, twice in one process.

(A) Hedging_backtest.run_headline on one trimmed cell with a deterministic
    toy provider set: identical rows twice in the same process.
(B) analyze_results.paired_ci_from_npz on a fixed npz bank: identical pooled
    statistic and CI twice in the same process.

The cross-process leg is run by r07_determinism_run.sh, which invokes this
file twice and diffs the digests.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
import analyze_results as ar  # noqa: E402
import Hedging_backtest as hb  # noqa: E402


class Toy:
    """Deterministic provider: pure function of (S, v, tau, K)."""

    def __init__(self, bump=0.0):
        self.bump = bump

    def evaluate(self, S, v, tau, K):
        S = np.asarray(S, float)
        v = np.asarray(v, float)
        m = (S - K) / (K * np.sqrt(np.maximum(v, 1e-8) * max(tau, 1e-8)))
        d = 0.5 * (1.0 + np.tanh(m)) + self.bump
        return {"price": np.maximum(S - K, 0.0) + 2.0 + self.bump,
                "delta": d, "gamma": (1.0 - np.tanh(m) ** 2) / (2.0 * K)}


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_engine_once():
    cfg = hb.resolve_config()
    mis = cfg["benchmark"]["hedging_simulation"]["misspecification"]
    mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
    mis["cross_model"] = []
    cfg["engine"]["misspecification"]["magnitudes"] = [1.0]
    cfg["engine"]["simulation"]["n_paths"] = 400
    cfg["derived"]["seeds"] = [42, 43]
    cfg["engine"]["risk"]["bootstrap_B"] = 50
    provs = {"oracle": Toy(0.0), "standard_pinn": Toy(0.01), "rung3": Toy(0.004)}
    rows = hb.run_headline(cfg, provs, out_dir=None)
    return [{k: v for k, v in r.items() if not str(k).startswith("_")} for r in rows]


def run_analysis_once():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        tag = {"direction": "combined", "magnitude": 1.0, "lambda_j": "",
               "sigma_j": "", "in_model": False, "sweep": "perturbation"}
        for s in (42, 43, 44):
            rng = np.random.default_rng(7000 + s)
            base = rng.standard_normal(1500)
            np.savez_compressed(d / (hb._cell_slug(tag, s) + ".npz"), **{
                "standard_pinn__tc0.01": base,
                "rung3__tc0.01": base + 0.4})
        res = ar.paired_ci_from_npz(d, "rung3", "standard_pinn", 0.01, n_boot=200)
        return {"pooled": res["pooled"],
                "per_seed": [{k: r[k] for k in ("seed", "diff", "ci_lo", "ci_hi")}
                             for r in res["per_seed"]]}


e1, e2 = run_engine_once(), run_engine_once()
a1, a2 = run_analysis_once(), run_analysis_once()

de, da = _digest(e1), _digest(a1)
print(f"engine   rows: {len(e1)}   in-process repeat identical: {_digest(e1) == _digest(e2)}")
print(f"analysis      : pooled diff={a1['pooled']['diff']:.10g}  "
      f"in-process repeat identical: {_digest(a1) == _digest(a2)}")
print(f"ENGINE_DIGEST={de}")
print(f"ANALYSIS_DIGEST={da}")
