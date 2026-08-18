"""R02 — two behaviours of analyze_results.py verdict funcs, driven end-to-end
on synthetic-but-well-formed PnL npz banks written in the engine's own layout.

(a) confirmatory_cell() returns verdict='pass' on a partial run carrying only
    3 of the pre-registered 10 seeds, with nothing in the verdict row flagging
    the shortfall (n_seeds appears only inside the free-text notes).
(b) _mechanism_reading() labels a tc=0 gap in the WRONG direction (rung3 WORSE
    than standard_pinn, CI strictly above 0) as reading='channel_i', i.e. the
    robustness channel, because the test is |CI excludes 0| with no sign check.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/melvin/Documents/Heston Research")
import analyze_results as ar  # noqa: E402
import Hedging_backtest as hb  # noqa: E402

TC = 0.01
N_PATH = 4000


def write_bank(dirpath: Path, seeds, shift, magnitude=1.0, tc_list=(TC,)):
    """One npz per (cell, seed) in Hedging_backtest._cell_slug layout.

    rung3 PnL = base + shift (shift>0 => rung3 has HIGHER PnL => LOWER loss
    => lower CVaR => rung3 better).
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    tag = {"direction": "combined", "magnitude": magnitude, "lambda_j": "",
           "sigma_j": "", "in_model": magnitude == 0, "sweep": "perturbation"}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        base = rng.standard_normal(N_PATH)
        bank = {}
        for tc in tc_list:
            bank[f"standard_pinn__tc{tc}"] = base
            bank[f"rung3__tc{tc}"] = base + shift
            bank[f"rung1__tc{tc}"] = base
            bank[f"rung2__tc{tc}"] = base + shift
        np.savez_compressed(dirpath / (hb._cell_slug(tag, s) + ".npz"), **bank)


print("=" * 74)
print("(a) confirmatory_cell on 3 seeds vs the pre-registered 10")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    for seeds in ([42, 43, 44], list(range(42, 52))):
        d = Path(td) / f"n{len(seeds)}"
        write_bank(d, seeds, shift=0.60)
        v = ar.confirmatory_cell(d, tc=TC, n_boot=300)
        print(f"\nseeds present = {len(seeds)}  {seeds}")
        print(f"  verdict   : {v['verdict']}")
        print(f"  statistic : {v['statistic']:.4f}  (pooled rel improvement)")
        print(f"  CI        : [{v['ci_lo']:.4g}, {v['ci_hi']:.4g}]")
        print(f"  row keys  : {sorted(v)}")
        print(f"  n_seeds appears only in notes: "
              f"{'yes' if 'seeds;' in v['notes'] else 'no'}")
    print("\n-> the 3-seed run yields the SAME verdict string as the 10-seed run;")
    print("   no field of the verdict record carries the seed count.")

print()
print("=" * 74)
print("(b) _mechanism_reading with the tc=0 gap in the WRONG direction")
print("=" * 74)
# gap = cvar(rung3) - cvar(standard).  diff > 0 and CI strictly above 0 means
# rung3 is significantly WORSE at zero transaction cost.
worse = [{"tc": 0.0,  "diff": +0.42, "ci_lo": +0.20, "ci_hi": +0.65},
         {"tc": 0.01, "diff": +0.30, "ci_lo": +0.10, "ci_hi": +0.50},
         {"tc": 0.02, "diff": +0.25, "ci_lo": +0.05, "ci_hi": +0.45}]
t_ex_flat = {"diff": 0.001, "ci_lo": -0.02, "ci_hi": 0.02}
print("gaps (rung3 - standard, loss units; POSITIVE = rung3 worse):")
for g in worse:
    print(f"   tc={g['tc']:<5} diff={g['diff']:+.3f}  CI=[{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]")
print("T_ex diff CI covers 0 (turnover unmoved).")
print("\n_mechanism_reading ->", ar._mechanism_reading(worse, t_ex_flat))

better = [{"tc": 0.0,  "diff": -0.42, "ci_lo": -0.65, "ci_hi": -0.20},
          {"tc": 0.01, "diff": -0.30, "ci_lo": -0.50, "ci_hi": -0.10},
          {"tc": 0.02, "diff": -0.25, "ci_lo": -0.45, "ci_hi": -0.05}]
print("\nthe MIRRORED case (rung3 genuinely better at every tier):")
print("_mechanism_reading ->", ar._mechanism_reading(better, t_ex_flat))
print("\n-> identical reading 'channel_i' for 'rung3 is robustly BETTER at 0% TC'")
print("   and 'rung3 is robustly WORSE at 0% TC'.")
