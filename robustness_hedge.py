"""Hedge the robustness-row grid: the confirmatory contrast at the rung3-sourced
lambda_pde = 0.0, at the confirmatory seed count.

Only the two arms the contract's robustness_row names. rung1/rung2 were not
retrained at this lambda -- the row is about the HEADLINE contrast, not the ladder.
Registered TC tiers plus the A&T-anchored exploratory ones, so the robustness row
carries the same cost profile as the headline it is compared against.
"""
import time
import copy
import Hedging_backtest as hb, run_hedging as rh

while len(list(__import__('pathlib').Path('results/grid_robustness').rglob('best.pt'))) < 20:
    time.sleep(10)
print("all 20 checkpoints present; hedging", flush=True)

cfg = hb.resolve_config('heston_benchmark_v6.yaml', 'hedging_config.yaml')
cfg["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"] = [
    0.0, 0.00005, 0.0025, 0.005, 0.01, 0.02]
meta = cfg["benchmark"]["meta"]
g, n = int(meta["global_seed"]), int(meta["seeds_confirmatory_cell"])

prog = copy.deepcopy(cfg)
prog["derived"]["seeds"] = [g + i for i in range(n)]
prog["engine"]["risk"]["persist_pnl"] = True
mis = prog["benchmark"]["hedging_simulation"]["misspecification"]
mis["perturbations"] = {"combined": mis["perturbations"]["combined"]}
mis["cross_model"] = []
prog["engine"]["misspecification"]["magnitudes"] = [0.0, 1.0]

res = rh._run_program(prog, 'results/grid_robustness',
                      'results/hedging_robustness/confirmatory',
                      ["standard_pinn", "rung3"], "robustness_row")
print({k: res[k] for k in ("n_ran", "n_skipped", "seeds")})
