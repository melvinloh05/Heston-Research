#!/bin/zsh
# CODE_AUDIT_2026-08-20 action 2 — the confirmatory headline as a function of lambda_pde.
#
# Known endpoints: 0.01 (registered) -> +31.50% ; 0.0 (robustness row) -> +2.86%.
# This fills {1e-4, 1e-3, 3e-3} for the two arms the confirmatory contrast names,
# at the confirmatory seed count, then hedges each. Exploratory sensitivity, NOT a
# second confirmatory test.
set -e
cd "/Users/melvin/Documents/Heston Research"
D=data/frozen/v6-labels-20260812
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
date
for L in 1e-4 1e-3 3e-3; do
  OUT=results/grid_lampde_$L; LOG=$OUT/_logs; mkdir -p $LOG
  i=0
  for seed in 42 43 44 45 46 47 48 49 50 51; do
    for arm in rung3_delta_gamma_vega standard_pinn; do
      dest=$OUT/$arm/s$seed
      [[ -f $dest/best.pt ]] && { echo "SKIP $L $arm s$seed"; continue; }
      ( python train.py --arm $arm --seed $seed \
          --data $D/train_val/train_val_labels.npz --lambdas lambdas_pde_$L.yaml \
          --out $dest --steps 20000 --matched-epochs --device cpu \
          > $LOG/$arm.s$seed.log 2>&1 && echo "OK   $L $arm s$seed $(date +%H:%M:%S)" \
          || echo "FAIL $L $arm s$seed" ) &
      i=$((i+1)); (( i % 2 == 0 )) && wait
    done
  done
  wait
  echo "== $L trained: $(find $OUT -name best.pt | wc -l | tr -d ' ')/20 ; hedging =="
  python robustness_hedge.py --ckpt-root $OUT --out-dir results/hedging_lampde_$L \
    2>/dev/null || python - <<PY
import sys; sys.path.insert(0,".")
import copy, Hedging_backtest as hb, run_hedging as rh
cfg = hb.resolve_config('heston_benchmark_v6.yaml','hedging_config.yaml')
cfg["benchmark"]["hedging_simulation"]["transaction_costs"]["tiers"]=[0.0,0.00005,0.005,0.01,0.02]
m=cfg["benchmark"]["meta"]; g,n=int(m["global_seed"]),int(m["seeds_confirmatory_cell"])
prog=copy.deepcopy(cfg); prog["derived"]["seeds"]=[g+i for i in range(n)]
prog["engine"]["risk"]["persist_pnl"]=True
mis=prog["benchmark"]["hedging_simulation"]["misspecification"]
mis["perturbations"]={"combined":mis["perturbations"]["combined"]}; mis["cross_model"]=[]
prog["engine"]["misspecification"]["magnitudes"]=[0.0,1.0]
print(rh._run_program(prog,"$OUT","results/hedging_lampde_$L/confirmatory",
                      ["standard_pinn","rung3"],"lampde_$L")["n_ran"], "cells")
PY
done
date; echo "CURVE COMPLETE"
