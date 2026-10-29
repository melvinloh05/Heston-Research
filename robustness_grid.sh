#!/bin/zsh
# Contract lambda_selection.robustness_row, stage 2: retrain the confirmatory
# contrast arms at the rung3-sourced lambda_pde = 0.0, then hedge them.
#
# Only the two arms the row names (rung3 vs standard_pinn), only the confirmatory
# seed set. Separate ckpt root so the registered grid is never overwritten.
set -e
script_dir=${0:A:h}
cd -- "$script_dir"
D=data/frozen/v6-labels-20260812
if [[ ! -f "$D/train_val/train_val_labels.npz" ]]; then
  print -u2 "Missing label artifact: $D/train_val/train_val_labels.npz"
  print -u2 "See docs/REPRODUCIBILITY.md for data preparation before running this experiment."
  exit 1
fi
OUT=results/grid_robustness
LOG=$OUT/_logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p $OUT $LOG
date
i=0
for seed in 42 43 44 45 46 47 48 49 50 51; do
  for arm in rung3_delta_gamma_vega standard_pinn; do
    dest=$OUT/$arm/s$seed
    if [[ -f $dest/best.pt ]]; then echo "SKIP $arm s$seed"; continue; fi
    ( python train.py --arm $arm --seed $seed \
        --data $D/train_val/train_val_labels.npz \
        --lambdas lambdas_robustness_row.yaml \
        --out $dest --steps 20000 --matched-epochs --device cpu \
        > $LOG/$arm.s$seed.log 2>&1 \
        && echo "OK   $arm s$seed $(date +%H:%M:%S)" \
        || echo "FAIL $arm s$seed -> $LOG/$arm.s$seed.log" ) &
    i=$((i+1)); if (( i % 2 == 0 )); then wait; fi
  done
done
wait
date
echo "checkpoints: $(find $OUT -name best.pt | wc -l | tr -d ' ') / 20"
