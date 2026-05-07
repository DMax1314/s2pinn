#!/bin/bash
# PDEBench experiments: Darcy-2D and Reaction-Diffusion-2D

set -e

CONFIG="configs/base.yaml"
LOGDIR="logs/pdebench"
declare -a BENCH=("darcy2d" "reactdiff2d")
declare -a MODE=("supervised" "semisupervised")
declare -a RATIO=("0.1" "0.25" "0.5")
declare -a SEEDS=("0" "1" "2")

# Baselines: FNO, U-Net, DeepONet (assume scripts or modules available)
for bench in "${BENCH[@]}"; do
  for ratio in "${RATIO[@]}"; do
    for seed in "${SEEDS[@]}"; do
      python scripts/run_fno.py --bench $bench --train_ratio $ratio --seed $seed --log_dir $LOGDIR/fno
      python scripts/run_unet.py --bench $bench --train_ratio $ratio --seed $seed --log_dir $LOGDIR/unet
      python scripts/run_deeponet.py --bench $bench --train_ratio $ratio --seed $seed --log_dir $LOGDIR/deeponet
    done
  done
done

# sPINN: supervised and semisupervised
for bench in "${BENCH[@]}"; do
  for mode in "${MODE[@]}"; do
    for ratio in "${RATIO[@]}"; do
      for seed in "${SEEDS[@]}"; do
        TAG="${bench}_${mode}_ratio${ratio}_seed${seed}"
        python experiments.py --config $CONFIG --pde $bench --coeff cp --rank 16 --R 3 --p 3 --quadrature tensor --seed $seed --log_dir $LOGDIR/$TAG --mode $mode --train_ratio $ratio
      done
    done
  done
done
