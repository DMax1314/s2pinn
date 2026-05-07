#!/bin/bash
# Evaluate model checkpoints: Grid-Rel-L2, QoI, E/Var

set -e

LOGDIR="logs"
OUTDIR="eval_outputs"
mkdir -p $OUTDIR

declare -a EXPERIMENTS=("det" "rand" "pdebench")
declare -a SEEDS=("0" "1" "2")

for exp in "${EXPERIMENTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    CKPT="$LOGDIR/$exp/model_seed${seed}.pt"
    OUT="$OUTDIR/eval_${exp}_seed${seed}.json"
    python experiments.py --eval_ckpt $CKPT --output $OUT
  done
done
