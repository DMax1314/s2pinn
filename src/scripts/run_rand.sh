#!/bin/bash
# Stochastic sPINN experiments for four (R, p) groups and ablation axes

set -e

CONFIG="configs/base.yaml"
LOGDIR="logs/rand"

declare -a RP=("3 3" "4 4" "6 3" "6 4")
declare -a WPROJ=("0" "0.3" "1" "3")
declare -a RANK=("8" "16" "32")
declare -a COEFF=("cp" "full")
declare -a FACORTH=("0" "0.05" "0.1")
declare -a QUAD=("tensor" "smolyak")
declare -a COLLOC=("8000" "16000" "32000" "64000")
declare -a SEEDS=("0" "1" "2")

for rp in "${RP[@]}"; do
  set -- $rp
  R=$1
  P=$2
  for wproj in "${WPROJ[@]}"; do
    for rank in "${RANK[@]}"; do
      for coeff in "${COEFF[@]}"; do
        for facorth in "${FACORTH[@]}"; do
          for quad in "${QUAD[@]}"; do
            for colloc in "${COLLOC[@]}"; do
              for seed in "${SEEDS[@]}"; do
                TAG="R${R}_p${P}_wproj${wproj}_rank${rank}_${coeff}_facorth${facorth}_${quad}_colloc${colloc}_seed${seed}"
                python experiments.py --config $CONFIG --pde diffusion2d --coeff $coeff --rank $rank --R $R --p $P --quadrature $quad --seed $seed --log_dir $LOGDIR/$TAG
              done
            done
          done
        done
      done
    done
  done
done
