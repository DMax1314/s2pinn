#!/bin/bash
source ~/.bashrc
module load miniconda3/24.7.1
conda activate s2PINN

export PYTHONUNBUFFERED=1

cd ~/s2PINN/src
mkdir -p ../logs

python experiments.py \
    --config configs/base.yaml \
    "$@"
