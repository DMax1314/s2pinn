#!/bin/bash
#SBATCH --job-name=s2pinn
#SBATCH --partition=hawkgpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

source ~/.bashrc
module load miniconda3/24.7.1
conda activate s2PINN

export PYTHONUNBUFFERED=1

cd ~/s2PINN/src
mkdir -p ../logs

python experiments.py \
    --config configs/base.yaml \
    "$@"
