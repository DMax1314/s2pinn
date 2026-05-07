#!/bin/bash
#SBATCH --job-name=viz_diff
#SBATCH --partition=hawkgpu
#SBATCH --output=../logs/viz_diff_%j.out
#SBATCH --error=../logs/viz_diff_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

source ~/.bashrc
module load miniconda3/24.7.1
conda activate s2PINN

python scripts/viz_solution.py --checkpoint checkpoints/ckpt/model_seed42.pt
