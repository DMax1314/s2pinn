#!/bin/bash
#SBATCH --job-name=s2pinn_inv
#SBATCH --partition=hawkgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/INV_%x_%j.out
#SBATCH --error=logs/INV_%x_%j.err

source ~/.bashrc
module load miniconda3/24.7.1
conda activate s2PINN

export PYTHONPATH=src
export PYTHONUNBUFFERED=1

echo "Starting Inverse Problem Job..."
python -u src/spinn/train_inverse.py --config src/configs/inverse_diffusion.yaml --run_name "inverse_diffusion_seed42" --seed 42 --lbfgs_steps 2000 --lbfgs_keep_reg
echo "Job complete."
