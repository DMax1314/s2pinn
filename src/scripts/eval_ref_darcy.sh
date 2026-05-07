#!/bin/bash
# ============================================================
# FEniCS reference solver for Darcy flow (stochastic + deterministic)
#
# Usage:
#   cd ~/s2PINN/src
#   sbatch scripts/eval_ref_darcy.sh
#
# Produces:
#   results/darcy_det_ref.npz    — deterministic (Z=0) snapshots
#   results/darcy_uq_R3_ref.npz — E[u], Var[u] via quadrature (R=3, p=5)
# ============================================================

#SBATCH --job-name=darcy_fem_ref
#SBATCH --partition=hawkmem
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=04:00:00

source ~/.bashrc
module load miniconda3/24.7.1
conda activate s2PINN

export PYTHONUNBUFFERED=1

cd ~/s2PINN/src
mkdir -p ../results

echo "=== Darcy FEM reference solver ==="

# 1. Deterministic reference (Z=0)
echo "--- Deterministic solve (mesh=64) ---"
python solvers/fenics/darcy2d_ref.py \
    --mesh 64 \
    --num_steps 100 \
    --sigma 0.5 \
    --gamma 0.3 \
    --output ../results/darcy_det_ref.npz

# 2. Stochastic reference (R=3, p=5 quadrature => 6^3=216 nodes)
echo "--- Stochastic solve (R=3, p=5, mesh=64) ---"
python solvers/fenics/darcy2d_ref.py \
    --mesh 64 \
    --num_steps 100 \
    --rand_R 3 \
    --rand_p 5 \
    --sigma 0.5 \
    --gamma 0.3 \
    --output ../results/darcy_uq_R3_ref.npz

# 3. Stochastic reference (R=5, p=3 quadrature => 4^5=1024 nodes)
echo "--- Stochastic solve (R=5, p=3, mesh=64) ---"
python solvers/fenics/darcy2d_ref.py \
    --mesh 64 \
    --num_steps 100 \
    --rand_R 5 \
    --rand_p 3 \
    --sigma 0.5 \
    --gamma 0.3 \
    --output ../results/darcy_uq_R5_ref.npz

echo "=== Done ==="
