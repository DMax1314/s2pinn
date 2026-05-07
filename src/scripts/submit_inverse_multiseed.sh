#!/bin/bash
# Submit inverse problem experiments across 3 seeds on separate GPUs

for SEED in 42 43 44; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=s2pinn_inv_s${SEED}
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

echo "Starting Inverse Problem Job (seed=${SEED})..."
python -u src/spinn/train_inverse.py \
    --config src/configs/inverse_diffusion.yaml \
    --run_name "inverse_diffusion_seed${SEED}" \
    --seed ${SEED} \
    --lbfgs_steps 2000 \
    --lbfgs_keep_reg
echo "Job complete (seed=${SEED})."
EOF
    echo "Submitted seed=${SEED}"
done
