#!/bin/bash
# Submit inverse Burgers experiments across 3 seeds on separate GPUs

for SEED in 42 43 44; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=INV_burgers_s${SEED}
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

echo "Starting Inverse Burgers Job (seed=${SEED})..."
python -u src/spinn/train_inverse.py \
    --config src/configs/inverse_burgers.yaml \
    --run_name "inverse_burgers_seed${SEED}" \
    --seed ${SEED}
echo "Job complete (seed=${SEED})."
EOF
    echo "Submitted seed=${SEED}"
done
