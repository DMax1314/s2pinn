#!/bin/bash
# Submit only reg=1 L-BFGS ablation jobs with unbuffered Python output

mkdir -p ~/s2PINN/logs

CONFIGS=(base.yaml burgers.yaml allen_cahn.yaml darcy.yaml)
PDES=(Diffusion Burgers Allen_Cahn Darcy)
SEEDS=(42 43 44)
STEPS=15000
LBFGS_STEPS=5000

echo "Submitting 12 L-BFGS reg=1 jobs..."

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    pde_name="${PDES[$i]}"

    for seed in "${SEEDS[@]}"; do
        run_name="LBFGS_${pde_name}_reg1_seed${seed}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${run_name}
#SBATCH --partition=hawkgpu
#SBATCH --output=${HOME}/s2PINN/logs/${run_name}_%j.out
#SBATCH --error=${HOME}/s2PINN/logs/${run_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load miniconda3/24.7.1
source ~/.bashrc
conda activate s2PINN
cd \${HOME}/s2PINN
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo "Running ${run_name} on GPU"
echo "=========================================================="

python -u src/spinn/train.py \
    --config src/configs/${config} \
    --steps ${STEPS} \
    --lbfgs_steps ${LBFGS_STEPS} \
    --lbfgs_keep_reg \
    --seed ${seed} \
    --uq \
    --no_wandb
EOT
        echo "Submitted: ${run_name}"
        sleep 0.3
    done
done

echo "All 12 reg=1 jobs submitted."
echo "Monitor with: squeue -u \$USER"
