#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# PDEs to test
CONFIGS=("base.yaml" "burgers.yaml" "allen_cahn.yaml" "darcy.yaml")
PDES=("Diffusion" "Burgers" "Allen_Cahn" "Darcy")
SEEDS=(42 43 44)
KEEP_REGS=(0 1)

# Common params
STEPS=15000
LBFGS_STEPS=5000

echo "Submitting 24 independent L-BFGS ablation jobs to SLURM..."

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    pde_name="${PDES[$i]}"
    
    for seed in "${SEEDS[@]}"; do
        for keep_reg in "${KEEP_REGS[@]}"; do
            
            run_name="LBFGS_${pde_name}_reg${keep_reg}_seed${seed}"
            
            # Submit a separate job for each configuration using heredoc
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${run_name}
#SBATCH --partition=hawkgpu
#SBATCH --output=logs/${run_name}_%j.out
#SBATCH --error=logs/${run_name}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Load environment
module load miniconda3/24.7.1
source ~/.bashrc

conda activate s2PINN
cd ~/s2PINN
export PYTHONPATH=src

echo "=========================================================="
echo "Running \$SLURM_JOB_NAME on GPU"
echo "=========================================================="

if [ "${keep_reg}" = "1" ]; then
    python src/spinn/train.py \
        --config src/configs/${config} \
        --steps ${STEPS} \
        --lbfgs_steps ${LBFGS_STEPS} \
        --seed ${seed} \
        --lbfgs_keep_reg \
        --run_name ${run_name}
else
    python src/spinn/train.py \
        --config src/configs/${config} \
        --steps ${STEPS} \
        --lbfgs_steps ${LBFGS_STEPS} \
        --seed ${seed} \
        --run_name ${run_name}
fi

EOT
            echo "Submitted job: ${run_name}"
            # Sleep briefly to avoid hammering the SLURM scheduler
            sleep 0.5
        done
    done
done

echo "=========================================================="
echo "All 24 L-BFGS ablation jobs have been submitted to SLURM."
echo "You can monitor them using: squeue -u \$USER"
