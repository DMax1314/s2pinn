#!/bin/bash
# ============================================================
# MVP-E3: MC-Dropout PINN Baseline
# 4 PDEs × 5 seeds = 20 jobs
#
# MC-Dropout uncertainty quantification baseline using
# Vanilla PINN architecture with dropout=0.1 and 50 MC samples.
#
# Spreads jobs across hawkgpu (T4), haswell/1080, haswell/2080Ti
# via round-robin for maximum overnight throughput.
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_mvp_e3.sh
# ============================================================

set -euo pipefail

# --- GPU pool (round-robin across all 3) ---
# Each entry: "partition [constraint_flag]"
GPU_POOL=(
    "hawkgpu"
    "haswell --constraint=gpu:1080"
    "haswell --constraint=gpu:2080"
)
GPU_POOL_SIZE=${#GPU_POOL[@]}

# Time limits
TIME_SHORT="04:00:00"    # diffusion/AC/burgers (20K steps)
TIME_DARCY="06:00:00"    # Darcy (30K steps, higher w_bc/w_ic)
CPUS=4

# Script path
BASELINE_SCRIPT="scripts/run_baseline.sh"

# PDE configs
declare -A PDE_CONFIGS
PDE_CONFIGS[diff]="configs/base.yaml"
PDE_CONFIGS[ac]="configs/allen_cahn.yaml"
PDE_CONFIGS[burg]="configs/burgers.yaml"
PDE_CONFIGS[darcy]="configs/darcy.yaml"

# All 5 seeds for fresh comparison
ALL_SEEDS=(42 123 456 789 1024)

# MC-dropout hyperparameters
DROPOUT=0.1
MC_SAMPLES=50

mkdir -p ../logs

COUNT=0

get_time() {
    local pde="$1"
    local method="$2"
    if [ "${pde}" = "darcy" ]; then
        echo "${TIME_DARCY}"
    else
        echo "${TIME_SHORT}"
    fi
}

submit_job() {
    local job_name="$1"
    local script="$2"
    local time_limit="$3"
    shift 3
    local extra_args="$@"

    # Round-robin GPU selection
    local gpu_idx=$((COUNT % GPU_POOL_SIZE))
    local gpu_entry="${GPU_POOL[${gpu_idx}]}"
    local part="${gpu_entry%% *}"           # first word = partition
    local constraint="${gpu_entry#* }"      # rest = constraint (if any)
    if [ "${constraint}" = "${part}" ]; then
        constraint=""                       # no constraint if single word
    fi

    echo "  [${part} ${constraint}] ${job_name}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${part}" \
        ${constraint} \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${time_limit}" \
        "${script}" ${extra_args}
    COUNT=$((COUNT+1))
}

echo "================================================"
echo " MVP-E3: MC-Dropout PINN Baseline"
echo " GPU pool: hawkgpu(T4), haswell(1080), haswell(2080Ti)"
echo " MC-Dropout: dropout=0.1, mc_samples=50"
echo "================================================"

# ============================================================
# MC-DROPOUT PINN (dropout=0.1, mc_samples=50, no projection)
#    4 PDEs × 5 seeds = 20 jobs
# ============================================================
echo ""
echo "--- MC-Dropout PINN (dropout=0.1, mc_samples=50, w_proj=0.0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "mcdrop")
    for seed in "${ALL_SEEDS[@]}"; do
        submit_job "mcdrop_${pde_key}_R5_s${seed}" "${BASELINE_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 \
            --dropout ${DROPOUT} --mc_samples ${MC_SAMPLES} \
            --seed ${seed}
    done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================"
echo " MVP-E3 submitted: ${COUNT} total jobs (4 PDEs × 5 seeds)"
echo "  MC-Dropout PINN:  20 jobs"
echo " Seeds: 42, 123, 456, 789, 1024"
echo " Hyperparameters: dropout=${DROPOUT}, mc_samples=${MC_SAMPLES}"
echo " Monitor: squeue -u \$USER"
echo "================================================"
