#!/bin/bash
# ============================================================
# MVP-E1: 5-Seed Reruns for Tables 1-3
# All 5 methods × 4 PDEs × 3 new seeds (456, 789, 1024)
#
# Extends existing 2-seed (42, 123) runs to 5 seeds total
# for improved statistical confidence in main tables.
#
# Spreads jobs across hawkgpu (T4), haswell/1080, haswell/2080Ti
# via round-robin for maximum overnight throughput.
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_mvp_e1.sh
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

# Script paths
S2P_SCRIPT="scripts/run_experiment.sh"
DARCY_SCRIPT="scripts/run_darcy.sh"
BASELINE_SCRIPT="scripts/run_baseline.sh"
DEEPONET_SCRIPT="scripts/run_deeponet.sh"

# PDE configs
declare -A PDE_CONFIGS
PDE_CONFIGS[diff]="configs/base.yaml"
PDE_CONFIGS[ac]="configs/allen_cahn.yaml"
PDE_CONFIGS[burg]="configs/burgers.yaml"
PDE_CONFIGS[darcy]="configs/darcy.yaml"

# NEW seeds only (existing 42, 123 already done)
NEW_SEEDS=(456 789 1024)

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
echo " MVP-E1: 5-Seed Reruns for Tables 1-3"
echo " GPU pool: hawkgpu(T4), haswell(1080), haswell(2080Ti)"
echo " New seeds: 456, 789, 1024 (adds to existing 42, 123)"
echo "================================================"

# ============================================================
# 1. S²-PINN WITH PROJECTION (w_proj=1.0, default)
#    4 PDEs × 3 seeds = 12 jobs
# ============================================================
echo ""
echo "--- S²-PINN with projection (w_proj=1.0) ---"
for pde_key in diff ac burg; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "s2p")
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "s2p_${pde_key}_proj_s${seed}" "${S2P_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --seed ${seed}
    done
done
# Darcy separately (uses run_darcy.sh)
for seed in "${NEW_SEEDS[@]}"; do
    submit_job "s2p_darcy_proj_s${seed}" "${DARCY_SCRIPT}" "${TIME_DARCY}" \
        --R 5 --seed ${seed}
done

# ============================================================
# 2. S²-PINN WITHOUT PROJECTION (w_proj=0.0)
#    4 PDEs × 3 seeds = 12 jobs
# ============================================================
echo ""
echo "--- S²-PINN without projection (w_proj=0.0) ---"
for pde_key in diff ac burg; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "s2p")
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "s2p_${pde_key}_nopr_s${seed}" "${S2P_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 --seed ${seed}
    done
done
for seed in "${NEW_SEEDS[@]}"; do
    submit_job "s2p_darcy_nopr_s${seed}" "${DARCY_SCRIPT}" "${TIME_DARCY}" \
        --R 5 --w_proj 0.0 --seed ${seed}
done

# ============================================================
# 3. VANILLA PINN (no projection — fair baseline)
#    4 PDEs × 3 seeds = 12 jobs
# ============================================================
echo ""
echo "--- Vanilla PINN (w_proj=0.0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "pinn")
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "vpinn_${pde_key}_R5_s${seed}" "${BASELINE_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 --seed ${seed}
    done
done

# ============================================================
# 4. VANILLA PINN + gPC PROJECTION (w_proj=1.0)
#    Tests whether projection helps even vanilla models
#    4 PDEs × 3 seeds = 12 jobs
# ============================================================
echo ""
echo "--- Vanilla PINN + gPC projection (w_proj=1.0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "pinn")
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "vpinn_proj_${pde_key}_R5_s${seed}" "${BASELINE_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 1.0 --seed ${seed}
    done
done

# ============================================================
# 5. PI-DeepONet (no projection — fair baseline)
#    4 PDEs × 3 seeds = 12 jobs
# ============================================================
echo ""
echo "--- PI-DeepONet (w_proj=0.0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "deeponet")
    for seed in "${NEW_SEEDS[@]}"; do
        submit_job "don_${pde_key}_R5_s${seed}" "${DEEPONET_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 --seed ${seed}
    done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================"
echo " MVP-E1 submitted: ${COUNT} total jobs (3 new seeds × 5 methods × 4 PDEs)"
echo "  S²-PINN (proj):       12 jobs"
echo "  S²-PINN (no proj):    12 jobs"
echo "  Vanilla PINN:         12 jobs"
echo "  Vanilla PINN+proj:    12 jobs"
echo "  PI-DeepONet:          12 jobs"
echo " Seeds: 456, 789, 1024 (adds to existing 42, 123)"
echo " Monitor: squeue -u \$USER"
echo "================================================"
