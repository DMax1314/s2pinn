#!/bin/bash
# ============================================================
# S²-PINN Baseline Comparisons: Vanilla PINN + PI-DeepONet + SC
# All 4 PDEs × R=5 × {proj, no_proj} × 2 seeds
#
# Spreads jobs across hawkgpu (T4), haswell/1080, haswell/2080Ti
# via round-robin for maximum overnight throughput.
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_baselines.sh
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
TIME_SC="06:00:00"       # SC (q=2, 32 nodes × 20K each)
TIME_SC_DARCY="08:00:00" # SC Darcy (q=2, 32 nodes × 30K each)
CPUS=4

# Script paths
BASELINE_SCRIPT="scripts/run_baseline.sh"
DEEPONET_SCRIPT="scripts/run_deeponet.sh"
SC_SCRIPT="scripts/run_sc.sh"

# PDE configs + labels
declare -A PDE_CONFIGS
PDE_CONFIGS[diff]="configs/base.yaml"
PDE_CONFIGS[ac]="configs/allen_cahn.yaml"
PDE_CONFIGS[burg]="configs/burgers.yaml"
PDE_CONFIGS[darcy]="configs/darcy.yaml"

SEEDS=(42 123)

mkdir -p ../logs

COUNT=0

get_time() {
    local pde="$1"
    local method="$2"
    if [ "${pde}" = "darcy" ]; then
        if [ "${method}" = "sc" ]; then
            echo "${TIME_SC_DARCY}"
        else
            echo "${TIME_DARCY}"
        fi
    else
        if [ "${method}" = "sc" ]; then
            echo "${TIME_SC}"
        else
            echo "${TIME_SHORT}"
        fi
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
echo " S²-PINN Baseline Experiments"
echo " GPU pool: hawkgpu(T4), haswell(1080), haswell(2080Ti)"
echo "================================================"

# ============================================================
# 1. VANILLA PINN (no projection — fair baseline)
#    4 PDEs × R=5 × 2 seeds = 8 jobs
# ============================================================
echo ""
echo "--- Vanilla PINN (w_proj=0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "pinn")
    for seed in "${SEEDS[@]}"; do
        submit_job "vpinn_${pde_key}_R5_s${seed}" "${BASELINE_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 --seed ${seed}
    done
done

# ============================================================
# 2. PI-DeepONet (no projection — fair baseline)
#    4 PDEs × R=5 × 2 seeds = 8 jobs
# ============================================================
echo ""
echo "--- PI-DeepONet (w_proj=0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "deeponet")
    for seed in "${SEEDS[@]}"; do
        submit_job "don_${pde_key}_R5_s${seed}" "${DEEPONET_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 0.0 --seed ${seed}
    done
done

# ============================================================
# 3. STOCHASTIC COLLOCATION (q=2 for tractability)
#    R=5, q=2 → 2^5=32 quadrature nodes per job
#    4 PDEs × 2 seeds = 8 jobs
# ============================================================
echo ""
echo "--- Stochastic Collocation (q=2, 32 nodes, reduced steps) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    if [ "${pde_key}" = "darcy" ]; then
        sc_steps=8000
        sc_time="16:00:00"
    else
        sc_steps=5000
        sc_time="12:00:00"
    fi
    for seed in "${SEEDS[@]}"; do
        submit_job "sc_${pde_key}_R5_q2_s${seed}" "${SC_SCRIPT}" "${sc_time}" \
            --config "${config}" --R 5 --q 2 --steps ${sc_steps} --seed ${seed}
    done
done

# ============================================================
# 4. Vanilla PINN + gPC projection
#    Tests whether projection helps even vanilla models
#    (strengthens the paper story if yes)
#    4 PDEs × R=5 × 2 seeds = 8 jobs
# ============================================================
echo ""
echo "--- Vanilla PINN + gPC projection (w_proj=1.0) ---"
for pde_key in diff ac burg darcy; do
    config="${PDE_CONFIGS[${pde_key}]}"
    time_limit=$(get_time "${pde_key}" "pinn")
    for seed in "${SEEDS[@]}"; do
        submit_job "vpinn_proj_${pde_key}_R5_s${seed}" "${BASELINE_SCRIPT}" "${time_limit}" \
            --config "${config}" --R 5 --w_proj 1.0 --seed ${seed}
    done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================"
echo " Baselines submitted: ${COUNT} total jobs"
echo "  Vanilla PINN (no proj):  8 jobs"
echo "  PI-DeepONet (no proj):   8 jobs"
echo "  SC (q=2, 32 nodes):      8 jobs"
echo "  Vanilla PINN (w/ proj):  8 jobs"
echo " Monitor: squeue -u \$USER"
echo "================================================"
