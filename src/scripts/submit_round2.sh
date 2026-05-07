#!/bin/bash
# ============================================================
# S²-PINN Round 2: γ-sweep, p-sweep, UQ for all PDEs, R=10 rerun
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_round2.sh
# ============================================================

set -euo pipefail

PARTITION="hawkgpu"
TIME_SHORT="04:00:00"   # diffusion/AC/burgers (20K steps)
TIME_DARCY="04:00:00"   # Darcy R≤5 (30K steps)
TIME_R10="08:00:00"     # Darcy R=10 (30K steps, needs longer)
CPUS=4
DARCY_SCRIPT="scripts/run_darcy.sh"
BASE_SCRIPT="scripts/run_experiment.sh"

mkdir -p ../logs

submit_darcy() {
    local job_name="$1"
    local time="$2"
    shift 2
    local extra_args="$@"

    echo "  [darcy] ${job_name}  args: ${extra_args}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${time}" \
        "${DARCY_SCRIPT}" ${extra_args}
}

submit_pde() {
    local job_name="$1"
    local config="$2"
    shift 2
    local extra_args="$@"

    echo "  [pde] ${job_name}  config: ${config}  args: ${extra_args}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${TIME_SHORT}" \
        "${BASE_SCRIPT}" --config "${config}" ${extra_args}
}

COUNT=0

echo "================================================"
echo " S²-PINN Round 2 Experiments"
echo "================================================"

# ============================================================
# 1. γ-SWEEP ON DARCY: γ ∈ {0.05, 0.1, 0.2, 0.5}
#    R=5, p=3, {proj, no_proj} × 2 seeds
#    (γ=0.3 already done in Round 1)
# ============================================================
echo ""
echo "--- Batch 1: γ-sweep on Darcy (R=5) ---"
for gamma in 0.05 0.1 0.2 0.5; do
    for seed in 42 123; do
        submit_darcy "darcy_g${gamma}_proj_s${seed}" "${TIME_DARCY}" \
            --R 5 --gamma ${gamma} --seed ${seed}
        COUNT=$((COUNT+1))

        submit_darcy "darcy_g${gamma}_nopr_s${seed}" "${TIME_DARCY}" \
            --R 5 --gamma ${gamma} --w_proj 0.0 --seed ${seed}
        COUNT=$((COUNT+1))
    done
done

# ============================================================
# 2. p-SWEEP ON DARCY: p ∈ {5, 7}
#    R=5, γ=0.3, {proj, no_proj} × 2 seeds
#    (p=3 already done in Round 1)
# ============================================================
echo ""
echo "--- Batch 2: p-sweep on Darcy (R=5, γ=0.3) ---"
for p in 5 7; do
    for seed in 42 123; do
        submit_darcy "darcy_p${p}_proj_s${seed}" "${TIME_DARCY}" \
            --R 5 --p ${p} --seed ${seed}
        COUNT=$((COUNT+1))

        submit_darcy "darcy_p${p}_nopr_s${seed}" "${TIME_DARCY}" \
            --R 5 --p ${p} --w_proj 0.0 --seed ${seed}
        COUNT=$((COUNT+1))
    done
done

# ============================================================
# 3. UQ FOR ALL PDEs: diffusion, allen_cahn, burgers
#    R=5, {proj, no_proj} × 3 seeds
#    UQ now enabled in all configs
# ============================================================
echo ""
echo "--- Batch 3: UQ metrics for diffusion/Allen-Cahn/Burgers (R=5) ---"

# Diffusion (base.yaml, pde.type defaults to "diffusion")
for seed in 42 123 456; do
    submit_pde "diff_R5_proj_uq_s${seed}" "configs/base.yaml" \
        --R 5 --seed ${seed}
    COUNT=$((COUNT+1))

    submit_pde "diff_R5_nopr_uq_s${seed}" "configs/base.yaml" \
        --R 5 --w_proj 0.0 --seed ${seed}
    COUNT=$((COUNT+1))
done

# Allen-Cahn
for seed in 42 123 456; do
    submit_pde "ac_R5_proj_uq_s${seed}" "configs/allen_cahn.yaml" \
        --R 5 --seed ${seed}
    COUNT=$((COUNT+1))

    submit_pde "ac_R5_nopr_uq_s${seed}" "configs/allen_cahn.yaml" \
        --R 5 --w_proj 0.0 --seed ${seed}
    COUNT=$((COUNT+1))
done

# Burgers
for seed in 42 123 456; do
    submit_pde "burg_R5_proj_uq_s${seed}" "configs/burgers.yaml" \
        --R 5 --seed ${seed}
    COUNT=$((COUNT+1))

    submit_pde "burg_R5_nopr_uq_s${seed}" "configs/burgers.yaml" \
        --R 5 --w_proj 0.0 --seed ${seed}
    COUNT=$((COUNT+1))
done

# ============================================================
# 4. RERUN R=10 DARCY with 8h wall-clock
#    {proj, no_proj} × 3 seeds
# ============================================================
echo ""
echo "--- Batch 4: Darcy R=10 rerun (8h wall-clock) ---"
for seed in 42 123 456; do
    submit_darcy "darcy_R10_proj_v2_s${seed}" "${TIME_R10}" \
        --R 10 --seed ${seed}
    COUNT=$((COUNT+1))

    submit_darcy "darcy_R10_nopr_v2_s${seed}" "${TIME_R10}" \
        --R 10 --w_proj 0.0 --seed ${seed}
    COUNT=$((COUNT+1))
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================"
echo " Round 2 submitted: ${COUNT} total jobs"
echo "  Batch 1 (γ-sweep):  16 jobs"
echo "  Batch 2 (p-sweep):   8 jobs"
echo "  Batch 3 (UQ PDEs):  18 jobs"
echo "  Batch 4 (R=10 v2):   6 jobs"
echo " Monitor: squeue -u \$USER"
echo "================================================"
