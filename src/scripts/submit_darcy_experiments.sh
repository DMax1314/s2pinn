#!/bin/bash
# ============================================================
# S²-PINN: Darcy Flow + Projection Scaling Experiments
# Hard benchmark with nonlinear Z-dependence (divergence form)
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_darcy_experiments.sh
# ============================================================

set -euo pipefail

PARTITION="hawkgpu"
TIME="04:00:00"      # Darcy uses 30K steps → ~90 min on T4
CPUS=4
DARCY_SCRIPT="scripts/run_darcy.sh"
BASE_SCRIPT="scripts/run_experiment.sh"

mkdir -p ../logs

submit_darcy() {
    local job_name="$1"
    shift
    local extra_args="$@"

    echo "Submitting: ${job_name}  args: ${extra_args}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${TIME}" \
        "${DARCY_SCRIPT}" ${extra_args}
}

submit_base() {
    local job_name="$1"
    shift
    local extra_args="$@"

    echo "Submitting: ${job_name}  args: ${extra_args}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${TIME}" \
        "${BASE_SCRIPT}" ${extra_args}
}

echo "================================================"
echo " S²-PINN Darcy + Projection Scaling Experiments"
echo "================================================"
echo ""

# ============================================================
# 1. DARCY R-SWEEP: R ∈ {3, 5, 10} × {proj, no_proj} × 3 seeds
#    This is the hard benchmark with nonlinear Z-dependence.
# ============================================================
echo "--- Darcy R-sweep (with projection, 3 seeds) ---"
for R in 3 5 10; do
    for seed in 42 123 456; do
        submit_darcy "darcy_R${R}_proj_s${seed}" --R ${R} --seed ${seed}
    done
done

echo ""
echo "--- Darcy R-sweep (no projection, 3 seeds) ---"
for R in 3 5 10; do
    for seed in 42 123 456; do
        submit_darcy "darcy_R${R}_nopr_s${seed}" --R ${R} --w_proj 0.0 --seed ${seed}
    done
done

# ============================================================
# 2. PROJECTION SCALING: w_proj sweep on Darcy R=5 and R=10
#    Tests whether w_proj ~ B/G normalization matters.
#    Default colloc=32768, default p=3.
#    For R=5: G = C(5+3,3) = 56
#    For R=10: G = C(10+3,3) = 286
# ============================================================
echo ""
echo "--- Projection scaling: w_proj sweep on Darcy R=5 ---"
for wp in 0.01 0.1 10.0; do
    for seed in 42 123; do
        submit_darcy "darcy_R5_wp${wp}_s${seed}" --R 5 --w_proj ${wp} --seed ${seed}
    done
done

echo ""
echo "--- Projection scaling: w_proj sweep on Darcy R=10 ---"
for wp in 0.01 0.1 10.0; do
    for seed in 42 123; do
        submit_darcy "darcy_R10_wp${wp}_s${seed}" --R 10 --w_proj ${wp} --seed ${seed}
    done
done

# ============================================================
# 3. RE-RUN DIFFUSION with UQ metrics enabled
#    Using base config (which now has uq.enabled patched in darcy config)
#    But we also need to re-run diffusion/allen_cahn/burgers with UQ
# ============================================================
echo ""
echo "--- Diffusion R=5 with UQ metrics (no extra config needed — added via code) ---"
echo "NOTE: UQ is only enabled in darcy.yaml. For diffusion/allen_cahn/burgers,"
echo "      UQ will be added in a follow-up patch to base/allen_cahn/burgers configs."

# ============================================================
# Summary
# ============================================================
echo ""
echo "================================================"
echo " Darcy experiments submitted!"
echo "  - R-sweep: 18 jobs (R={3,5,10} × {proj,nopr} × 3 seeds)"
echo "  - w_proj scaling: 12 jobs (R={5,10} × wp={0.01,0.1,10} × 2 seeds)"
echo "  - Total: 30 jobs"
echo " Monitor with: squeue -u \$USER"
echo " Results in WandB project: s2pinn"
echo "================================================"
