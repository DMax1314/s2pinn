#!/bin/bash
# ============================================================
# S²-PINN: Master Experiment Launcher
# Submits ALL experiments to SLURM for the ICML 2026 paper.
#
# Usage:
#   cd ~/s2PINN/src
#   bash scripts/submit_all_experiments.sh
#
# This script submits experiments to hawkgpu (T4 GPUs).
# Each run: ~50 min for 20K steps => ~6h for 7 runs.
# Total: ~35 jobs => spread across queue.
# ============================================================

set -euo pipefail

PARTITION="hawkgpu"
TIME="02:00:00"
CPUS=4
BASE_SCRIPT="scripts/run_experiment.sh"

mkdir -p ../logs

submit() {
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
echo " S²-PINN Experiment Submission"
echo "================================================"
echo ""

# ============================================================
# 1. DEFAULT CONFIGURATION — 3 seeds for error bars
# ============================================================
echo "--- Default config (3 seeds) ---"
for seed in 42 123 456; do
    submit "s2p_default_s${seed}" --seed ${seed}
done

# ============================================================
# 2. ABLATION: Remove projection loss
# ============================================================
echo ""
echo "--- Ablation: no projection loss ---"
for seed in 42 123 456; do
    submit "s2p_no_proj_s${seed}" --w_proj 0.0 --seed ${seed}
done

# ============================================================
# 3. ABLATION: Remove Gram penalty (spatial feature orthogonality)
# ============================================================
echo ""
echo "--- Ablation: no Gram penalty ---"
for seed in 42 123 456; do
    submit "s2p_no_gram_s${seed}" --w_orth 0.0 --seed ${seed}
done

# ============================================================
# 4. ABLATION: Remove CP factor orthogonality
# ============================================================
echo ""
echo "--- Ablation: no factor orthogonality ---"
for seed in 42 123 456; do
    submit "s2p_no_fac_s${seed}" --w_fac_orth 0.0 --seed ${seed}
done

# ============================================================
# 5. ABLATION: Remove ALL regularization
# ============================================================
echo ""
echo "--- Ablation: no regularization ---"
for seed in 42 123 456; do
    submit "s2p_no_reg_s${seed}" --w_orth 0.0 --w_fac_orth 0.0 --seed ${seed}
done

# ============================================================
# 6. SWEEP: CP rank  q ∈ {4, 8, 16, 32}
# ============================================================
echo ""
echo "--- Sweep: CP rank ---"
for rank in 4 8 32; do
    for seed in 42 123 456; do
        submit "s2p_rank${rank}_s${seed}" --rank ${rank} --seed ${seed}
    done
done
# rank=16 is the default, covered by section 1

# ============================================================
# 7. SWEEP: gPC order  p ∈ {1, 2, 3, 4}
# ============================================================
echo ""
echo "--- Sweep: gPC order ---"
for p in 1 2 4; do
    for seed in 42 123 456; do
        submit "s2p_p${p}_s${seed}" --p ${p} --seed ${seed}
    done
done
# p=3 is the default, covered by section 1

# ============================================================
# 8. SWEEP: Spatial atoms  M ∈ {64, 128, 256, 512}
# ============================================================
echo ""
echo "--- Sweep: spatial atoms ---"
for M in 64 128 512; do
    for seed in 42 123 456; do
        submit "s2p_M${M}_s${seed}" --M ${M} --seed ${seed}
    done
done
# M=256 is the default, covered by section 1

# ============================================================
# 9. SWEEP: Stochastic dimension  R ∈ {1, 2, 3, 5}
# ============================================================
echo ""
echo "--- Sweep: stochastic dimension ---"
for R in 1 2 5; do
    for seed in 42 123 456; do
        submit "s2p_R${R}_s${seed}" --R ${R} --seed ${seed}
    done
done
# R=3 is the default, covered by section 1

# ============================================================
# Summary
# ============================================================
echo ""
echo "--- Baselines: Vanilla PINN (MLP) ---"
BASELINE_SCRIPT="scripts/run_baseline.sh"

submit_baseline() {
    local job_name="$1"
    shift
    local extra_args="$@"

    echo "Submitting baseline: ${job_name}  args: ${extra_args}"
    sbatch \
        --job-name="${job_name}" \
        --partition="${PARTITION}" \
        --output="../logs/%x_%j.out" \
        --error="../logs/%x_%j.err" \
        --nodes=1 --ntasks=1 \
        --cpus-per-task=${CPUS} \
        --gres=gpu:1 \
        --time="${TIME}" \
        "${BASELINE_SCRIPT}" ${extra_args}
}

for hidden in 128 256; do
    for layers in 4 6; do
        for seed in 42 123 456; do
            submit_baseline "bl_h${hidden}_L${layers}_s${seed}" \
                --hidden_dim ${hidden} --n_layers ${layers} --seed ${seed}
        done
    done
done

echo ""
echo "================================================"
echo " All experiments submitted!"
echo " Monitor with: squeue -u \$USER"
echo " Results will appear in WandB project: s2pinn"
echo "================================================"
