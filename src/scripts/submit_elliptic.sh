#!/bin/bash
# ============================================================
# Submit Elliptic PDE experiments
# ============================================================

set -euo pipefail

PARTITION="hawkgpu"
TIME="01:00:00"
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
        "${BASE_SCRIPT}" --config configs/elliptic.yaml ${extra_args}
}

echo "--- Elliptic PDE (3 seeds) ---"
for seed in 42 123 456; do
    submit "s2p_elliptic_s${seed}" --seed ${seed}
done
