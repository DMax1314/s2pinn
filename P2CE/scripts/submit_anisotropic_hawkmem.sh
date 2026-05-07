#!/usr/bin/env bash
# Submit fair anisotropic PC^2 main comparison on Sol hawkmem.

set -euo pipefail

MODE="${1:-formal}" # formal | aggregate
ROOT="${ROOT:-$HOME/s2PINN}"
P2CE_ROOT="${P2CE_ROOT:-$ROOT/P2CE}"
LOGDIR="$ROOT/logs/p2ce"
RESULTDIR="${P2CE_RESULTDIR:-$P2CE_ROOT/results_anisotropic}"
mkdir -p "$LOGDIR" "$RESULTDIR"

submit_formal() {
    local solvers_csv="${P2CE_SOLVERS:-ols,lar}"
    IFS=',' read -r -a solver_probe <<< "$solvers_csv"
    local total=$((3 * 3 * 3 * ${#solver_probe[@]}))
    local last=$((total - 1))

    sbatch --parsable --array=0-${last} <<EOF
#!/bin/bash
#SBATCH --job-name=p2ce_aniso
#SBATCH --partition=hawkmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=12:00:00
#SBATCH --output=${LOGDIR}/%x_%A_%a.out
#SBATCH --error=${LOGDIR}/%x_%A_%a.err

set -eo pipefail
source ~/.bashrc
module load miniconda3/24.7.1 || true
conda activate s2PINN
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="${P2CE_ROOT}:${ROOT}:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK

PROBLEMS=(diffusion_r5 poisson_mc_r5 darcy_mc_r5)
SEEDS=(42 123 456)
DIFFUSION_PDET=(3 4 5)
POISSON_PDET=(4 6 8)
DARCY_PDET=(4 6 8)
IFS=',' read -r -a SOLVERS <<< "${solvers_csv}"

idx=\$SLURM_ARRAY_TASK_ID
n_solvers=\${#SOLVERS[@]}
solver=\${SOLVERS[\$((idx % n_solvers))]}
idx=\$((idx / n_solvers))
seed=\${SEEDS[\$((idx % \${#SEEDS[@]}))]}
idx=\$((idx / \${#SEEDS[@]}))
pdet_idx=\$((idx % 3))
idx=\$((idx / 3))
problem=\${PROBLEMS[\$idx]}

case "\$problem" in
    diffusion_r5)
        pdet=\${DIFFUSION_PDET[\$pdet_idx]}
        ;;
    poisson_mc_r5)
        pdet=\${POISSON_PDET[\$pdet_idx]}
        ;;
    darcy_mc_r5)
        pdet=\${DARCY_PDET[\$pdet_idx]}
        ;;
    *)
        echo "Unknown problem: \$problem" >&2
        exit 2
        ;;
esac

cd "${ROOT}"
echo "Running anisotropic PC^2: problem=\$problem p_z=3 p_det=\$pdet seed=\$seed solver=\$solver"
python -m p2ce.runner \
    --config "${P2CE_ROOT}/configs/\${problem}.yaml" \
    --basis_type anisotropic_product \
    --p_z 3 \
    --p_det "\$pdet" \
    --seed "\$seed" \
    --solver "\$solver" \
    --output_dir "${RESULTDIR}"
EOF
}

submit_aggregate() {
    local dependency_arg=()
    if [[ "${1:-}" != "" ]]; then
        dependency_arg=(--dependency="afterok:$1")
    fi

    sbatch --parsable "${dependency_arg[@]}" <<EOF
#!/bin/bash
#SBATCH --job-name=p2ce_aniso_agg
#SBATCH --partition=hawkmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=${LOGDIR}/%x_%j.out
#SBATCH --error=${LOGDIR}/%x_%j.err

set -eo pipefail
source ~/.bashrc
module load miniconda3/24.7.1 || true
conda activate s2PINN
set -u
export PYTHONPATH="${P2CE_ROOT}:${ROOT}:\${PYTHONPATH:-}"
cd "${ROOT}"
python "${P2CE_ROOT}/scripts/aggregate_results.py" \
    --root "${RESULTDIR}" \
    --out_dir "${RESULTDIR}/tables"
EOF
}

case "$MODE" in
    formal)
        formal_job="$(submit_formal)"
        aggregate_job="$(submit_aggregate "$formal_job")"
        echo "Submitted anisotropic PC^2 array: ${formal_job}"
        echo "Submitted anisotropic aggregate job: ${aggregate_job}"
        ;;
    aggregate)
        aggregate_job="$(submit_aggregate)"
        echo "Submitted anisotropic aggregate job: ${aggregate_job}"
        ;;
    *)
        echo "Usage: $0 [formal|aggregate]" >&2
        exit 2
        ;;
esac

echo "Logs: ${LOGDIR}"
echo "Results: ${RESULTDIR}"
