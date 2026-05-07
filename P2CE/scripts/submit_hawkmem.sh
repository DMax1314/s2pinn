#!/usr/bin/env bash
# Submit cloud-only PC^2 sanity and formal experiments on Sol hawkmem.

set -euo pipefail

MODE="${1:-all}" # sanity | formal | all
ROOT="${ROOT:-$HOME/s2PINN}"
P2CE_ROOT="${P2CE_ROOT:-$ROOT/P2CE}"
LOGDIR="$ROOT/logs/p2ce"
RESULTDIR="$P2CE_ROOT/results"
mkdir -p "$LOGDIR" "$RESULTDIR"

submit_sanity() {
    sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=p2ce_sanity
#SBATCH --partition=hawkmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=${LOGDIR}/%x_%j.out
#SBATCH --error=${LOGDIR}/%x_%j.err

set -eo pipefail
source ~/.bashrc
module load miniconda3/24.7.1 || true
conda activate s2PINN
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="${P2CE_ROOT}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK

cd "${ROOT}"
python -m pip install -q -r "${P2CE_ROOT}/requirements.txt"
python -m p2ce.sanity_uqpy --root "${P2CE_ROOT}"
EOF
}

submit_formal() {
    local dependency_arg=()
    if [[ "${1:-}" != "" ]]; then
        dependency_arg=(--dependency="afterok:$1")
    fi

    local solvers_csv="${P2CE_SOLVERS:-ols,lar}"
    local num_solvers
    IFS=',' read -r -a solver_probe <<< "$solvers_csv"
    num_solvers="${#solver_probe[@]}"
    local total=$((3 * 3 * 3 * num_solvers))
    local last=$((total - 1))

    sbatch --parsable "${dependency_arg[@]}" --array=0-${last} <<EOF
#!/bin/bash
#SBATCH --job-name=p2ce_formal
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
export PYTHONPATH="${P2CE_ROOT}:${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK

PROBLEMS=(diffusion_r5 poisson_mc_r5 darcy_mc_r5)
ORDERS=(2 3 4)
SEEDS=(42 123 456)
IFS=',' read -r -a SOLVERS <<< "${solvers_csv}"

idx=\$SLURM_ARRAY_TASK_ID
n_solvers=\${#SOLVERS[@]}
solver=\${SOLVERS[\$((idx % n_solvers))]}
idx=\$((idx / n_solvers))
seed=\${SEEDS[\$((idx % \${#SEEDS[@]}))]}
idx=\$((idx / \${#SEEDS[@]}))
order=\${ORDERS[\$((idx % \${#ORDERS[@]}))]}
idx=\$((idx / \${#ORDERS[@]}))
problem=\${PROBLEMS[\$idx]}

cd "${ROOT}"
echo "Running PC^2: problem=\$problem p=\$order seed=\$seed solver=\$solver"
python -m p2ce.runner \
    --config "${P2CE_ROOT}/configs/\${problem}.yaml" \
    --p "\$order" \
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
#SBATCH --job-name=p2ce_aggregate
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
export PYTHONPATH="${P2CE_ROOT}:${ROOT}:${PYTHONPATH:-}"
cd "${ROOT}"
python "${P2CE_ROOT}/scripts/aggregate_results.py" \
    --root "${RESULTDIR}" \
    --out_dir "${RESULTDIR}/tables"
EOF
}

case "$MODE" in
    sanity)
        sanity_job="$(submit_sanity)"
        echo "Submitted P2CE sanity job: ${sanity_job}"
        ;;
    formal)
        formal_job="$(submit_formal)"
        aggregate_job="$(submit_aggregate "$formal_job")"
        echo "Submitted P2CE formal array: ${formal_job}"
        echo "Submitted P2CE aggregate job: ${aggregate_job}"
        ;;
    all)
        sanity_job="$(submit_sanity)"
        formal_job="$(submit_formal "$sanity_job")"
        aggregate_job="$(submit_aggregate "$formal_job")"
        echo "Submitted P2CE sanity job: ${sanity_job}"
        echo "Submitted P2CE formal array after sanity: ${formal_job}"
        echo "Submitted P2CE aggregate job after formal array: ${aggregate_job}"
        ;;
    *)
        echo "Usage: $0 [sanity|formal|all]" >&2
        exit 2
        ;;
esac

echo "Logs: ${LOGDIR}"
echo "Results: ${RESULTDIR}"
