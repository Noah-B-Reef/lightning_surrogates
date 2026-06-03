#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_optuna_parallel
#SBATCH --output=optimize_parallel_%j.out
#SBATCH --error=optimize_parallel_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
mkdir -p "${OPTUNA_PARALLEL_RESULTS_DIR}"
prepare_logging "optimize_parallel"
validate_split_dir
print_config_summary
echo "PARALLEL_STUDY_NAME=${PARALLEL_STUDY_NAME}"
echo "PARALLEL_N_TRIALS=${PARALLEL_N_TRIALS}"
echo "PARALLEL_TUNE_EPOCHS=${PARALLEL_TUNE_EPOCHS}"
activate_conda

if [ -z "${PARALLEL_OPTUNA_STORAGE}" ]; then
    echo "Set optimize_parallel.storage in ${MLP_SLURM_CONFIG} to a server-backed Optuna RDB URL."
    echo "SQLite is intentionally not used for multi-node optimization."
    exit 4
fi

srun python "${SCRIPT_DIR}/src/optimize_parallel.py" \
    "${DATA_DIR}" \
    --results-dir "${OPTUNA_PARALLEL_RESULTS_DIR}" \
    --storage "${PARALLEL_OPTUNA_STORAGE}" \
    --study-name "${PARALLEL_STUDY_NAME}" \
    --journal-mode "${JOURNAL_MODE}" \
    --num-trials "${PARALLEL_N_TRIALS}" \
    --tune-epochs "${PARALLEL_TUNE_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --patience "${PRUNER_PATIENCE}" \
    --min-relative-improvement "${MIN_RELATIVE_IMPROVEMENT}" \
    --num-workers "${NUM_WORKERS}"
