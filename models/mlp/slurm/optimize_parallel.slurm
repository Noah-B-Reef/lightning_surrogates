#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_optuna_parallel
#SBATCH --output=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/output/optimize_parallel_%j.out
#SBATCH --error=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/error/optimize_parallel_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
        source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
    else
        source "${SLURM_SUBMIT_DIR}/common.sh"
    fi
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
mkdir -p "${OPTUNA_PARALLEL_RESULTS_DIR}"
validate_split_dir
print_config_summary
echo "PARALLEL_STUDY_NAME=${PARALLEL_STUDY_NAME}"
echo "PARALLEL_N_TRIALS=${PARALLEL_N_TRIALS}"
echo "PARALLEL_TUNE_EPOCHS=${PARALLEL_TUNE_EPOCHS}"
activate_conda

if [ -z "${PARALLEL_OPTUNA_STORAGE}" ]; then
    PARALLEL_OPTUNA_STORAGE="sqlite:///${OPTUNA_PARALLEL_RESULTS_DIR}/optuna.sqlite3"
    echo "PARALLEL_OPTUNA_STORAGE is unset; using ${PARALLEL_OPTUNA_STORAGE}"
    echo "WARNING: SQLite is not recommended for multi-node Optuna workers."
    SQLITE_STORAGE_ARG=(--allow-sqlite)
else
    SQLITE_STORAGE_ARG=()
fi

srun python "${SCRIPT_DIR}/src/optimize_parallel.py" \
    "${DATA_DIR}" \
    --results-dir "${OPTUNA_PARALLEL_RESULTS_DIR}" \
    --storage "${PARALLEL_OPTUNA_STORAGE}" \
    --study-name "${PARALLEL_STUDY_NAME}" \
    --journal-mode "${JOURNAL_MODE}" \
    --num-trials "${PARALLEL_N_TRIALS}" \
    --tune-epochs "${PARALLEL_TUNE_EPOCHS}" \
    --rollout-steps "${MODEL_ROLLOUT_STEPS}" \
    --hidden-layers "${MODEL_NUM_LAYERS}" \
    --hidden-units "${MODEL_HIDDEN_UNITS}" \
    --batch-size "${MODEL_BATCH_SIZE}" \
    --prediction-mode "${MODEL_PREDICTION_MODE}" \
    --train-sample-stride "${MODEL_TRAIN_SAMPLE_STRIDE}" \
    --val-fraction "${MODEL_VAL_FRACTION}" \
    --val-every-n-epochs "${MODEL_VAL_EVERY_N_EPOCHS}" \
    --val-rollout-steps "${MODEL_VAL_ROLLOUT_STEPS}" \
    --compact-batches \
    --init-checkpoint "${MODEL_INIT_CHECKPOINT}" \
    --no-logger \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --patience "${PRUNER_PATIENCE}" \
    --min-relative-improvement "${MIN_RELATIVE_IMPROVEMENT}" \
    --num-workers "${NUM_WORKERS}" \
    "${SQLITE_STORAGE_ARG[@]}"
