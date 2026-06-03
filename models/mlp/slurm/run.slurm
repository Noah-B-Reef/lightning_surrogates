#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_pipeline
#SBATCH --output=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/output/run_%j.out
#SBATCH --error=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/error/run_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
mkdir -p "${OPTUNA_RESULTS_DIR}"
prepare_logging "run"
validate_split_dir
print_config_summary
activate_conda

if [ "${OPTUNA_STORAGE:-auto}" = "auto" ]; then
    OPTUNA_STORAGE_ARG=()
else
    OPTUNA_STORAGE_ARG=(--storage "${OPTUNA_STORAGE}")
fi

python "${SCRIPT_DIR}/src/optimize.py" \
    "${DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}" \
    --study-name "${STUDY_NAME}" \
    --journal-mode "${JOURNAL_MODE}" \
    "${OPTUNA_STORAGE_ARG[@]}" \
    --num-trials "${N_TRIALS}" \
    --tune-epochs "${TUNE_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --patience "${PRUNER_PATIENCE}" \
    --min-relative-improvement "${MIN_RELATIVE_IMPROVEMENT}" \
    --num-workers "${NUM_WORKERS}"

python "${SCRIPT_DIR}/src/train.py" \
    "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${TRAIN_CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"

python "${SCRIPT_DIR}/src/test.py" \
    "${DATA_DIR}" \
    --model-checkpoint "${TEST_MODEL_CHECKPOINT}" \
    --output-dir "${TEST_OUTPUT_DIR}"
