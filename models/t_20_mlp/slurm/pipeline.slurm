#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=t20_full_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

# Full pipeline: sampling -> optimization -> training -> testing.
#
# All paths and arguments come from the repo config (config.sh). To run on a
# different raw .h5 dataset, set SAMPLERS_RAW_H5 (and DATASET_NAME for the
# split/results path component) in config.sh, or override at submit time:
#   sbatch --export=ALL,DATASET_NAME=gow17_R0.05_M6.0,SAMPLERS_RAW_H5=/path/to/file.h5 slurm/pipeline.slurm
#
# Submit from models/t_20_mlp: sbatch slurm/pipeline.slurm
# Locate common.sh whether the job was submitted from models/t_20_mlp/ or from
# models/t_20_mlp/slurm/. sbatch copies this script to the slurmd spool directory,
# so its own path (BASH_SOURCE) is only usable when run outside SLURM.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
mkdir -p logs
print_config_summary
activate_conda

echo "=== STAGE 1/4: sampling (${SAMPLING_PROCEDURE}, ${STORAGE_FORMAT}) ==="
# run_samplers.py flattens the raw H5 if needed and skips sampling when the
# split already exists.
python "${REPO_DIR}/samplers/run_samplers.py" \
    --raw-h5 "${SAMPLERS_RAW_H5}" \
    --data-dir "${SAMPLERS_DATA_DIR}" \
    --n-samples "${N_SAMPLES}" \
    --max-similarity-tracers "${MAX_SIMILARITY_TRACERS}" \
    --storage-format "${STORAGE_FORMAT}" \
    --samplers ${SAMPLING_PROCEDURE}

validate_split_dir

echo "=== STAGE 2/4: hyperparameter optimization ==="
if [ "${OPTUNA_STORAGE:-auto}" = "auto" ]; then
    OPTUNA_STORAGE_ARG=()
else
    OPTUNA_STORAGE_ARG=(--storage "${OPTUNA_STORAGE}")
fi

python "${SCRIPT_DIR}/src/optimize.py" \
    "${LS_DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}" \
    --study-name "${STUDY_NAME}" \
    --journal-mode "${JOURNAL_MODE}" \
    "${OPTUNA_STORAGE_ARG[@]}" \
    --num-trials "${N_TRIALS}" \
    --tune-epochs "${TUNE_EPOCHS}" \
    --patience "${PRUNER_PATIENCE}" \
    --min-relative-improvement "${MIN_RELATIVE_IMPROVEMENT}" \
    --no-logger \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"

echo "=== STAGE 3/4: training ==="
python "${SCRIPT_DIR}/src/train.py" \
    "${LS_DATA_DIR}" \
    --results-dir "${EXPERIMENT_DIR}" \
    --config-file "${OPTUNA_RESULTS_DIR}/best_params.json" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-relative-improvement "${EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT}" \
    --no-logger \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"

echo "=== STAGE 4/4: testing (autoregressive rollout) ==="
python "${SCRIPT_DIR}/src/test.py" \
    "${LS_DATA_DIR}" \
    --model-checkpoint "${EXPERIMENT_DIR}/${CHECKPOINT_NAME}" \
    --output-dir "${EXPERIMENT_DIR}/test_results" \
    --num-tracers "${TEST_NUM_TRACERS}"

echo "=== PIPELINE COMPLETE: results in ${EXPERIMENT_DIR} ==="
