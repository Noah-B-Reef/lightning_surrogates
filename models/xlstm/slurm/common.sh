#!/bin/bash

set -Eeuo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XLSTM_DIR="$(cd "${SLURM_DIR}/.." && pwd)"
REPO_DIR="$(cd "${XLSTM_DIR}/../.." && pwd)"

# Single shared config at the repository root (override with LS_CONFIG).
LS_CONFIG="${LS_CONFIG:-${REPO_DIR}/config.sh}"
export LS_CONFIG

if [ ! -f "${LS_CONFIG}" ]; then
    echo "Missing config: ${LS_CONFIG}" >&2
    exit 2
fi

source "${LS_CONFIG}"

SCRIPT_DIR="${XLSTM_DIR}"

# config.sh defaults the checkpoint/study names to the MLP architecture; the
# xLSTM owns its own naming so its results never collide with the MLP's.
CHECKPOINT_NAME="xlstm_${DATASET_NAME}.ckpt"
STUDY_NAME="xlstm_${DATASET_NAME}_optimization"
export CHECKPOINT_NAME STUDY_NAME

# Experiment results live inside the model directory:
# models/xlstm/results/{dataset name}/{sampler}
EXPERIMENT_DIR="${RESULTS_ROOT:-${XLSTM_DIR}/results}/${DATASET_NAME}/${SAMPLING_PROCEDURE}"
OPTUNA_RESULTS_DIR="${EXPERIMENT_DIR}/optimization"
export EXPERIMENT_DIR
export OPTUNA_RESULTS_DIR

if [ "${CONFIG_NUM_WORKERS:-auto}" = "auto" ]; then
    NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-0}}"
else
    NUM_WORKERS="${NUM_WORKERS:-${CONFIG_NUM_WORKERS}}"
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

activate_conda() {
    if command -v module >/dev/null 2>&1; then
        module list || true
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Could not find conda. Load the module that provides conda or set up ${CONDA_ENV} before submitting."
        exit 3
    fi

    conda activate "${CONDA_ENV}"
}

validate_split_dir() {
    for split_name in train val test; do
        if [ ! -f "${LS_DATA_DIR}/${split_name}.${STORAGE_FORMAT}" ]; then
            echo "Missing ${LS_DATA_DIR}/${split_name}.${STORAGE_FORMAT}"
            echo "LS_DATA_DIR must point to a split directory containing train/val/test .${STORAGE_FORMAT} files."
            exit 2
        fi
    done
}

print_config_summary() {
    echo "LS_CONFIG=${LS_CONFIG}"
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "SAMPLING_PROCEDURE=${SAMPLING_PROCEDURE}"
    echo "STORAGE_FORMAT=${STORAGE_FORMAT}"
    echo "LS_DATA_DIR=${LS_DATA_DIR}"
    echo "RESULTS_ROOT=${RESULTS_ROOT:-${XLSTM_DIR}/results}"
    echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
    echo "OPTUNA_RESULTS_DIR=${OPTUNA_RESULTS_DIR}"
    echo "CHECKPOINT_NAME=${CHECKPOINT_NAME}"
    echo "STUDY_NAME=${STUDY_NAME}"
    echo "JOURNAL_MODE=${JOURNAL_MODE}"
    echo "CONDA_ENV=${CONDA_ENV}"
    echo "ACCELERATOR=${ACCELERATOR}"
    echo "DEVICES=${DEVICES}"
    echo "PRECISION=${PRECISION}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "MODEL_XLSTM_CELL_TYPE=${MODEL_XLSTM_CELL_TYPE:-slstm}"
    echo "MODEL_XLSTM_NUM_BLOCKS=${MODEL_XLSTM_NUM_BLOCKS:-2}"
    echo "MODEL_XLSTM_HIDDEN_DIM=${MODEL_XLSTM_HIDDEN_DIM:-256}"
    echo "MODEL_XLSTM_NUM_HEADS=${MODEL_XLSTM_NUM_HEADS:-4}"
    echo "MODEL_BATCH_SIZE=${MODEL_BATCH_SIZE}"
    echo "MODEL_LEARNING_RATE=${MODEL_LEARNING_RATE}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}
