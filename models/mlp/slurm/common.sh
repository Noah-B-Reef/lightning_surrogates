#!/bin/bash

set -Eeuo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLP_DIR="$(cd "${SLURM_DIR}/.." && pwd)"
MLP_SLURM_CONFIG="${MLP_SLURM_CONFIG:-${MLP_CONFIG:-${MLP_DIR}/config.sh}}"
export MLP_SLURM_CONFIG

if [ ! -f "${MLP_SLURM_CONFIG}" ]; then
    echo "Missing Slurm config: ${MLP_SLURM_CONFIG}" >&2
    exit 2
fi

source "${MLP_SLURM_CONFIG}"

OPTUNA_RESULTS_DIR="${OPTUNA_RESULTS_DIR:-${RESULTS_DIR}/optimization}"
OPTUNA_PARALLEL_RESULTS_DIR="${OPTUNA_PARALLEL_RESULTS_DIR:-${RESULTS_DIR}/optimization_parallel}"
export OPTUNA_RESULTS_DIR
export OPTUNA_PARALLEL_RESULTS_DIR

if [ "${CONFIG_NUM_WORKERS:-auto}" = "auto" ]; then
    NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-0}}"
else
    NUM_WORKERS="${NUM_WORKERS:-${CONFIG_NUM_WORKERS}}"
fi

export MLP_DATASETS_DIR="${DATASETS_DIR}"
export MLP_DATA_DIR="${DATA_DIR}"
export MLP_RESULTS_DIR="${RESULTS_DIR}"
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
    for split_file in train.csv val.csv test.csv; do
        if [ ! -f "${DATA_DIR}/${split_file}" ]; then
            echo "Missing ${DATA_DIR}/${split_file}"
            echo "DATA_DIR must point to a split directory containing train.csv, val.csv, and test.csv."
            exit 2
        fi
    done
}

print_config_summary() {
    echo "MLP_SLURM_CONFIG=${MLP_SLURM_CONFIG}"
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "DATA_DIR=${DATA_DIR}"
    echo "RESULTS_DIR=${RESULTS_DIR}"
    echo "OPTUNA_RESULTS_DIR=${OPTUNA_RESULTS_DIR}"
    echo "OPTUNA_PARALLEL_RESULTS_DIR=${OPTUNA_PARALLEL_RESULTS_DIR}"
    echo "JOURNAL_MODE=${JOURNAL_MODE}"
    echo "TRAIN_CONFIG_FILE=${TRAIN_CONFIG_FILE}"
    echo "TEST_MODEL_CHECKPOINT=${TEST_MODEL_CHECKPOINT}"
    echo "TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
    echo "CONDA_ENV=${CONDA_ENV}"
    echo "ACCELERATOR=${ACCELERATOR}"
    echo "DEVICES=${DEVICES}"
    echo "PRECISION=${PRECISION}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
    echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}
