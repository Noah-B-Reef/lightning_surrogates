#!/bin/bash

set -Eeuo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "${SLURM_DIR}/.." && pwd)"
REPO_DIR="$(cd "${BENCH_DIR}/../.." && pwd)"

# Single shared config at the repository root (override with LS_CONFIG).
LS_CONFIG="${LS_CONFIG:-${REPO_DIR}/config.sh}"
export LS_CONFIG

if [ ! -f "${LS_CONFIG}" ]; then
    echo "Missing config: ${LS_CONFIG}" >&2
    exit 2
fi

source "${LS_CONFIG}"

SCRIPT_DIR="${BENCH_DIR}"

# config.sh defaults SAMPLING_PROCEDURE to density; this benchmark always uses a
# random sampling of the dataset split. Allow a submit-time override
# (sbatch --export=ALL,SAMPLING_PROCEDURE=...) but default to random, then
# recompute the split dir since LS_DATA_DIR was derived from the old procedure.
SAMPLING_PROCEDURE="${BENCH_SAMPLING_PROCEDURE:-random}"
export SAMPLING_PROCEDURE
LS_DATA_DIR="${SAMPLERS_DATA_DIR}/${SAMPLING_PROCEDURE}/${STORAGE_FORMAT}"
export LS_DATA_DIR

# Array index -> model directory. Keep this order stable: benchmark.slurm uses
# #SBATCH --array=0-3 to map SLURM_ARRAY_TASK_ID onto these entries.
MODELS=(t_1_mlp t_20_mlp lstm xlstm)

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

# resolve_model [idx]
# Maps an array index (default: SLURM_ARRAY_TASK_ID) onto a model and sets the
# per-model source dir, checkpoint name, and results dir. Exits 4 if the model
# has no implementation yet (lstm/xlstm are scaffolds until their src lands).
resolve_model() {
    local idx="${1:-${SLURM_ARRAY_TASK_ID:?resolve_model needs an index or SLURM_ARRAY_TASK_ID}}"

    if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#MODELS[@]}" ]; then
        echo "Array index ${idx} out of range (have ${#MODELS[@]} models: ${MODELS[*]})" >&2
        exit 4
    fi

    MODEL="${MODELS[$idx]}"
    MODEL_SRC_DIR="${REPO_DIR}/models/${MODEL}"
    # Per-model checkpoint name so the five runs never collide.
    MODEL_CHECKPOINT="${MODEL}_${DATASET_NAME}.ckpt"
    # Results centralized under the benchmark so cross-model comparison is easy:
    # models/autoregressive_benchmark/results/{dataset}/{sampler}/{model}/
    MODEL_EXPERIMENT_DIR="${BENCH_DIR}/results/${DATASET_NAME}/${SAMPLING_PROCEDURE}/${MODEL}"
    export MODEL MODEL_SRC_DIR MODEL_CHECKPOINT MODEL_EXPERIMENT_DIR

    if [ ! -f "${MODEL_SRC_DIR}/src/train.py" ]; then
        echo "Model '${MODEL}' is not implemented yet (missing ${MODEL_SRC_DIR}/src/train.py)."
        echo "Add models/${MODEL}/src/{train,test}.py and re-run this array task."
        exit 4
    fi
}

print_config_summary() {
    echo "LS_CONFIG=${LS_CONFIG}"
    echo "SCRIPT_DIR=${SCRIPT_DIR}"
    echo "SAMPLING_PROCEDURE=${SAMPLING_PROCEDURE}"
    echo "STORAGE_FORMAT=${STORAGE_FORMAT}"
    echo "LS_DATA_DIR=${LS_DATA_DIR}"
    echo "MODELS=${MODELS[*]}"
    echo "MODEL=${MODEL:-unset}"
    echo "MODEL_SRC_DIR=${MODEL_SRC_DIR:-unset}"
    echo "MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-unset}"
    echo "MODEL_EXPERIMENT_DIR=${MODEL_EXPERIMENT_DIR:-unset}"
    echo "CONDA_ENV=${CONDA_ENV}"
    echo "ACCELERATOR=${ACCELERATOR}"
    echo "DEVICES=${DEVICES}"
    echo "PRECISION=${PRECISION}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "TEST_NUM_TRACERS=${TEST_NUM_TRACERS}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-unset}"
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}"
    echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
}
