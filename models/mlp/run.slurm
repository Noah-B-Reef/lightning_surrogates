#!/bin/bash
# SBATCH -A AST24021
#SBATCH --job-name=gc_train
#SBATCH --output=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/output/run_output_%j.txt
#SBATCH --error=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/error/run_error_%j.txt
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

set -Eeuo pipefail

trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

VISTA_ROOT="${VISTA_ROOT:-/work/10252/nbr525/vista}"
SCRIPT_DIR="${SCRIPT_DIR:-${VISTA_ROOT}/lightning_surrogates/models/mlp}"
DATASETS_DIR="${DATASETS_DIR:-${VISTA_ROOT}/datasets}"
CONDA_ENV="${CONDA_ENV:-mlp_torch}"

cd "${SCRIPT_DIR}"

# Allow overrides at submission time, for example:
#   sbatch --export=ALL,DATA_DIR=/path/to/split,RESULTS_DIR=/path/to/results run.slurm
# Defaults to the best-sampler split exported by the samplers benchmark on vista.
DATA_DIR="${DATA_DIR:-${DATASETS_DIR}/sampled_datasets/best_sampler}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
OPTUNA_RESULTS_DIR="${OPTUNA_RESULTS_DIR:-${RESULTS_DIR}/optimization}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-mlp_grav_collapse.ckpt}"
N_TRIALS="${N_TRIALS:-25}"
TUNE_EPOCHS="${TUNE_EPOCHS:-50}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-0}}"

export MLP_DATASETS_DIR="${DATASETS_DIR}"
export MLP_DATA_DIR="${DATA_DIR}"
export MLP_RESULTS_DIR="${RESULTS_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/error" "${RESULTS_DIR}" "${OPTUNA_RESULTS_DIR}"

for split_file in train.csv val.csv test.csv; do
    if [ ! -f "${DATA_DIR}/${split_file}" ]; then
        echo "Missing ${DATA_DIR}/${split_file}"
        echo "DATA_DIR must point to a split directory containing train.csv, val.csv, and test.csv."
        exit 2
    fi
done

echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "OPTUNA_RESULTS_DIR=${OPTUNA_RESULTS_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v module >/dev/null 2>&1; then
    module list || true
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # Batch shells often do not source the interactive Conda hook.
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda. Load the module that provides conda or set up ${CONDA_ENV} before submitting."
    exit 3
fi

conda activate "${CONDA_ENV}"

echo "Python=$(command -v python)"
python - <<'PY'
import importlib.util
import sys

print(f"Python version: {sys.version}")
for name in ("torch", "pytorch_lightning", "optuna", "pandas", "numpy"):
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing required Python package: {name}")

import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi

python "${SCRIPT_DIR}/src/optimize.py" \
    "${DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}" \
    --num-trials "${N_TRIALS}" \
    --tune-epochs "${TUNE_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --num-workers "${NUM_WORKERS}"

python "${SCRIPT_DIR}/src/train.py" \
    "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${OPTUNA_RESULTS_DIR}/best_params.json" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --num-workers "${NUM_WORKERS}"

python "${SCRIPT_DIR}/src/test.py" \
    "${DATA_DIR}" \
    --model-checkpoint "${RESULTS_DIR}/${CHECKPOINT_NAME}" \
    --output-dir "${RESULTS_DIR}/test_results"
