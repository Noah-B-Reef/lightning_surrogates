#!/bin/bash
#SBATCH --job-name=gc_train
#SBATCH --output=output/run_output_%j.txt
#SBATCH --error=error/run_error_%j.txt
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nreef@utexas.edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Allow overrides at submission time, for example:
#   sbatch --export=ALL,DATA_DIR=/path/to/split,RESULTS_DIR=/path/to/results run.slurm
# Defaults to the best-sampler split exported by the samplers benchmark on vista.
DATA_DIR="${DATA_DIR:-/work/10252/nbr525/vista/datasets/sampled_datasets/best_sampler}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
OPTUNA_RESULTS_DIR="${OPTUNA_RESULTS_DIR:-${RESULTS_DIR}/optimization}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-mlp_grav_collapse.ckpt}"
N_TRIALS="${N_TRIALS:-25}"
TUNE_EPOCHS="${TUNE_EPOCHS:-50}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-7}"

if [ -z "${DATA_DIR}" ]; then
    echo "DATA_DIR must point to a split directory containing train.csv, val.csv, and test.csv."
    exit 2
fi

mkdir -p "${RESULTS_DIR}" "${OPTUNA_RESULTS_DIR}"

# Load necessary modules if needed.
# module load module_name
conda activate mlp_torch

python src/optimize.py \
    "${DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}" \
    --num-trials "${N_TRIALS}" \
    --tune-epochs "${TUNE_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --num-workers "${NUM_WORKERS}"

python src/train.py \
    "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${OPTUNA_RESULTS_DIR}/best_params.json" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --num-workers "${NUM_WORKERS}"

python src/test.py \
    "${DATA_DIR}" \
    --model-checkpoint "${RESULTS_DIR}/${CHECKPOINT_NAME}" \
    --output-dir "${RESULTS_DIR}/test_results"
