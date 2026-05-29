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

cd "$(dirname "$0")"

# Allow overrides at submission time, for example:
#   sbatch --export=ALL,DATA_DIR=/path/to/split,RESULTS_DIR=/path/to/results run.slurm
DATA_DIR="${DATA_DIR:-/Users/nreef/research/datasets/sampled_datasets/random}"
RESULTS_DIR="${RESULTS_DIR:-results}"
OPTUNA_RESULTS_DIR="${OPTUNA_RESULTS_DIR:-${RESULTS_DIR}/optimization_results}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-mlp_grav_collapse.ckpt}"

mkdir -p "${RESULTS_DIR}" "${OPTUNA_RESULTS_DIR}"

# Load necessary modules if needed.
# module load module_name
conda activate mlp_torch

python src/optimize.py \
    --data-dir "${DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}"

python src/train.py \
    --data-dir "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${OPTUNA_RESULTS_DIR}/best_params.txt" \
    --checkpoint "${CHECKPOINT_NAME}"

python src/test.py \
    --model_checkpoint "${RESULTS_DIR}/${CHECKPOINT_NAME}" \
    --test_dir "${DATA_DIR}/test.csv" \
    --output_dir "${RESULTS_DIR}/test_results" \
    --epoch_checkpoint_dir "${RESULTS_DIR}/epoch_checkpoints"
