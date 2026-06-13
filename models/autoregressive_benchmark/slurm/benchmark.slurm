#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=arb_bench
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

# Train + test one surrogate variant per array task, in parallel. The array
# index maps onto MODELS=(t_1_mlp t_20_mlp lstm xlstm rnn) in common.sh.
# Fixed architecture, no hyperparameter optimization: train.py runs with
# --use-defaults (built-in defaults, no best_params.json, no Optuna stage).
#
# Requires the random split from sample.slurm; submit.sh gates this array on it
# with --dependency=afterok. Submit from models/autoregressive_benchmark:
#   sbatch slurm/benchmark.slurm        (or use slurm/submit.sh for chaining)
#
# Locate common.sh whether submitted from models/autoregressive_benchmark/ or
# from its slurm/ dir. sbatch copies this script to the slurmd spool dir, so its
# own path (BASH_SOURCE) is only usable when run outside SLURM.
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
validate_split_dir
resolve_model
print_config_summary
activate_conda

echo "=== [${MODEL}] STAGE 1/2: training (fixed architecture, no HPO) ==="
python "${MODEL_SRC_DIR}/src/train.py" \
    "${LS_DATA_DIR}" \
    --results-dir "${MODEL_EXPERIMENT_DIR}" \
    --use-defaults \
    --checkpoint "${MODEL_CHECKPOINT}" \
    --epochs "${TRAIN_EPOCHS}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    --early-stopping-min-relative-improvement "${EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT}" \
    --no-logger \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"

echo "=== [${MODEL}] STAGE 2/2: testing (autoregressive rollout) ==="
python "${MODEL_SRC_DIR}/src/test.py" \
    "${LS_DATA_DIR}" \
    --model-checkpoint "${MODEL_EXPERIMENT_DIR}/${MODEL_CHECKPOINT}" \
    --output-dir "${MODEL_EXPERIMENT_DIR}/test_results" \
    --num-tracers "${TEST_NUM_TRACERS}"

echo "=== [${MODEL}] COMPLETE: results in ${MODEL_EXPERIMENT_DIR} ==="
