#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR="${DATA_DIR:-${1:-}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
OPTUNA_RESULTS_DIR="${OPTUNA_RESULTS_DIR:-${RESULTS_DIR}/optimization}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-mlp_grav_collapse.ckpt}"
N_TRIALS="${N_TRIALS:-25}"
TUNE_EPOCHS="${TUNE_EPOCHS:-50}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
ACCELERATOR="${ACCELERATOR:-auto}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SKIP_OPTIMIZE="${SKIP_OPTIMIZE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

# DATA_DIR is optional: when unset, the Python scripts fall back to the
# best-sampler split (config.DEFAULT_SPLIT_DIR / MLP_DATA_DIR).
DATA_ARG=()
if [ -n "$DATA_DIR" ]; then
    DATA_ARG=("$DATA_DIR")
fi

mkdir -p "$RESULTS_DIR" "$OPTUNA_RESULTS_DIR"

step_start() {
    echo ""
    echo "=== Step $1: $2 ==="
    STEP_START_TIME=$(date +%s)
}

step_end() {
    local elapsed=$(( $(date +%s) - STEP_START_TIME ))
    echo "=== Step $1 complete (${elapsed}s) ==="
}

if [ "$SKIP_OPTIMIZE" != "1" ]; then
    step_start 1 "Optimize MLP hyperparameters"
    python "$SCRIPT_DIR/src/optimize.py" "${DATA_ARG[@]}"         --results-dir "$OPTUNA_RESULTS_DIR"         --num-trials "$N_TRIALS"         --tune-epochs "$TUNE_EPOCHS"         --accelerator "$ACCELERATOR"         --devices "$DEVICES"         --num-workers "$NUM_WORKERS"
    step_end 1
else
    echo "--- Skipping Step 1: Optimize ---"
fi

if [ "$SKIP_TRAIN" != "1" ]; then
    step_start 2 "Train final MLP"
    python "$SCRIPT_DIR/src/train.py" "${DATA_ARG[@]}"         --results-dir "$RESULTS_DIR"         --config-file "$OPTUNA_RESULTS_DIR/best_params.json"         --checkpoint "$CHECKPOINT_NAME"         --epochs "$TRAIN_EPOCHS"         --accelerator "$ACCELERATOR"         --devices "$DEVICES"         --num-workers "$NUM_WORKERS"
    step_end 2
else
    echo "--- Skipping Step 2: Train ---"
fi

if [ "$SKIP_TEST" != "1" ]; then
    step_start 3 "Autoregressive rollout test"
    python "$SCRIPT_DIR/src/test.py" "${DATA_ARG[@]}"         --model-checkpoint "$RESULTS_DIR/$CHECKPOINT_NAME"         --output-dir "$RESULTS_DIR/test_results"
    step_end 3
else
    echo "--- Skipping Step 3: Test ---"
fi

echo ""
echo "=== MLP benchmark complete ==="
