# Shared configuration for the lightning_surrogates pipeline.
#
# Single source of truth for all path-specific settings and CLI arguments
# used by sampling, optimization, training, and testing. The SLURM scripts
# source this file, and the Python entry points read the same values through
# models/mlp/src/settings.py and samplers/path_utils.py.
#
# Override the config location with LS_CONFIG=/path/to/config.sh.

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# REPO_DIR defaults to this file's directory and RESEARCH_DIR to its parent
# (the directory holding lightning_surrogates and datasets), so the same
# config works on local machines and on TACC without edits. Export either
# variable before sourcing to override.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
RESEARCH_DIR="${RESEARCH_DIR:-$(dirname "${REPO_DIR}")}"
export REPO_DIR RESEARCH_DIR
export DATASETS_DIR="${RESEARCH_DIR}/datasets"

# Name of the source dataset being sampled (used as a path component for
# splits and results).
export DATASET_NAME="${DATASET_NAME:-grav_collapse}"
export SAMPLERS_DATASET_NAME="${DATASET_NAME}"

# Raw postprocessed chemistry HDF5 (input to the samplers). THIS is where you
# point the pipeline at a different .h5 dataset: either edit the default
# below, or override at submit time without touching this file:
#   sbatch --export=ALL,DATASET_NAME=gow17_R0.05_M6.0,SAMPLERS_RAW_H5=/path/to/file.h5 slurm/pipeline.slurm
export SAMPLERS_RAW_H5="${SAMPLERS_RAW_H5:-${DATASETS_DIR}/${DATASET_NAME}/baseline/grav_collapse_postprocessed_chemistry_uclchem.h5}"

# Sampled splits live in
# {SAMPLED_DATASETS_DIR}/{DATASET_NAME}/{SAMPLING_PROCEDURE}/{STORAGE_FORMAT}/.
export SAMPLED_DATASETS_DIR="${DATASETS_DIR}/sampled_datasets"
export SAMPLERS_DATA_DIR="${SAMPLED_DATASETS_DIR}/${DATASET_NAME}"

# Experiment results live inside each model's own directory:
# models/{architecture}/results/{DATASET_NAME}/{SAMPLING_PROCEDURE}/.
# Set RESULTS_ROOT to relocate them (defaults to the model's results/ dir).

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
export SAMPLING_PROCEDURE="${SAMPLING_PROCEDURE:-density}"  # random | density | qr_pivot | svd_fps | similarity_constrained
export STORAGE_FORMAT="${STORAGE_FORMAT:-npy}"              # csv | npy
export N_SAMPLES=6000
export MAX_SIMILARITY_TRACERS=500

# Split directory used by optimization/training/testing.
export LS_DATA_DIR="${SAMPLERS_DATA_DIR}/${SAMPLING_PROCEDURE}/${STORAGE_FORMAT}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export CONDA_ENV="mlp_torch"

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
export ACCELERATOR="gpu"
export DEVICES=1
export PRECISION=32
export CONFIG_NUM_WORKERS="auto"

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
export MODEL_NUM_LAYERS=3
export MODEL_HIDDEN_UNITS=256
export MODEL_BATCH_SIZE=1024
export MODEL_LEARNING_RATE=1e-3
export MODEL_LOSS_FUNCTION="${MODEL_LOSS_FUNCTION:-l1}"   # l1 | mse | smooth_l1
export CHECKPOINT_NAME="mlp_grav_collapse.ckpt"

# ---------------------------------------------------------------------------
# Hyperparameter optimization (sequential Optuna study)
# ---------------------------------------------------------------------------
export N_TRIALS=25
export TUNE_EPOCHS=50
export STUDY_NAME="mlp_grav_collapse_optimization"
export OPTUNA_STORAGE="auto"                 # auto = sqlite in the optimization results dir
export JOURNAL_MODE="resume"                 # resume | fresh
export PRUNER_PATIENCE=8
export MIN_RELATIVE_IMPROVEMENT=0.02

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
export TRAIN_EPOCHS=100
export EARLY_STOPPING_PATIENCE=8
export EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT=0.02

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
export TEST_NUM_TRACERS=10
