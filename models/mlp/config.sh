# Paths
export VISTA_ROOT="/work/10252/nbr525/vista"
export SCRIPT_DIR="/work/10252/nbr525/vista/lightning_surrogates/models/mlp"
export DATASETS_DIR="/work/10252/nbr525/vista/datasets"
export DATA_DIR="/work/10252/nbr525/vista/datasets/sampled_datasets/best_sampler"
export RESULTS_DIR="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results"
export OPTUNA_RESULTS_DIR="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/optimization"
export OPTUNA_PARALLEL_RESULTS_DIR="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/optimization_parallel"
export CHECKPOINT_NAME="mlp_grav_collapse.ckpt"

# Environment
export CONDA_ENV="mlp_torch"

# Compute Settings
export ACCELERATOR="gpu"
export DEVICES=1
export PRECISION=32
export STRATEGY=""
export NUM_NODES=1
export CONFIG_NUM_WORKERS="auto"

# Model Defaults
export MODEL_NUM_LAYERS=3
export MODEL_HIDDEN_UNITS=256
export MODEL_BATCH_SIZE=32
export MODEL_LEARNING_RATE=1e-3
export MODEL_EPOCHS=100
export MODEL_LOG_ABUNDANCES="true"
export MODEL_ROLLOUT_STEPS=5

# Hyperparameter Optimization (Optimize) Settings
export N_TRIALS=25
export TUNE_EPOCHS=50
export STUDY_NAME="mlp_grav_collapse_optimization"
export OPTUNA_STORAGE="auto"
export JOURNAL_MODE="resume"
export PRUNER_PATIENCE=8
export MIN_RELATIVE_IMPROVEMENT=0.02

# Parallel Optimization Settings
export PARALLEL_N_TRIALS=25
export PARALLEL_TUNE_EPOCHS=50
export PARALLEL_STUDY_NAME="mlp_grav_collapse_optimization_parallel"
export PARALLEL_OPTUNA_STORAGE=""

# Train Settings
export TRAIN_EPOCHS=100
export TRAIN_CONFIG_FILE="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/optimization/best_params.json"

# Test Settings
export TEST_MODEL_CHECKPOINT="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/mlp_grav_collapse.ckpt"
export TEST_OUTPUT_DIR="/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/test_results"
