import os
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
MLP_DIR = SRC_DIR.parent
LIGHTNING_SURROGATES_DIR = SRC_DIR.parents[2]
RESEARCH_DIR = SRC_DIR.parents[3]
DEFAULT_DATASETS_DIR = RESEARCH_DIR / "datasets"


def resolve_path(path):
    return Path(path).expanduser().resolve()


def env_path(name, default):
    return resolve_path(os.environ.get(name, default))


# Model/training defaults
FORECAST_HORIZON = 1
NUM_LAYERS = 3
HIDDEN_UNITS = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.0
WEIGHT_DECAY = 1e-4
EPOCHS = 100

# Data/result defaults. Override these on HPC rather than editing code.
# Every model script takes an explicit split directory; these are only fallbacks.
DEFAULT_SPLIT_DIR = env_path("MLP_DATA_DIR", DEFAULT_DATASETS_DIR / "split")
DEFAULT_RESULTS_DIR = env_path("MLP_RESULTS_DIR", MLP_DIR / "results")
NUM_WORKERS = 0
LOG_ABUNDANCES = True

# Compute defaults. Override from CLI for HPC.
ACCELERATOR = "auto"
NUM_DEVICES = 1
PRECISION = 32
STRATEGY = None
NUM_NODES = 1

# Optuna defaults
OPTUNA_N_TRIALS = 25
OPTUNA_TUNE_EPOCHS = 50
OPTUNA_STUDY_NAME = "mlp_grav_collapse_optimization"
OPTUNA_PRUNER_PATIENCE = 8
OPTUNA_SEARCH_SPACE = {
    "num_layers": {"type": "int", "low": 2, "high": 8},
    "hidden_units": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "dropout": {"type": "float", "low": 0.0, "high": 0.3},
    "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
}
