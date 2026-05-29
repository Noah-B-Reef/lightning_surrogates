from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
MLP_DIR = SRC_DIR.parent
RESEARCH_DIR = SRC_DIR.parents[4]
DEFAULT_DATASETS_DIR = RESEARCH_DIR / "datasets"

# Model/training defaults
FORECAST_HORIZON = 1
NUM_LAYERS = 3
HIDDEN_UNITS = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100

# Data defaults
SAMPLED_DATASETS_DIR = DEFAULT_DATASETS_DIR / "sampled_datasets"
DEFAULT_SPLIT_DIR = SAMPLED_DATASETS_DIR / "random"
NUM_WORKERS = 0
LOG_ABUNDANCES = True

# Compute defaults. Override from CLI for TACC.
ACCELERATOR = "auto"
NUM_DEVICES = 1
PRECISION = 32
STRATEGY = None
NUM_NODES = 1

# Optuna defaults
OPTUNA_N_TRIALS = 25
OPTUNA_TUNE_EPOCHS = 50
OPTUNA_STUDY_NAME = "mlp_grav_collapse_optimization"
OPTUNA_STORAGE = "sqlite:///optuna.sqlite3"
OPTUNA_PRUNER_PATIENCE = 8
OPTUNA_SEARCH_SPACE = {
    "num_layers": {"type": "int", "low": 2, "high": 4},
    "hidden_units": {"type": "int", "low": 512, "high": 2048, "step": 512},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
}
