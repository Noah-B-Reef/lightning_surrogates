import os
from pathlib import Path

# Identify directories
SRC_DIR = Path(__file__).resolve().parent
MLP_DIR = SRC_DIR.parent
LIGHTNING_SURROGATES_DIR = SRC_DIR.parents[2]
RESEARCH_DIR = SRC_DIR.parents[3]

DEFAULT_CONFIG_FILE = MLP_DIR / "config.sh"
CONFIG_FILE = Path(
    os.environ.get("MLP_CONFIG")
    or os.environ.get("MLP_SLURM_CONFIG")
    or DEFAULT_CONFIG_FILE
).expanduser().resolve()


def load_env_file(path):
    """Load variables from a config.sh shell file into os.environ if not already set."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip quotes if present
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_env_file(CONFIG_FILE)


# Helper functions to retrieve typed environment variables
def env_str(key, default):
    return os.environ.get(key, str(default))


def env_path(key, default):
    val = os.environ.get(key)
    if val is None:
        return Path(default).expanduser().resolve()
    return Path(val).expanduser().resolve()


def env_int(key, default):
    return int(os.environ.get(key, default))


def env_float(key, default):
    return float(os.environ.get(key, default))


def env_bool(key, default):
    val = os.environ.get(key)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def env_num_workers(key="CONFIG_NUM_WORKERS", default=0):
    value = os.environ.get(key, str(default))
    if str(value).lower() == "auto":
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    return int(value)


def resolve_path(path):
    return Path(path).expanduser().resolve()


# Default Paths (resolved relative to workspace if not explicitly defined in environment)
DEFAULT_DATASETS_DIR = env_path("DATASETS_DIR", RESEARCH_DIR / "datasets")
DEFAULT_SPLIT_DIR = env_path(
    "DATA_DIR",
    DEFAULT_DATASETS_DIR / "sampled_datasets" / "best_sampler",
)
DEFAULT_RESULTS_DIR = env_path("RESULTS_DIR", MLP_DIR / "results")
DEFAULT_OPTUNA_RESULTS_DIR = env_path(
    "OPTUNA_RESULTS_DIR",
    DEFAULT_RESULTS_DIR / "optimization",
)
DEFAULT_OPTUNA_PARALLEL_RESULTS_DIR = env_path(
    "OPTUNA_PARALLEL_RESULTS_DIR",
    DEFAULT_RESULTS_DIR / "optimization_parallel",
)
CHECKPOINT_NAME = env_str("CHECKPOINT_NAME", "mlp_grav_collapse.ckpt")
TRAIN_CONFIG_FILE = env_path(
    "TRAIN_CONFIG_FILE",
    DEFAULT_OPTUNA_RESULTS_DIR / "best_params.json",
)
TEST_MODEL_CHECKPOINT = env_path(
    "TEST_MODEL_CHECKPOINT",
    DEFAULT_RESULTS_DIR / CHECKPOINT_NAME,
)
TEST_OUTPUT_DIR = env_path("TEST_OUTPUT_DIR", DEFAULT_RESULTS_DIR / "test_results")

SPLIT_FILES = ("train.csv", "val.csv", "test.csv")


def resolve_split_dir(dataset_path=None, required=SPLIT_FILES):
    """Resolve a split directory, defaulting to the configured split path."""
    using_default = dataset_path is None
    path = DEFAULT_SPLIT_DIR if using_default else resolve_path(dataset_path)
    if path.is_file():
        path = path.parent
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        hint = ""
        if using_default:
            hint = (
                " No split path was given, so the split configured in config.sh was "
                "used. Set DATA_DIR, set MLP_CONFIG/MLP_SLURM_CONFIG to another "
                "config file, or pass an explicit split directory."
            )
        raise FileNotFoundError(f"Missing {missing} in split directory: {path}.{hint}")
    return path


# Model defaults
NUM_LAYERS = env_int("MODEL_NUM_LAYERS", 3)
HIDDEN_UNITS = env_int("MODEL_HIDDEN_UNITS", 256)
BATCH_SIZE = env_int("MODEL_BATCH_SIZE", 32)
LEARNING_RATE = env_float("MODEL_LEARNING_RATE", 1e-3)
EPOCHS = env_int("TRAIN_EPOCHS", env_int("MODEL_EPOCHS", 100))
NUM_WORKERS = env_num_workers()
LOG_ABUNDANCES = env_bool("MODEL_LOG_ABUNDANCES", True)

# Compute parameters
ACCELERATOR = env_str("ACCELERATOR", "auto")
NUM_DEVICES = env_str("DEVICES", "1")  # Keep as string since devices can be list/str
PRECISION = env_str("PRECISION", "32")  # Keep as string/int
STRATEGY = env_str("STRATEGY", "") or None
NUM_NODES = env_int("NUM_NODES", 1)

# Optimization parameters (single-node)
OPTUNA_N_TRIALS = env_int("N_TRIALS", 25)
OPTUNA_TUNE_EPOCHS = env_int("TUNE_EPOCHS", 50)
OPTUNA_STUDY_NAME = env_str("STUDY_NAME", "mlp_grav_collapse_optimization")
OPTUNA_STORAGE = env_str("OPTUNA_STORAGE", "auto")
OPTUNA_JOURNAL_MODE = env_str("JOURNAL_MODE", "resume")
OPTUNA_PRUNER_PATIENCE = env_int("PRUNER_PATIENCE", 8)
OPTUNA_MIN_RELATIVE_IMPROVEMENT = env_float("MIN_RELATIVE_IMPROVEMENT", 0.02)

# Optimization parameters (parallel)
OPTUNA_PARALLEL_N_TRIALS = env_int("PARALLEL_N_TRIALS", OPTUNA_N_TRIALS)
OPTUNA_PARALLEL_TUNE_EPOCHS = env_int("PARALLEL_TUNE_EPOCHS", OPTUNA_TUNE_EPOCHS)
OPTUNA_PARALLEL_STUDY_NAME = env_str("PARALLEL_STUDY_NAME", f"{OPTUNA_STUDY_NAME}_parallel")
OPTUNA_PARALLEL_STORAGE = env_str("PARALLEL_OPTUNA_STORAGE", "")

# Search space for hyperparameter tuning (can remain hardcoded here)
OPTUNA_SEARCH_SPACE = {
    "num_layers": {"type": "int", "low": 2, "high": 8},
    "hidden_units": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
}
