import os
from pathlib import Path

# Identify directories
SRC_DIR = Path(__file__).resolve().parent
MLP_DIR = SRC_DIR.parent
LIGHTNING_SURROGATES_DIR = SRC_DIR.parents[2]
RESEARCH_DIR = SRC_DIR.parents[3]

# Single shared config file at the repository root. It is a shell-sourceable
# key=value file so the SLURM scripts and the Python entry points read the
# exact same values. Override with LS_CONFIG.
DEFAULT_CONFIG_FILE = LIGHTNING_SURROGATES_DIR / "config.sh"
CONFIG_FILE = Path(
    os.environ.get("LS_CONFIG") or DEFAULT_CONFIG_FILE
).expanduser().resolve()


def load_env_file(path):
    """Load variables from a config.sh shell file into os.environ if not already set.

    Supports ``${VAR}`` references to earlier variables. Lines using command
    substitution (``$(...)``) are skipped; the anchors they would compute
    (REPO_DIR, RESEARCH_DIR) are pre-seeded from this file's location instead.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return
    os.environ.setdefault("REPO_DIR", str(LIGHTNING_SURROGATES_DIR))
    os.environ.setdefault("RESEARCH_DIR", str(RESEARCH_DIR))
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if "$(" in value:
            continue
        # Strip quotes / trailing inline comments
        if value[:1] in {"'", '"'}:
            closing = value.find(value[0], 1)
            if closing != -1:
                value = value[1:closing]
        else:
            value = value.split("#", 1)[0].strip()
        os.environ.setdefault(key, os.path.expandvars(value))


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


def env_num_workers(key="CONFIG_NUM_WORKERS", default=0):
    value = os.environ.get(key, str(default))
    if str(value).lower() == "auto":
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    return int(value)


def resolve_path(path):
    return Path(path).expanduser().resolve()


# Default paths (resolved relative to the workspace unless set in the config)
DEFAULT_DATASETS_DIR = env_path("DATASETS_DIR", RESEARCH_DIR / "datasets")
DATASET_NAME = env_str("DATASET_NAME", "grav_collapse")
DEFAULT_SAMPLED_DATASETS_DIR = env_path(
    "SAMPLED_DATASETS_DIR",
    DEFAULT_DATASETS_DIR / "sampled_datasets",
)
DEFAULT_SPLIT_DIR = env_path(
    "LS_DATA_DIR",
    DEFAULT_SAMPLED_DATASETS_DIR
    / DATASET_NAME
    / env_str("SAMPLING_PROCEDURE", "density")
    / env_str("STORAGE_FORMAT", "npy"),
)
# Experiment results live in results/{dataset name}/{sampler}/{architecture}.
DEFAULT_RESULTS_ROOT = env_path(
    "RESULTS_ROOT", LIGHTNING_SURROGATES_DIR / "results"
)
MODEL_ARCHITECTURE = "mlp"
CHECKPOINT_NAME = env_str("CHECKPOINT_NAME", "mlp_grav_collapse.ckpt")


def experiment_relpath(split_dir):
    """Derive {dataset name}/{sampler} from a split directory.

    sampled_datasets/grav_collapse/density/npy -> grav_collapse/density.
    Falls back to the sampler directory name alone when the layout does not
    include a dataset-name level (e.g. an ad-hoc flat split directory).
    """
    split_dir = resolve_path(split_dir)
    sampler_dir = split_dir.parent if split_dir.name in ("csv", "npy") else split_dir
    dataset_dir = sampler_dir.parent
    if dataset_dir == sampler_dir or dataset_dir.name in ("", "sampled_datasets", "sampled_dataset", "datasets"):
        return Path(sampler_dir.name)
    return Path(dataset_dir.name) / sampler_dir.name


def experiment_dir(split_dir, results_root=None):
    """Return results/{dataset name}/{sampler}/{architecture} for a split dir."""
    root = resolve_path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    return root / experiment_relpath(split_dir) / MODEL_ARCHITECTURE


SPLIT_NAMES = ("train", "val", "test")


def has_split(path, name):
    return (path / f"{name}.csv").is_file() or (path / f"{name}.npy").is_file()


def resolve_split_dir(dataset_path=None, required=SPLIT_NAMES):
    """Resolve a split directory, defaulting to the configured split path."""
    using_default = dataset_path is None
    path = DEFAULT_SPLIT_DIR if using_default else resolve_path(dataset_path)
    if path.is_file():
        path = path.parent
    missing = [name for name in required if not has_split(path, name)]
    if missing:
        hint = ""
        if using_default:
            hint = (
                " No split path was given, so the split configured in config.sh was "
                "used. Set LS_DATA_DIR (or SAMPLING_PROCEDURE/STORAGE_FORMAT), set "
                "LS_CONFIG to another config file, or pass an explicit split "
                "directory."
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

# Compute parameters
ACCELERATOR = env_str("ACCELERATOR", "auto")
NUM_DEVICES = env_str("DEVICES", "1")  # Keep as string since devices can be list/str
PRECISION = env_str("PRECISION", "32")  # Keep as string/int

# Early stopping
EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 8)
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = env_float(
    "EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT", 0.02
)

# Optimization parameters (sequential Optuna study)
OPTUNA_N_TRIALS = env_int("N_TRIALS", 25)
OPTUNA_TUNE_EPOCHS = env_int("TUNE_EPOCHS", 50)
OPTUNA_STUDY_NAME = env_str("STUDY_NAME", "mlp_grav_collapse_optimization")
OPTUNA_STORAGE = env_str("OPTUNA_STORAGE", "auto")
OPTUNA_JOURNAL_MODE = env_str("JOURNAL_MODE", "resume")
OPTUNA_PRUNER_PATIENCE = env_int("PRUNER_PATIENCE", 8)
OPTUNA_MIN_RELATIVE_IMPROVEMENT = env_float("MIN_RELATIVE_IMPROVEMENT", 0.02)

# Search space for hyperparameter tuning
OPTUNA_SEARCH_SPACE = {
    "num_layers": {"type": "int", "low": 2, "high": 5},
    "hidden_units": {"type": "int", "low": 128, "high": 512, "step": 128},
    "learning_rate": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
}

# Test settings
TEST_NUM_TRACERS = env_int("TEST_NUM_TRACERS", 10)
