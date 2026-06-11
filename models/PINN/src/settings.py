"""Typed defaults for the PINN pipeline, read from the environment.

Mirrors models/mlp/src/settings.py: the single source of configuration is
``lightning_surrogates/config.sh`` (sourced by the SLURM scripts); local runs
fall back to the repo-relative defaults below. PINN-specific knobs are at the
bottom.
"""

import os
from pathlib import Path

# Identify directories
SRC_DIR = Path(__file__).resolve().parent
PINN_DIR = SRC_DIR.parent
LIGHTNING_SURROGATES_DIR = SRC_DIR.parents[2]
RESEARCH_DIR = SRC_DIR.parents[3]


# Helper functions to retrieve typed environment variables
def env_str(key, default):
    return os.environ.get(key, str(default))


def env_path(key, default):
    val = os.environ.get(key)
    if val is None:
        return Path(default).expanduser().resolve()
    return Path(val).expanduser().resolve()


def env_int(key, default):
    return int(os.environ.get(key, str(default)))


def env_float(key, default):
    return float(os.environ.get(key, str(default)))


def env_num_workers(key="CONFIG_NUM_WORKERS", default=0):
    value = os.environ.get(key, str(default))
    if str(value).lower() == "auto":
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    return int(value)


def resolve_path(path):
    return Path(path).expanduser().resolve()


# Default paths (resolved relative to the workspace unless set in the environment)
DEFAULT_DATASETS_DIR = env_path("DATASETS_DIR", RESEARCH_DIR / "datasets")
DATASET_NAME = env_str("DATASET_NAME", "gow17_R0.05_M6.0")
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
MODEL_ARCHITECTURE = "pinn"
CHECKPOINT_NAME = env_str("PINN_CHECKPOINT_NAME", "pinn_gow17.ckpt")


def experiment_relpath(split_dir):
    """Derive {dataset name}/{sampler} from a split directory."""
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
                " No split path was given, so the default split was used. Set "
                "LS_DATA_DIR (or DATASET_NAME/SAMPLING_PROCEDURE/STORAGE_FORMAT) "
                "in the environment — e.g. `source config.sh` — or pass an "
                "explicit split directory."
            )
        raise FileNotFoundError(f"Missing {missing} in split directory: {path}.{hint}")
    return path


# Model defaults
NUM_LAYERS = env_int("MODEL_NUM_LAYERS", 3)
HIDDEN_UNITS = env_int("MODEL_HIDDEN_UNITS", 256)
BATCH_SIZE = env_int("MODEL_BATCH_SIZE", 1024)
LEARNING_RATE = env_float("MODEL_LEARNING_RATE", 1e-3)
EPOCHS = env_int("TRAIN_EPOCHS", env_int("MODEL_EPOCHS", 100))
NUM_WORKERS = env_num_workers()

# Compute parameters
ACCELERATOR = env_str("ACCELERATOR", "auto")
NUM_DEVICES = env_str("DEVICES", "1")
PRECISION = env_str("PRECISION", "32")

# Early stopping
EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 8)
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = env_float(
    "EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT", 0.02
)

# Test settings
TEST_NUM_TRACERS = env_int("TEST_NUM_TRACERS", 10)

# ---------------------------------------------------------------------------
# Hyperparameter optimization (sequential Optuna study, mirrors models/mlp)
# ---------------------------------------------------------------------------
OPTUNA_N_TRIALS = env_int("N_TRIALS", 25)
OPTUNA_TUNE_EPOCHS = env_int("TUNE_EPOCHS", 50)
OPTUNA_STUDY_NAME = env_str(
    "PINN_STUDY_NAME", f"pinn_{DATASET_NAME}_optimization"
)
OPTUNA_STORAGE = env_str("OPTUNA_STORAGE", "auto")
OPTUNA_JOURNAL_MODE = env_str("JOURNAL_MODE", "resume")
OPTUNA_PRUNER_PATIENCE = env_int("PRUNER_PATIENCE", 8)
OPTUNA_MIN_RELATIVE_IMPROVEMENT = env_float("MIN_RELATIVE_IMPROVEMENT", 0.02)

# Search space. The Optuna objective is val_mse (data-space MSE on log10
# abundances) — a fixed metric independent of the loss weights, so trials
# with different physics/conservation weights are comparable.
OPTUNA_SEARCH_SPACE = {
    "num_layers": {"type": "int", "low": 2, "high": 6},
    "hidden_units": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 3e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [512, 1024, 2048]},
    "physics_weight": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    "conservation_weight": {"type": "float", "low": 1e-4, "high": 0.5, "log": True},
}

# ---------------------------------------------------------------------------
# PINN-specific settings
# ---------------------------------------------------------------------------
# Weight of the ODE-residual loss relative to the data loss.
PHYSICS_WEIGHT = env_float("PINN_PHYSICS_WEIGHT", 0.1)
# Weight of the charge/element conservation loss (0 disables).
CONSERVATION_WEIGHT = env_float("PINN_CONSERVATION_WEIGHT", 0.01)
# Training pairs span 1..MAX_HORIZON consecutive snapshots, so the model sees
# a range of Delta-t values (this is what makes time a real input).
MAX_HORIZON = env_int("PINN_MAX_HORIZON", 4)
# Delta-t normalization scale in years (dataset snapshot spacing is 253 yr).
DT_REF_YEARS = env_float("PINN_DT_REF_YEARS", 1000.0)
# Physical CR ionization rate: zeta column value (constant in this dataset)
# times this unit, in s^-1. Calibrated against the dataset itself from the
# H2+ quasi-steady-state balance (zeta_phys = 2.024e-17 s^-1, spread < 1%;
# see validate_rhs.py), since the column's unit is not documented.
ZETA_UNIT = env_float("PINN_ZETA_UNIT", 1.657e-17)
# Rate floor [s^-1] in the relative-residual denominator: species whose
# production and destruction are both slower than this are treated as frozen
# and contribute ~0 residual instead of noise.
RESIDUAL_RATE_FLOOR = env_float("PINN_RESIDUAL_RATE_FLOOR", 1e-14)
# Evaluate the ODE residual at a uniformly random tau in (0, dt] per sample
# (1) instead of at the endpoint dt (0).
RANDOM_COLLOCATION = env_int("PINN_RANDOM_COLLOCATION", 1)
