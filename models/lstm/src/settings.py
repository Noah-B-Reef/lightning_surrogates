"""Typed defaults for the LSTM pipeline, read from the environment.

The single source of configuration is the shell config at the repo root
(``lightning_surrogates/config.sh``). SLURM jobs get it because
``slurm/common.sh`` sources it before running Python; for local runs either
``source config.sh`` first or rely on the repo-relative defaults below, which
match the standard layout (raw data and splits in the sibling ``datasets/``
directory). This module performs no config-file parsing of its own.
"""

import math
import os
from pathlib import Path

# Identify directories
SRC_DIR = Path(__file__).resolve().parent
LSTM_DIR = SRC_DIR.parent
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
# Experiment results live inside this model's directory:
# models/lstm/results/{dataset name}/{sampler}/.
DEFAULT_RESULTS_ROOT = env_path("RESULTS_ROOT", LSTM_DIR / "results")
CHECKPOINT_NAME = env_str("CHECKPOINT_NAME", f"lstm_{DATASET_NAME}.ckpt")


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
    """Return {model dir}/results/{dataset name}/{sampler} for a split dir."""
    root = resolve_path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    return root / experiment_relpath(split_dir)


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


# Model defaults. The LSTM keeps the MLP's [phys, abund] -> abund_{t+1}
# interface but carries a recurrent hidden state across the trajectory.
# RNN_NUM_LAYERS and RNN_HIDDEN_DIM size the recurrent stack; the cell type is
# locked to LSTM.
RNN_NUM_LAYERS = env_int("MODEL_RNN_NUM_LAYERS", 2)
RNN_HIDDEN_DIM = env_int("MODEL_RNN_HIDDEN_DIM", 256)
RNN_CELL_TYPE = "lstm"  # locked to LSTM
# Dropout between recurrent layers (PyTorch applies it only when num_layers > 1).
RNN_DROPOUT = env_float("MODEL_RNN_DROPOUT", 0.0)
BATCH_SIZE = env_int("MODEL_BATCH_SIZE", 32)
LEARNING_RATE = env_float("MODEL_LEARNING_RATE", 1e-3)
LOSS_FUNCTION = env_str("MODEL_LOSS_FUNCTION", "l1")  # l1 | mse | smooth_l1
EPOCHS = env_int("TRAIN_EPOCHS", env_int("MODEL_EPOCHS", 100))
NUM_WORKERS = env_num_workers()

# Trace-species loss handling. Raw abundances are clipped at ABUND_FLOOR
# before the log10 transform (the old hardcoded floor was 1e-30, which pinned
# unresolved trace species at log10 = -30 and let their noise dominate the
# loss). Targets at or below TRACE_THRESHOLD are downweighted by TRACE_WEIGHT
# in the training loss: they carry no physical signal worth fitting at the
# expense of the dynamically important species.
ABUND_FLOOR = env_float("MODEL_ABUND_FLOOR", 1e-25)
TRACE_THRESHOLD = env_float("MODEL_TRACE_THRESHOLD", 1e-20)
TRACE_THRESHOLD_LOG10 = math.log10(TRACE_THRESHOLD)
TRACE_WEIGHT = env_float("MODEL_TRACE_WEIGHT", 0.1)

# Multi-step rollout training. Each sample is a window of ROLLOUT_STEPS
# consecutive transitions; during the loss the model is fed its own
# predictions autoregressively (true physical drivers, predicted abundances)
# while the recurrent hidden state is threaded through the window, so it trains
# on the error distribution it will face at rollout time. Step j (0-based) is
# weighted by ROLLOUT_DECAY_BASE**j. The training horizon follows a doubling
# curriculum (1, 2, 4, ... up to ROLLOUT_STEPS), advancing every
# ROLLOUT_CURRICULUM_EPOCHS epochs; validation always uses the full horizon so
# val_loss keeps a fixed definition across epochs.
ROLLOUT_STEPS = env_int("MODEL_ROLLOUT_STEPS", 10)
ROLLOUT_DECAY_BASE = env_float("MODEL_ROLLOUT_DECAY_BASE", 0.5)
ROLLOUT_CURRICULUM_EPOCHS = env_int("MODEL_ROLLOUT_CURRICULUM_EPOCHS", 5)

# Learning-rate schedule: none | cosine | plateau. L1/Huber losses keep a
# constant-magnitude gradient at the optimum, so a fixed LR leaves the
# weights dithering (noisy validation curve); decaying the LR settles them.
LR_SCHEDULER = env_str("MODEL_LR_SCHEDULER", "cosine")  # none | cosine | plateau
LR_MIN = env_float("MODEL_LR_MIN", 1e-6)
LR_PLATEAU_FACTOR = env_float("MODEL_LR_PLATEAU_FACTOR", 0.5)
LR_PLATEAU_PATIENCE = env_int("MODEL_LR_PLATEAU_PATIENCE", 3)

# Compute parameters
ACCELERATOR = env_str("ACCELERATOR", "auto")
NUM_DEVICES = env_str("DEVICES", "1")  # Keep as string since devices can be list/str
PRECISION = env_str("PRECISION", "32")  # Keep as string/int

# Early stopping
EARLY_STOPPING_PATIENCE = env_int("EARLY_STOPPING_PATIENCE", 8)
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = env_float(
    "EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT", 0.02
)
# EMA smoothing factor for the early-stopping decision (None disables). A
# noisy val curve can trip patience on a single epoch; smoothing the stop
# signal decouples it from that jitter. Empty/<=0 -> disabled.
_ema = os.environ.get("EARLY_STOPPING_EMA_ALPHA", "0.3").strip()
EARLY_STOPPING_EMA_ALPHA = (
    float(_ema) if _ema and float(_ema) > 0.0 else None
)

# Optimization parameters (sequential Optuna study)
OPTUNA_N_TRIALS = env_int("N_TRIALS", 25)
OPTUNA_TUNE_EPOCHS = env_int("TUNE_EPOCHS", 50)
OPTUNA_STUDY_NAME = env_str("STUDY_NAME", f"lstm_{DATASET_NAME}_optimization")
OPTUNA_STORAGE = env_str("OPTUNA_STORAGE", "auto")
OPTUNA_JOURNAL_MODE = env_str("JOURNAL_MODE", "resume")
OPTUNA_PRUNER_PATIENCE = env_int("PRUNER_PATIENCE", 8)
OPTUNA_MIN_RELATIVE_IMPROVEMENT = env_float("MIN_RELATIVE_IMPROVEMENT", 0.02)

# Search space for hyperparameter tuning. The training loss function is fixed
# to LOSS_FUNCTION; Optuna's objective is val_mse (a fixed metric). The
# recurrent stack depth/width replace the MLP's layers/units.
OPTUNA_SEARCH_SPACE = {
    "rnn_num_layers": {"type": "int", "low": 1, "high": 3},
    "rnn_hidden_dim": {"type": "int", "low": 128, "high": 1024, "step": 128},
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    # Small batches (32-128) made the val loss extremely noisy on the 4M+
    # sample splits; large batches stabilize it and train far faster per epoch.
    "batch_size": {"type": "categorical", "choices": [256, 512, 1024, 2048]},
}

# Test settings
TEST_NUM_TRACERS = env_int("TEST_NUM_TRACERS", 10)
