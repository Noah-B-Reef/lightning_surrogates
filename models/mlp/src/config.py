import os
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
MLP_DIR = SRC_DIR.parent
LIGHTNING_SURROGATES_DIR = SRC_DIR.parents[2]
RESEARCH_DIR = SRC_DIR.parents[3]


def resolve_path(path):
    return Path(path).expanduser().resolve()


def env_path(name, default):
    return resolve_path(os.environ.get(name, default))


# Root of the datasets tree. Defaults to the sibling `datasets/` of the repo,
# which is correct for the local checkout. On clusters where that relative
# layout does not hold (e.g. TACC /work), set MLP_DATASETS_DIR or point
# MLP_DATA_DIR straight at the split directory.
DEFAULT_DATASETS_DIR = env_path("MLP_DATASETS_DIR", RESEARCH_DIR / "datasets")


# Model/training defaults
NUM_LAYERS = 3
HIDDEN_UNITS = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100

# Data/result defaults. Override these on HPC rather than editing code.
# Every model script takes an explicit split directory; when none is given they
# fall back to the best-sampler split exported by the samplers benchmark
# (datasets/sampled_datasets/best_sampler). That directory does not exist until
# run_sampler_benchmark.py has been run.
BEST_SAMPLER_DIRNAME = "best_sampler"
DEFAULT_SPLIT_DIR = env_path(
    "MLP_DATA_DIR", DEFAULT_DATASETS_DIR / "sampled_datasets" / BEST_SAMPLER_DIRNAME
)
DEFAULT_RESULTS_DIR = env_path("MLP_RESULTS_DIR", MLP_DIR / "results")

SPLIT_FILES = ("train.csv", "val.csv", "test.csv")


def resolve_split_dir(dataset_path=None, required=SPLIT_FILES):
    """Resolve a split directory, defaulting to the best-sampler split.

    Pass a directory (or its train.csv) to use an explicit split. With no
    argument the default best-sampler split is used, and a missing one raises a
    message pointing at the samplers benchmark that produces it.
    """
    using_default = dataset_path is None
    path = DEFAULT_SPLIT_DIR if using_default else resolve_path(dataset_path)
    if path.is_file():
        path = path.parent
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        hint = ""
        if using_default:
            hint = (
                " No split path was given, so the default best-sampler dataset was "
                "used. Run the samplers benchmark (run_sampler_benchmark.py) to "
                "generate it, set MLP_DATA_DIR, or pass an explicit split directory."
            )
        raise FileNotFoundError(f"Missing {missing} in split directory: {path}.{hint}")
    return path
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
}
