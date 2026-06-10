"""Path resolution utilities for the samplers package.

The samplers live inside the lightning_surrogates repository
(lightning_surrogates/samplers/). All default paths are resolved from
environment variables so the code runs without modification on both local
machines and TACC HPC nodes. The shared repo config
(lightning_surrogates/config.sh) exports these variables for SLURM jobs.

Environment variables
---------------------
SAMPLERS_RAW_H5
    Absolute path to the raw postprocessed chemistry HDF5 file.
    The file may be named either:
      - grav_collapse_postprocessed_uclchem.h5          (short alias)
      - grav_collapse_postprocessed_chemistry_uclchem.h5 (full name)
    Both names are accepted; the resolver tries the short alias first, then
    falls back to the full name in the same directory.

SAMPLERS_DATA_DIR
    Directory that contains ``flattened_dataset.h5`` and receives sampler
    split output sub-directories ({procedure}/{format}/, e.g. density/npy/).

SAMPLERS_RESULTS_DIR
    Directory for benchmark outputs (sampler_benchmark_results.json, etc.).
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Anchor paths derived from this file's location
# ---------------------------------------------------------------------------
SAMPLERS_DIR = Path(__file__).resolve().parent          # lightning_surrogates/samplers/
REPO_DIR = SAMPLERS_DIR.parent                          # lightning_surrogates/
RESEARCH_DIR = REPO_DIR.parent                          # research/
DATASETS_DIR = RESEARCH_DIR / "datasets"                # research/datasets/

# Default relative paths (used when env vars are not set)
_DEFAULT_DATA_DIR = DATASETS_DIR / "sampled_dataset"
_DEFAULT_BUNDLE_PATH = _DEFAULT_DATA_DIR / "flattened_dataset.h5"
_DEFAULT_RESULTS_DIR = SAMPLERS_DIR / "results"

# Canonical raw HDF5 filename (full name used by the postprocessing pipeline)
_RAW_H5_FULL_NAME = "grav_collapse_postprocessed_chemistry_uclchem.h5"
# Short alias used in documentation and by the user
_RAW_H5_SHORT_NAME = "grav_collapse_postprocessed_uclchem.h5"

_DEFAULT_RAW_H5 = (
    DATASETS_DIR / "grav_collapse" / "baseline" / _RAW_H5_FULL_NAME
)

# Public constant kept for backward compatibility with benchmark_samplers.py
DEFAULT_BUNDLE_PATH = _DEFAULT_BUNDLE_PATH
DEFAULT_DATA_DIR = _DEFAULT_DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(path) -> Path:
    """Expand user home and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def _resolve_raw_h5_candidate(candidate: Path) -> Path:
    """Given a candidate path, also try the short-alias sibling if needed."""
    if candidate.exists():
        return candidate
    # If the env var points to the short alias, try the full name in the same dir
    sibling_full = candidate.parent / _RAW_H5_FULL_NAME
    if sibling_full.exists():
        return sibling_full
    # If the env var points to the full name, try the short alias
    sibling_short = candidate.parent / _RAW_H5_SHORT_NAME
    if sibling_short.exists():
        return sibling_short
    # Return the original candidate; callers handle missing files
    return candidate


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def default_data_dir() -> Path:
    """Return the sampled-dataset directory, respecting SAMPLERS_DATA_DIR."""
    return resolve_path(os.environ.get("SAMPLERS_DATA_DIR", _DEFAULT_DATA_DIR))


def default_raw_h5() -> Path:
    """Return the raw HDF5 path, respecting SAMPLERS_RAW_H5.

    Accepts both the short alias (grav_collapse_postprocessed_uclchem.h5) and
    the full name (grav_collapse_postprocessed_chemistry_uclchem.h5).
    """
    env_val = os.environ.get("SAMPLERS_RAW_H5")
    if env_val:
        return _resolve_raw_h5_candidate(resolve_path(env_val))
    return _resolve_raw_h5_candidate(_DEFAULT_RAW_H5)


def default_results_dir() -> Path:
    """Return the results directory, respecting SAMPLERS_RESULTS_DIR."""
    return resolve_path(
        os.environ.get("SAMPLERS_RESULTS_DIR", _DEFAULT_RESULTS_DIR)
    )
