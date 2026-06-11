# Lightning MLP Surrogate

This directory owns model hyperparameter optimization, final training, and
autoregressive rollout testing for the MLP surrogate. The model is a plain
feed-forward MLP trained with L1 loss on the log10 abundances: it maps
`[physical parameters at t, log10 abundances at t]` to
`log10 abundances at t + 1`.

## Configuration

All path-specific settings and pipeline arguments live in the shared repo
config at `lightning_surrogates/config.sh`. The SLURM scripts source it
directly (location override: `LS_CONFIG`); the Python entry points simply
read the resulting environment variables through `src/settings.py`. For
local runs, `source config.sh` first — or skip it entirely: the Python
defaults are repo-relative and match the standard layout.

Key values:

| Variable | Meaning |
|----------|---------|
| `DATASET_NAME`, `SAMPLING_PROCEDURE`, `STORAGE_FORMAT` | Select the split: `datasets/sampled_datasets/{dataset name}/{sampler}/{format}/` |
| `LS_DATA_DIR` | Explicit split directory (overrides the two above) |
| `RESULTS_ROOT` | Root of experiment results (default `models/mlp/results`) |
| `MODEL_*`, `TRAIN_EPOCHS`, `N_TRIALS`, `TUNE_EPOCHS`, ... | Model / training / Optuna arguments |

A split directory must contain `train`, `val`, and `test` splits in either
storage format produced by the samplers:

```text
train.csv  val.csv  test.csv
# or
train.npy  val.npy  test.npy  columns.json
```

## Experiment results layout

Every experiment writes to its own directory inside this model directory:

```text
models/mlp/results/{dataset name}/{sampler}/
```

where the dataset name and sampler come from the split path (e.g.
`grav_collapse` sampled with `density`). For example:

```text
models/mlp/results/grav_collapse/density/
    optimization/            # Optuna journal, best_params.json
    checkpoints/             # best val_loss checkpoint
    mlp_grav_collapse.ckpt   # final exported checkpoint
    trained_model_config.json
    loss_curves.png
    test_results/            # rollout metrics and plots
```

The scripts derive this directory from the split path automatically; pass
`--results-dir` / `--output-dir` only to override it.

## Optimize hyperparameters (sequential Optuna study)

```bash
python src/optimize.py /path/to/split --num-trials 25 --tune-epochs 50
```

Searches layers, hidden units, learning rate, and batch size (see
`OPTUNA_SEARCH_SPACE` in `src/settings.py`). The training loss function is
fixed to `LOSS_FUNCTION` (`MODEL_LOSS_FUNCTION` env var). The Optuna objective
is validation MSE. Writes
`best_params.json` to
`models/mlp/results/{dataset}/{sampler}/optimization/`. The study journal is a SQLite file in
the same directory; `--journal-mode resume` (default) continues an existing
study, `--journal-mode fresh` starts over.

## Train

```bash
python src/train.py /path/to/split --config-file /path/to/best_params.json
# or with the config.sh defaults:
python src/train.py /path/to/split --use-defaults
```

## Test

```bash
python src/test.py /path/to/split
```

Runs a full autoregressive rollout per test tracer and writes error summaries
and rollout plots to `models/mlp/results/{dataset}/{sampler}/test_results/`.

## SLURM

Submit from this directory (`models/mlp/`); job logs go to `logs/`:

```bash
sbatch slurm/optimize.slurm
sbatch slurm/train.slurm
sbatch slurm/test.slurm
sbatch slurm/pipeline.slurm   # sampling -> optimize -> train -> test
```

To run the full pipeline on a different raw .h5 without editing the config:

```bash
sbatch --export=ALL,DATASET_NAME=gow17_R0.05_M6.0,SAMPLERS_RAW_H5=/path/to/file.h5 slurm/pipeline.slurm
```

Each script sources `slurm/common.sh`, which loads the repo config and
derives the experiment directories.
