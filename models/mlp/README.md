# Lightning MLP Surrogate

This directory owns model hyperparameter optimization, final training, and
autoregressive rollout testing for the MLP surrogate. The model is a plain
feed-forward MLP trained with L1 loss on the log10 abundances: it maps
`[physical parameters at t, log10 abundances at t]` to
`log10 abundances at t + 1`.

## Configuration

All path-specific settings and pipeline arguments live in the shared repo
config at `lightning_surrogates/config.sh` (override its location with
`LS_CONFIG`). The SLURM scripts source it, and the Python entry points read
the same values through `src/settings.py`. Paths default to the repository
location, so the same config works locally and on TACC.

Key values:

| Variable | Meaning |
|----------|---------|
| `DATASET_NAME`, `SAMPLING_PROCEDURE`, `STORAGE_FORMAT` | Select the split: `datasets/sampled_datasets/{dataset name}/{sampler}/{format}/` |
| `LS_DATA_DIR` | Explicit split directory (overrides the two above) |
| `RESULTS_ROOT` | Root of experiment results (default `lightning_surrogates/results`) |
| `MODEL_*`, `TRAIN_EPOCHS`, `N_TRIALS`, `TUNE_EPOCHS`, ... | Model / training / Optuna arguments |

A split directory must contain `train`, `val`, and `test` splits in either
storage format produced by the samplers:

```text
train.csv  val.csv  test.csv
# or
train.npy  val.npy  test.npy  columns.json
```

## Experiment results layout

Every experiment writes to its own directory:

```text
{RESULTS_ROOT}/{dataset name}/{sampler}/{model architecture}/
```

where the dataset name and sampler come from the split path (e.g.
`grav_collapse` sampled with `density`) and the architecture is `mlp`.
For example:

```text
results/grav_collapse/density/mlp/
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
`OPTUNA_SEARCH_SPACE` in `src/settings.py`) and writes `best_params.json` to
`results/{dataset}/{sampler}/mlp/optimization/`. The study journal is a SQLite file in
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
and rollout plots to `results/{dataset}/{sampler}/mlp/test_results/`.

## SLURM

Submit from this directory (`models/mlp/`); job logs go to `logs/`:

```bash
sbatch slurm/optimize.slurm
sbatch slurm/train.slurm
sbatch slurm/test.slurm
sbatch slurm/run.slurm     # optimize -> train -> test
```

Each script sources `slurm/common.sh`, which loads the repo config and
derives the experiment directories.
