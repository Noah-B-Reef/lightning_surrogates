# Lightning RNN Surrogate

This directory owns model hyperparameter optimization, final training, and
autoregressive rollout testing for the RNN surrogate. The model is a recurrent
network (LSTM by default, GRU optional) trained on the log10 abundances: it maps
`[physical parameters at t, log10 abundances at t]` to
`log10 abundances at t + 1`.

Unlike the stateless MLP, the RNN carries a recurrent hidden state across the
trajectory, so each prediction is conditioned on the history of the rollout
rather than only the current state. Training unrolls the model autoregressively
over a window of consecutive transitions (true physical drivers, the model's own
abundance predictions, hidden state threaded through), matching the error
distribution it faces at rollout time; at test time the hidden state is threaded
across the full tracer.

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
| `RESULTS_ROOT` | Root of experiment results (default `models/rnn/results`) |
| `MODEL_RNN_CELL_TYPE` | Recurrent cell: `lstm` (default) or `gru` |
| `MODEL_RNN_NUM_LAYERS` | Stacked recurrent layers (default 2) |
| `MODEL_RNN_HIDDEN_DIM` | Hidden size per layer (default 256) |
| `MODEL_RNN_DROPOUT` | Inter-layer dropout, applied only when `num_layers > 1` (default 0) |
| `MODEL_*`, `TRAIN_EPOCHS`, `N_TRIALS`, `TUNE_EPOCHS`, ... | Shared model / training / Optuna arguments |

The shared loss, trace-species downweighting, multi-step rollout curriculum,
LR schedule, and early-stopping knobs (`MODEL_LOSS_FUNCTION`,
`MODEL_TRACE_*`, `MODEL_ROLLOUT_*`, `MODEL_LR_*`, `EARLY_STOPPING_*`) are read
exactly as for the MLP.

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
models/rnn/results/{dataset name}/{sampler}/
```

where the dataset name and sampler come from the split path (e.g.
`grav_collapse` sampled with `density`). For example:

```text
models/rnn/results/grav_collapse/density/
    optimization/            # Optuna journal, best_params.json
    checkpoints/             # best val_loss checkpoint
    rnn_grav_collapse.ckpt   # final exported checkpoint
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

Searches recurrent layers, hidden dim, learning rate, and batch size (see
`OPTUNA_SEARCH_SPACE` in `src/settings.py`). The training loss function is
fixed to `LOSS_FUNCTION` (`MODEL_LOSS_FUNCTION` env var); the cell type and
dropout come from `MODEL_RNN_CELL_TYPE` / `MODEL_RNN_DROPOUT`. The Optuna
objective is validation MSE. Writes `best_params.json` to
`models/rnn/results/{dataset}/{sampler}/optimization/`. The study journal is a
SQLite file in the same directory; `--journal-mode resume` (default) continues
an existing study, `--journal-mode fresh` starts over.

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

Runs a full autoregressive rollout per test tracer (hidden state threaded
across the whole tracer) and writes error summaries and rollout plots to
`models/rnn/results/{dataset}/{sampler}/test_results/`.

## SLURM

Submit from this directory (`models/rnn/`); job logs go to `logs/`:

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

Each script sources `slurm/common.sh`, which loads the repo config, sets the
RNN-specific checkpoint/study names, and derives the experiment directories.
