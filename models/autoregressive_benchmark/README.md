# Autoregressive Benchmark (`autoregressive_benchmark`)

Cross-model autoregressive rollout benchmark. Compares surrogate variants
(MLP, LSTM, RNN, xLSTM, ...) on multi-step rollout accuracy and timing.

## Configuration

Path-specific settings and pipeline arguments live in the shared repo config
at `lightning_surrogates/config.sh`. SLURM scripts source it directly
(override: `LS_CONFIG`); Python entry points read the resulting environment
variables through `src/settings.py`. For local runs, `source config.sh` first.

## Layout

| Dir | Purpose |
| --- | --- |
| `src/` | benchmark driver and metrics code |
| `tests/` | unit / integration tests |
| `slurm/` | cluster job scripts |
| `logs/` | run logs |

## Running

Fixed-architecture benchmark (no hyperparameter optimization) of five variants,
trained **and** tested in parallel via a SLURM job array, after a single
**random** sampling of the dataset split:

```bash
cd models/autoregressive_benchmark
./slurm/submit.sh
```

`submit.sh` submits `sample.slurm` (random sampling) and then the
`benchmark.slurm` array gated on it with `--dependency=afterok`, so training
only starts once the split exists and validates.

Override the dataset / raw H5 for both jobs at submit time:

```bash
./slurm/submit.sh --export=ALL,DATASET_NAME=grav_collapse,SAMPLERS_RAW_H5=/path/to/file.h5
```

The array index maps onto the models (`benchmark.slurm` uses `--array=0-4`):

| Index | Model |
| --- | --- |
| 0 | `t_1_mlp` |
| 1 | `t_20_mlp` |
| 2 | `lstm` |
| 3 | `xlstm` |
| 4 | `rnn` |

Each task runs `train.py --use-defaults` (built-in architecture defaults, no
Optuna) then `test.py` (autoregressive rollout). Results land in
`results/{dataset}/random/{model}/` (checkpoint + `test_results/`).

**Note:** `lstm` (index 2) is a scaffold without `src/` yet, so its task
exits with a clear "not implemented" message until `models/lstm/src/{train,test}.py`
are added — at which point it runs with no further changes.
