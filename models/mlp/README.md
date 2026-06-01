# Lightning MLP Surrogate

This directory owns model hyperparameter optimization, final training, and autoregressive rollout testing for the MLP surrogate. It is sampler-agnostic: every model script takes a split dataset directory as input.

A split directory must contain:

```text
train.csv
val.csv
test.csv
```

The sampler repository can generate several candidate split directories, but choosing which split to use is a user decision.

When no split path is given, every script defaults to the best-sampler split exported by the samplers benchmark at `<datasets>/sampled_datasets/best_sampler/`. The `<datasets>` root is the repo's sibling `datasets/` by default; on clusters where that relative layout does not hold (e.g. TACC `/work/...`), set `MLP_DATASETS_DIR` to the datasets root, or `MLP_DATA_DIR` straight to the split directory. That directory does not exist until `run_sampler_benchmark.py` has been run; until then the scripts raise an error pointing you to that benchmark. So `python run_benchmark.py` with no arguments runs against the best sampler automatically.

## Optimize Hyperparameters

```bash
python src/optimize.py /path/to/split   --results-dir /path/to/results/optimization   --num-trials 25   --tune-epochs 50
```

Outputs:

```text
/path/to/results/optimization/best_params.json
/path/to/results/optimization/best_params.txt
/path/to/results/optimization/optimization_summary.json
/path/to/results/optimization/optuna.sqlite3
```

## Train Final Model

```bash
python src/train.py /path/to/split   --results-dir /path/to/results   --config-file /path/to/results/optimization/best_params.json   --epochs 100
```

Outputs:

```text
/path/to/results/mlp_grav_collapse.ckpt
/path/to/results/trained_model_config.json
/path/to/results/loss_curves.png
/path/to/results/checkpoints/
```

## Test With Autoregressive Rollout

```bash
python src/test.py /path/to/split   --model-checkpoint /path/to/results/mlp_grav_collapse.ckpt   --output-dir /path/to/results/test_results
```

Outputs:

```text
/path/to/results/test_results/error_summary.json
/path/to/results/test_results/tracer_errors.csv
/path/to/results/test_results/species_mse.csv
/path/to/results/test_results/test_predictions_log10.csv
/path/to/results/test_results/rollouts/*.png
```

## Run Full Model Benchmark

Run optimize → train → test for one chosen split:

```bash
./run_benchmark.sh /path/to/split
```

or:

```bash
python run_benchmark.py /path/to/split --results-dir /path/to/results
```

Skip individual phases when resuming:

```bash
SKIP_OPTIMIZE=1 ./run_benchmark.sh /path/to/split
python run_benchmark.py /path/to/split --skip-optimize
```

## Search Space

Default Optuna search space:

- hidden layers: 2–8
- hidden units: 128–1024, step 128
- learning rate: 1e-5 to 1e-2, log scale
- batch size: 16, 32, 64, 128
- dropout: 0.0–0.3
- weight decay: 1e-5 to 1e-2, log scale

This broadens capacity beyond the previous shallow/narrow defaults while avoiding the largest 2048-unit models unless explicitly configured.

## Environment Variables

- `MLP_DATASETS_DIR`: datasets root used to locate the default best-sampler split (defaults to the repo's sibling `datasets/`; set this on TACC).
- `MLP_DATA_DIR`: optional default split directory when no path is passed (overrides `MLP_DATASETS_DIR`).
- `MLP_RESULTS_DIR`: default MLP results directory.
- `DATA_DIR`: split directory for `run_benchmark.sh` and `run.slurm`.
- `RESULTS_DIR`: result root for `run_benchmark.sh` and `run.slurm`.
- `OPTUNA_RESULTS_DIR`: Optuna output directory for wrappers.
- `N_TRIALS`, `TUNE_EPOCHS`, `TRAIN_EPOCHS`: wrapper runtime settings.
- `ACCELERATOR`, `DEVICES`, `NUM_WORKERS`: compute settings.
- `SKIP_OPTIMIZE`, `SKIP_TRAIN`, `SKIP_TEST`: skip flags for `run_benchmark.sh`.

## SLURM

```bash
sbatch --export=ALL,DATA_DIR=/path/to/split,RESULTS_DIR=/path/to/results run.slurm
```
