# Lightning MLP Surrogate

This directory owns model hyperparameter optimization, final training, and autoregressive rollout testing for the MLP surrogate. It is sampler-agnostic: every model script takes a split dataset directory as input.

A split directory must contain:

```text
train.csv
val.csv
test.csv
```

The sampler repository can generate several candidate split directories, but choosing which split to use is a user decision.

When no split path is given, every script defaults to the best-sampler split exported by the samplers benchmark at `<datasets>/sampled_datasets/best_sampler/`. The `<datasets>` root is the repo's sibling `datasets/` by default; on clusters where that relative layout does not hold (e.g. TACC `/work/...`), set `MLP_DATASETS_DIR` to the datasets root, or `MLP_DATA_DIR` straight to the split directory. That directory does not exist until `run_sampler_benchmark.py` has been run; until then the scripts raise an error pointing you to that benchmark.

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

## Optimize Hyperparameters In Parallel

`src/optimize_parallel.py` lets multiple Slurm ranks run Optuna trials against
one shared study. Use server-backed Optuna storage such as PostgreSQL or MySQL;
do not use SQLite for multi-node optimization.

```bash
srun -n 4 python src/optimize_parallel.py /path/to/split \
  --results-dir /path/to/results/optimization_parallel \
  --storage postgresql://user:password@host:5432/database \
  --num-trials 40 \
  --tune-epochs 50
```

For Slurm batch submission:

```bash
sbatch slurm/optimize_parallel.slurm
```

`--num-trials` is the total trial count across all ranks. The default parallel
study name is `mlp_grav_collapse_optimization_parallel`; set `--study-name` or
`STUDY_NAME` to reuse a different study intentionally. Rank 0 waits for the
requested total to finish, then writes `best_params.json`, `best_params.txt`,
and `optimization_summary.json`.

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

Run optimize -> train -> test for one chosen split through Slurm:

```bash
sbatch slurm/run.slurm
```

Run individual pipeline phases through Slurm:

```bash
sbatch slurm/optimize.slurm
sbatch slurm/train.slurm
sbatch slurm/test.slurm
```

Edit `slurm/config.yaml`, or pass `MLP_SLURM_CONFIG=/path/to/config.yaml`, to
change split paths, result paths, optimization settings, training epochs, and
test output paths.

## Search Space

Default Optuna search space:

- hidden layers: 2–8
- hidden units: 128–1024, step 128
- learning rate: 1e-5 to 1e-2, log scale
- batch size: 16, 32, 64, 128

This broadens capacity beyond the previous shallow/narrow defaults while avoiding the largest 2048-unit models unless explicitly configured.

## Environment Variables

- `MLP_DATASETS_DIR`: datasets root used to locate the default best-sampler split (defaults to the repo's sibling `datasets/`; set this on TACC).
- `MLP_DATA_DIR`: optional default split directory when no path is passed (overrides `MLP_DATASETS_DIR`).
- `MLP_RESULTS_DIR`: default MLP results directory.
- `DATA_DIR`: split directory for Slurm wrappers.
- `RESULTS_DIR`: result root for Slurm wrappers.
- `OPTUNA_RESULTS_DIR`: serial Optuna output directory.
- `N_TRIALS`, `TUNE_EPOCHS`, `TRAIN_EPOCHS`: Slurm wrapper runtime settings.
- `ACCELERATOR`, `DEVICES`, `NUM_WORKERS`: compute settings.
- `MLP_SLURM_CONFIG`: optional path to a replacement Slurm `config.yaml`.

## SLURM

Runtime paths and Python parameters for the Slurm wrappers live in:

```bash
slurm/config.yaml
```

Submit the full serial optimize -> train -> test pipeline:

```bash
sbatch slurm/run.slurm
```

Submit individual stages:

```bash
sbatch slurm/optimize.slurm
sbatch slurm/train.slurm
sbatch slurm/test.slurm
```

Submit parallel Optuna optimization:

```bash
sbatch slurm/optimize_parallel.slurm
```

Set `optimize_parallel.storage` in `slurm/config.yaml` to a server-backed Optuna
RDB URL before using the parallel optimizer. SQLite is not safe for multi-node
Optuna workers.

Use a different config file without editing the repository:

```bash
sbatch --export=ALL,MLP_SLURM_CONFIG=/path/to/config.yaml slurm/run.slurm
```
