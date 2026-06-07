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

Optimization and final training use a masked autoregressive loss over up to
`MODEL_ROLLOUT_STEPS` future steps, defaulting to 5. Short tracer tails are
masked, so samples near the end of a trajectory contribute only their available
future steps.

By default, optimization resumes from the configured SQLite journal and treats
`--num-trials` as the target number of finished trials in that journal. For
example, if the journal already has 10 finished trials and `--num-trials 25`,
the optimizer runs 15 more. Use `--journal-mode fresh` to remove the SQLite
journal and start a new study:

```bash
python src/optimize.py /path/to/split --journal-mode fresh --num-trials 25
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
one shared study. When no storage is provided, it creates
`optuna.sqlite3` under the parallel results directory with longer SQLite busy
timeouts and worker retries for transient lock contention. Server-backed Optuna
storage such as PostgreSQL or MySQL is still recommended for multi-node runs.

```bash
srun -n 4 python src/optimize_parallel.py /path/to/split \
  --results-dir /path/to/results/optimization_parallel \
  --storage postgresql://user:password@host:5432/database \
  --num-trials 40 \
  --tune-epochs 50
```

For Slurm batch submission:

```bash
slurm/submit_optimize_parallel.sh
```

The submit wrapper creates the output/error directories before calling `sbatch`.
If `PARALLEL_OPTUNA_STORAGE` is unset, the job defaults to
`results/optimization_parallel/optuna.sqlite3` and enables retry handling for
short SQLite write-lock collisions. `sbatch` only reports whether the job was
accepted; check runtime failures with `sacct -j <jobid>` and the configured
`results/output/optimize_parallel_<jobid>.out` and
`results/error/optimize_parallel_<jobid>.err` files.

Like the serial optimizer, `--num-trials` is the target number of finished
trials in the study, not a per-run increment. On resume the ranks run only
enough new trials (distributed across ranks) to reach that target; if the study
already has that many finished trials, no new trials run. Use
`--trials-per-worker` for explicit additive behavior instead: each rank runs
exactly that many new trials on top of whatever is already finished. The default
parallel study name is `mlp_grav_collapse_optimization_parallel`; set
`--study-name` or `STUDY_NAME` to reuse a different study intentionally. Rank 0
waits for the target total to finish, then writes `best_params.json`,
`best_params.txt`, and `optimization_summary.json`.

## Train Final Model

```bash
python src/train.py /path/to/split   --results-dir /path/to/results   --config-file /path/to/results/optimization/best_params.json   --epochs 100
```

Prediction behavior and rollout-loss horizon can be selected without editing
the model code:

```bash
# Predict x_{t+1} directly with one-step rollout loss.
python src/train.py /path/to/split \
  --results-dir /path/to/results/direct_h1 \
  --use-defaults --epochs 100 --rollout-steps 1 \
  --prediction-mode direct --no-early-stopping --no-epoch-checkpoints

# Predict x_{t+1} directly with five-step rollout loss.
python src/train.py /path/to/split \
  --results-dir /path/to/results/direct_h5 \
  --use-defaults --epochs 100 --rollout-steps 5 \
  --prediction-mode direct --no-early-stopping --no-epoch-checkpoints

# Predict dx and advance with x_{t+1} = x_t + dx.
python src/train.py /path/to/split \
  --results-dir /path/to/results/delta_h5 \
  --use-defaults --epochs 100 --rollout-steps 5 \
  --prediction-mode delta --no-early-stopping --no-epoch-checkpoints
```

The prediction mode is stored in the checkpoint, so `test.py` automatically
uses the correct direct or delta update during autoregressive rollout:

```bash
python src/test.py /path/to/split \
  --model-checkpoint /path/to/results/mlp_grav_collapse.ckpt \
  --output-dir /path/to/results/test_results \
  --accelerator auto
```

### Fast Horizon-5 Profile

Training supports independent controls for batch size, sample overlap,
validation cost, cache layout, staged initialization, and architecture. The
recommended speed-quality profile keeps the 3x256 LayerNorm architecture,
initializes from a horizon-1 checkpoint, and reduces redundant data work:

```bash
python src/train.py /path/to/split \
  --results-dir /path/to/results/fast_h5 \
  --use-defaults --epochs 100 \
  --rollout-steps 5 --prediction-mode direct \
  --batch-size 1024 \
  --train-sample-stride 2 \
  --val-fraction 0.2 \
  --val-every-n-epochs 5 \
  --val-rollout-steps 1 \
  --compact-batches \
  --init-checkpoint /path/to/horizon1/mlp_grav_collapse.ckpt \
  --no-early-stopping --no-epoch-checkpoints --no-logger
```

`--compact-batches` uses row-level memory-mapped arrays and constructs each
batch vectorially. Compact caches omit duplicated `initial`, `phys_seq`,
`target_seq`, and `mask` arrays. Existing full caches are migrated with hard
links rather than reparsing the CSV.

Architecture experiments are also CLI-selectable:

```bash
python src/train.py /path/to/split \
  --use-defaults --hidden-layers 2 --hidden-units 128 --no-layer-norm
```

The smaller architecture is faster but performed substantially worse in the
recorded full-rollout benchmark under `results/speed_bench/`.

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

Edit `config.sh`, or pass `MLP_SLURM_CONFIG=/path/to/config.sh`, to
change split paths, result paths, optimization settings, training epochs, and
test output paths.

## Search Space

Default Optuna search space:

- hidden layers: 2–5
- hidden units: 128–512, step 128
- learning rate: 1e-4 to 5e-3, log scale
- batch size: 32, 64, 128

The standard and local optimizers share this search space.

## Local MPS Helpers

Local-only wrappers live under:

```bash
src/local_impl/
```

Run the local optimize -> train -> test path with:

```bash
python src/local_impl/run_local.py
```

These helpers support optional tracer subsampling and cache loaded datasets
across threaded local Optuna trials.

## Environment Variables

- `MLP_DATASETS_DIR`: datasets root used to locate the default best-sampler split (defaults to the repo's sibling `datasets/`; set this on TACC).
- `MLP_DATA_DIR`: optional default split directory when no path is passed (overrides `MLP_DATASETS_DIR`).
- `MLP_RESULTS_DIR`: default MLP results directory.
- `DATA_DIR`: split directory for Slurm wrappers.
- `RESULTS_DIR`: result root for Slurm wrappers.
- `OPTUNA_RESULTS_DIR`: serial Optuna output directory.
- `N_TRIALS`, `TUNE_EPOCHS`, `TRAIN_EPOCHS`: Slurm wrapper runtime settings.
- `MODEL_ROLLOUT_STEPS`: maximum rollout horizon used by the training loss.
- `ACCELERATOR`, `DEVICES`, `NUM_WORKERS`: compute settings.
- `MLP_SLURM_CONFIG`: optional path to a replacement config.sh script.

## SLURM

Runtime paths and Python parameters for the Slurm wrappers live in:

```bash
config.sh
```

The Python defaults in `src/settings.py` are also loaded from this shell script.
Set `MLP_CONFIG` or `MLP_SLURM_CONFIG` to point both Python and Slurm wrappers
at a different config. When changing `OPTUNA_RESULTS_DIR`, update
`TRAIN_CONFIG_FILE` if final training should consume that optimizer output.

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

The Slurm optimize/train wrappers request `--cpus-per-task=8`, so
`CONFIG_NUM_WORKERS=auto` resolves to eight DataLoader workers by default.

Set `PARALLEL_OPTUNA_STORAGE` in `config.sh` to a server-backed Optuna
RDB URL before using the parallel optimizer. SQLite is not safe for multi-node
Optuna workers at high concurrency, even with retry handling.

Use a different config file without editing the repository:

```bash
sbatch --export=ALL,MLP_SLURM_CONFIG=/path/to/config.sh slurm/run.slurm
```
