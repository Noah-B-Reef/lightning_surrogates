# Samplers and Dataset Similarity Benchmarking

This directory owns dataset preparation and sampler evaluation for the
gravitational-collapse UCLCHEM chemistry dataset. It stops at producing and
benchmarking train/val/test split datasets.

---

## Directory Contents

| File | Purpose |
|------|---------|
| `flatten_dataset.py` | Converts the raw postprocessed chemistry HDF5 into a flattened trajectory bundle. |
| `main.py` | Generates train/val/test split CSVs for each sampler. |
| `benchmark_samplers.py` | Benchmarks split similarity and memorization-risk metrics. |
| `samplers.py` | Sampler implementations: `random_sample`, `density_sampler`, `QR_sampler`, `svd_fps`. |
| `util.py` | Flatten / reformat / split helper functions. |
| `path_utils.py` | Environment-variable-aware path resolution (plug-and-play on TACC). |
| `run_samplers.py` | Pipeline step 1: flatten if needed and generate sampler split datasets. |
| `run_sampler_benchmark.py` | Pipeline step 2: benchmark samplers and export the best split dataset. |
| `run_benchmark.sh` | Shell wrapper that runs both pipeline steps. |
| `sample_datasets.slurm` | TACC SLURM script for `run_samplers.py`. |
| `benchmark_samplers.slurm` | TACC SLURM script for `run_sampler_benchmark.py`. |

---

## Dataset

**Raw file:** `grav_collapse_postprocessed_uclchem.h5`
(also stored as `grav_collapse_postprocessed_chemistry_uclchem.h5` — both
names are accepted by `path_utils.py`).

The file is a pandas HDF5 containing a long-format DataFrame with columns:

- `Tracer` — tracer particle identifier
- `Time` — simulation time
- Physical parameters: `Density`, `gasTemp`, `dustTemp`, `Av`, `radfield`, `zeta`
- ~330 chemical abundance species (UCLCHEM output)

`util.dataset_flatten()` converts this into a **flattened bundle**: one vector
per tracer trajectory of shape `(n_tracers, T × n_features)`.  Abundance
columns are stored as `log10(max(value, 1e-30))`; physical columns remain
linear.

---

## Plug-and-Play Path Configuration

All scripts resolve paths from environment variables.  Set these once and
every script picks them up automatically — no code edits needed on TACC or
any other system:

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMPLERS_RAW_H5` | `datasets/grav_collapse/baseline/grav_collapse_postprocessed_chemistry_uclchem.h5` | Raw HDF5 file (short alias also accepted) |
| `SAMPLERS_DATA_DIR` | `datasets/sampled_dataset` | Flattened bundle + split outputs ({sampler}/{format}/) |
| `SAMPLERS_RESULTS_DIR` | `lightning_surrogates/samplers/results` | Benchmark JSON and CSV outputs |

On SLURM, these are exported by the shared repo config
(`lightning_surrogates/config.sh`), which `sample_datasets.slurm` sources.

Example (TACC):

```bash
export SAMPLERS_RAW_H5=/scratch/$USER/data/grav_collapse_postprocessed_uclchem.h5
export SAMPLERS_DATA_DIR=/scratch/$USER/sampled_dataset
export SAMPLERS_RESULTS_DIR=/scratch/$USER/results
```

### Split output layout and storage format

`main.py` / `run_samplers.py` accept `--storage-format {csv,npy}` (default
`csv`). Each sampler writes its splits to:

```
SAMPLERS_DATA_DIR/{sampling procedure}/{storage format}/
    train.csv  val.csv  test.csv            # csv format
    train.npy  val.npy  test.npy  columns.json   # npy format
```

The `.npy` files are float64 arrays in long format (rows = tracer timesteps);
`columns.json` records the shared column order.

---

## Sampling Strategies

### `random`

**File:** `samplers.py → random_sample()`

Uniformly samples `n_samples` tracer IDs without replacement using a seeded
RNG, shuffles the result, and applies a 60/20/20 train/val/test split.

- **Strengths:** Fast, unbiased, simple baseline.
- **Weaknesses:** No guarantee of coverage across the density or chemical
  abundance space; may over-represent common trajectories.
- **Memorization risk:** Moderate — val/test trajectories are drawn from the
  same uniform distribution as training, so near-duplicate pairs can appear
  by chance.

### `density`

**File:** `samplers.py → density_sampler()`

Stratifies tracers by their **final-timestep gas density** (`Density` column
at the last time step).  The sampler:

1. Bins `log10(final_density)` into `num_strat_bins` equal-width bins.
2. Allocates samples to bins proportionally to their population in the full
   dataset (with remainder distributed to the bins with the largest
   fractional shortfall).
3. Samples without replacement inside each bin.
4. Shuffles the combined result and applies a 60/20/20 split.

- **Strengths:** Preserves the final-density distribution of the full dataset
  better than uniform random sampling; reduces density-space gaps in training.
- **Weaknesses:** Only stratifies on one physical dimension; does not
  explicitly diversify the chemical abundance trajectories.
- **Memorization risk:** Similar to random within each density bin; the
  stratification does not prevent near-duplicate chemical trajectories.

### `qr_pivot`

**File:** `samplers.py → QR_sampler()`

Selects training tracers via **column-pivoted QR decomposition** of the
trajectory matrix:

1. Builds a matrix `A` of shape `(n_features, n_candidates)` where each
   column is a flattened tracer trajectory.
2. Runs Householder QR with column pivoting (`scipy.linalg.qr(pivoting=True,
   mode="r")`).  The pivot order ranks columns by their contribution to the
   column space of `A`.
3. Takes the first `n_train` pivots as training tracers — these are the most
   linearly independent trajectories in the candidate pool.
4. Samples val and test tracers uniformly from the remaining IDs.

The R matrix and pivot indices are saved under
`<data_dir>/qr_pivot/qr_samples/` for inspection.

- **Strengths:** Maximally diverse training set in a linear-algebra sense;
  minimizes redundancy among training trajectories.
- **Weaknesses:** Can shift the training distribution away from the full
  population (extreme trajectories are preferred); review the density KS
  statistic before using for final training.
- **Memorization risk:** Low for training-set redundancy; val/test are still
  random draws so cross-split similarity depends on the dataset.

### `svd_fps`

**File:** `samplers.py → svd_fps()`

Selects tracers via **farthest-point sampling (FPS) in a truncated SVD
embedding**:

1. Centers the trajectory matrix by subtracting the column mean.
2. Computes a randomized truncated SVD (`sklearn.utils.extmath.randomized_svd`)
   with `n_components` principal components.
3. Projects tracers into the reduced space: `embedding = U[:, :k] * S[:k]`.
4. Runs greedy FPS: starts from one seeded random tracer and iteratively
   selects the tracer farthest (in Euclidean distance) from the already-
   selected set, updating minimum distances after each selection.
5. Applies a 60/20/20 split on the selected tracers.

- **Strengths:** Strongly favors diverse, well-spread trajectories; covers
  the extremes of the trajectory space; reduces within-split redundancy.
- **Weaknesses:** Computationally heavier than random/density for large
  datasets (O(n_tracers × n_selected) distance updates); extreme trajectories
  may not be representative of the typical training distribution.
- **Memorization risk:** Low — FPS by construction maximizes pairwise
  distances, so near-duplicate training trajectories are rare.

---

## Similarity Metrics and Memorization Risk

`benchmark_samplers.py` computes the following metrics for each sampler:

### Cross-split nearest-trajectory similarity (`val_to_train`, `test_to_train`)

For each val (or test) trajectory, finds its nearest training trajectory by
**centered cosine similarity** and records the similarity value.  Reports:

- `mean_nearest_similarity` — average over all val/test trajectories
- `median_nearest_similarity`
- `p05` / `p95` quantiles
- `fraction_at_or_above` at each configured threshold (e.g. 0.95, 0.99,
  0.999, 0.9999)

**Near-duplicate rate** = average of `val_to_train` and `test_to_train`
fractions at the primary threshold (default 0.9999).  A high near-duplicate
rate means the model can memorize training trajectories and still score well
on val/test — the primary signal for memorization risk.

### Within-split pairwise similarity

Pairwise cosine similarity among trajectories within each split (train, val,
test).  High within-train similarity means the training set is redundant;
the model may overfit to a narrow region of trajectory space.

### Coverage: full dataset → training set

Nearest-neighbor similarity from the full dataset to the training set.  High
`mean_nearest_distance` means the training set leaves large gaps in trajectory
space uncovered.

### Final-density KS statistic

Kolmogorov–Smirnov statistic comparing the final-timestep density distribution
of the training split to the full dataset.  Low KS = training density
distribution matches the full dataset.

### Composite score

Lower is better.  Weighted rank sum:

```
score = 3.0 × near_duplicate_rate_rank
      + 1.5 × coverage_distance_rank
      + 1.5 × density_ks_rank
      + 1.0 × train_nearest_similarity_rank
      + 0.25 × seconds_rank
```

---

## Running the Pipeline

Run the workflow in two separate steps:

1. `run_samplers.py` flattens the raw HDF5 if needed, instantiates the samplers,
   and saves split CSVs.
2. `run_sampler_benchmark.py` runs the similarity benchmark, selects the
   best-ranked sampler, and copies that split dataset into `best_sampler/`.

### Quick smoke-test (500 tracers, ~1 min)

```bash
python run_samplers.py --n-samples 500
python run_sampler_benchmark.py --n-samples 500
```

### Full run (6000 tracers)

```bash
python run_samplers.py --n-samples 6000
python run_sampler_benchmark.py --n-samples 6000
```

### Override paths without editing code

```bash
export SAMPLERS_RAW_H5=/path/to/grav_collapse_postprocessed_uclchem.h5
export SAMPLERS_DATA_DIR=/path/to/sampled_dataset
export SAMPLERS_RESULTS_DIR=/path/to/results
python run_samplers.py --n-samples 500
python run_sampler_benchmark.py --n-samples 500
```

### Sampler CLI

```
usage: run_samplers.py [-h]
  [--raw-h5 RAW_H5]
  [--data-dir DATA_DIR]
  [--n-samples N_SAMPLES]
  [--max-similarity-tracers MAX_SIMILARITY_TRACERS]
  [--samplers {random,density,qr_pivot,svd_fps,similarity_constrained} ...]
  [--force-flatten]
  [--force-sample]
```

### Benchmark CLI

```
usage: run_sampler_benchmark.py [-h]
  [--data-dir DATA_DIR]
  [--results-dir RESULTS_DIR]
  [--best-sampler-dir BEST_SAMPLER_DIR]
  [--n-samples N_SAMPLES]
  [--random-state RANDOM_STATE]
  [--samplers {random,density,qr_pivot,qr,svd_fps,similarity_constrained} ...]
  [--thresholds THRESHOLDS ...]
  [--primary-threshold PRIMARY_THRESHOLD]
  [--max-reference MAX_REFERENCE]
  [--max-candidate MAX_CANDIDATE]
  [--max-pairwise MAX_PAIRWISE]
  [--force-benchmark]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--raw-h5` | `SAMPLERS_RAW_H5` env var | Path to raw HDF5 |
| `--data-dir` | `SAMPLERS_DATA_DIR` env var | Split CSV output directory |
| `--results-dir` | `SAMPLERS_RESULTS_DIR` env var | Benchmark output directory |
| `--best-sampler-dir` | `DATA_DIR/best_sampler` | Directory receiving the best split dataset |
| `--n-samples` | `6000` | Tracers per sampler |
| `--max-similarity-tracers` | `500` | Max tracers used for saved per-split similarity plots |
| `--force-flatten` | off | Re-flatten even if bundle exists |
| `--force-sample` | off | Re-run samplers even if splits exist |
| `--force-benchmark` | off | Re-run benchmark even if results exist |

### Outputs

```text
$SAMPLERS_DATA_DIR/
  flattened_dataset.h5              ← flattened vector bundle
  random/train.csv, val.csv, test.csv
  density/train.csv, val.csv, test.csv
  qr_pivot/train.csv, val.csv, test.csv
  qr_pivot/qr_samples/R.npy, qr_indices.npy
  svd_fps/train.csv, val.csv, test.csv
  best_sampler/train.csv, val.csv, test.csv
  best_sampler/best_sampler.txt      ← selected sampler name

$SAMPLERS_RESULTS_DIR/
  sampler_benchmark/
    sampler_benchmark_results.json  ← full metrics for all samplers
    sampler_ranking.csv             ← ranked summary table
```

---

## TACC HPC Submission

### Using `sample_datasets.slurm`

```bash
# Generate sampled datasets only.
sbatch --export=ALL,\
  SAMPLERS_DATA_DIR=/scratch/$USER/sampled_dataset,\
  N_SAMPLES=500 \
  sample_datasets.slurm
```

### Using `benchmark_samplers.slurm`

Run this after `sample_datasets.slurm` has produced the sampler split
directories:

```bash
sbatch --export=ALL,\
  SAMPLERS_DATA_DIR=/scratch/$USER/sampled_dataset,\
  SAMPLERS_RESULTS_DIR=/scratch/$USER/results,\
  N_SAMPLES=500 \
  benchmark_samplers.slurm
```

The benchmark job writes the selected split to:

```text
$SAMPLERS_DATA_DIR/best_sampler/
  train.csv
  val.csv
  test.csv
  best_sampler.txt
```

For the full local pipeline on an interactive node, use the shell wrapper:

```bash
export SAMPLERS_RAW_H5=/scratch/$USER/data/grav_collapse_postprocessed_uclchem.h5
export SAMPLERS_DATA_DIR=/scratch/$USER/sampled_dataset
export SAMPLERS_RESULTS_DIR=/scratch/$USER/results
./run_benchmark.sh
```

---

## Step-by-Step Manual Workflow

If you prefer to run each step individually:

### Step 0: Flatten the Raw HDF5

```bash
python flatten_dataset.py \
  --input-h5  /path/to/grav_collapse_postprocessed_uclchem.h5 \
  --output-path /path/to/sampled_dataset/flattened_dataset.h5
```

Or with env vars:

```bash
export SAMPLERS_RAW_H5=/path/to/grav_collapse_postprocessed_uclchem.h5
export SAMPLERS_DATA_DIR=/path/to/sampled_dataset
python flatten_dataset.py
```

### Step 1: Generate Candidate Split Datasets

```bash
python run_samplers.py \
  --n-samples 500
```

### Step 2: Benchmark Split Similarity

```bash
python run_sampler_benchmark.py \
  --n-samples 500
```

Or run both steps with the wrapper:

```bash
./run_benchmark.sh
```

---

## Flattened Bundle Format

`util.save_bundle()` writes an `.npz` file (even when the conventional
filename ends in `.h5`).  `util.load_bundle()` handles both extensions.

Bundle keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `vectors` | `(n_tracers, T × n_features)` | Flattened trajectory vectors |
| `tracer_ids` | `(n_tracers,)` | Tracer IDs aligned with `vectors` rows |
| `feature_cols` | `(n_features,)` | Feature order within each timestep |
| `log_cols` | `(n_abundance_cols,)` | Abundance columns stored in log10 space |
| `time_grid` | `(T,)` | Original time values |
| `T` | scalar | Number of timesteps per tracer |

Physical columns (`Density`, `gasTemp`, `dustTemp`, `Av`, `radfield`, `zeta`)
remain linear.  All other feature columns are stored as `log10(max(v, 1e-30))`.
