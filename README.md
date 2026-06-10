# lightning_surrogates

Machine-learned surrogate models for the chemical evolution of tracer
particles in a gravitational-collapse simulation post-processed with UCLCHEM.
The repository owns the full experiment pipeline: dataset flattening and
sampling, hyperparameter optimization, training, and autoregressive rollout
testing, locally or on TACC via SLURM.

## Problem formulation

The raw dataset follows ~10,000 tracer particles through a gravitational
collapse. Each tracer $i$ has a trajectory of $T = 296$ timesteps; at each
step we know:

- **Physical parameters** $p_t \in \mathbb{R}^4$: density, gas temperature,
  visual extinction $A_v$, and radiation field (dust temperature and cosmic
  ray ionization rate $\zeta$ are dropped — duplicate and constant,
  respectively).
- **Chemical abundances** $n_t \in \mathbb{R}^{333}$: UCLCHEM species
  abundances spanning many orders of magnitude.

Solving the chemistry with UCLCHEM is the expensive step in large
simulations. The surrogate replaces it with a learned **one-step map** in log
abundance space. With $a_t = \log_{10} \max(n_t, 10^{-30})$:

$$a_{t+1} \approx f_\theta\big(\,[\tilde p_t,\; a_t]\,\big)$$

where $f_\theta$ is a plain MLP and $\tilde p_t$ is the normalized physical
state. Applied recursively from an initial condition, the map reproduces a
full chemical trajectory given only the physical history.

## Pipeline summary

```
raw HDF5 (UCLCHEM postprocessed)
   │  flatten_dataset.py        one vector per tracer trajectory
   ▼
flattened bundle (npz)
   │  main.py / run_samplers.py select ~6000 tracers, split 60/20/20
   ▼
sampled splits   datasets/sampled_datasets/{dataset}/{sampler}/{csv|npy}/
   │  optimize.py               sequential Optuna search (val L1 loss)
   ▼
best_params.json
   │  train.py                  final model, early stopping on val loss
   ▼
checkpoint + loss curves        results/{dataset}/{sampler}/mlp/
   │  test.py                   autoregressive rollout on test tracers
   ▼
error summaries + rollout plots results/{dataset}/{sampler}/mlp/test_results/
```

### The math, step by step

**1. Flattening.** Each tracer trajectory is reshaped into a single vector
$x_i \in \mathbb{R}^{T \cdot F}$ (timesteps × features, abundances already in
$\log_{10}$). This gives a matrix of trajectories that the samplers operate
on.

**2. Sampling.** Training on all tracers is redundant — many trajectories are
nearly identical. Each sampler selects $N = 6000$ tracers, split 60/20/20
into train/val/test **by tracer**, so no trajectory leaks across splits:

- `random` — uniform sample of tracer IDs (baseline).
- `density` — stratified sampling over $\log_{10}$ final density bins with
  proportional allocation, guaranteeing coverage of the collapse range.
- `qr_pivot` — column-pivoted QR factorization $A P = Q R$ of the matrix of
  candidate trajectory vectors ($A$: features × tracers). The first $k$
  pivots greedily select the most linearly independent trajectories — an
  interpolative basis for the dataset.
- `svd_fps` — randomized SVD to a low-dimensional embedding
  $U \Sigma$, then farthest-point sampling: iteratively pick the trajectory
  maximizing the minimum distance to those already selected (coverage in
  trajectory space).
- `similarity_constrained` — k-means clusters in the SVD embedding are split
  between train and holdout; holdout tracers whose centered cosine
  similarity to any training trajectory exceeds a threshold are filtered
  out, bounding train/test leakage.

`benchmark_samplers.py` scores the resulting splits (cosine-similarity
leakage, KS statistics on the density distribution) to compare samplers.

**3. Input normalization.** Abundances enter the network as
$\log_{10}$ values. The physical columns are normalized inside the model
(stats stored in the checkpoint so rollout uses identical transforms):
multi-decade positive columns (density, $A_v$, radfield) are
$\log_{10}$-transformed after a per-column floor, then all physical columns
are standardized with training-split statistics:

$$\tilde p = \frac{\phi(p) - \mu_{\text{train}}}{\sigma_{\text{train}}}, \qquad
\phi = \log_{10} \circ \max(\cdot, \text{floor}) \text{ on masked columns.}$$

**4. Model and loss.** $f_\theta$ is a feed-forward MLP (Linear + ReLU
stacks; depth/width set by Optuna) with 337 inputs
$[\tilde p_t, a_t]$ and 333 outputs $a_{t+1}$, trained with **L1 loss on the
log abundances**:

$$\mathcal{L}(\theta) = \mathbb{E}_{(t,i)} \big\| f_\theta([\tilde p_t^i, a_t^i]) - a_{t+1}^i \big\|_1$$

L1 in $\log_{10}$ space means the loss is the mean absolute error **in dex**,
weighting a factor-of-ten error equally for abundant and trace species.
Optimizer: AdamW with gradient clipping; early stopping when validation loss
stops improving by a relative threshold.

**5. Hyperparameter search.** A sequential Optuna study (TPE sampler, median
pruner) minimizes validation L1 loss over hidden layers (2–5), hidden units
(128–512), learning rate ($10^{-4}$–$5\times10^{-3}$, log scale), and batch
size (32/64/128). The journal is a SQLite file, so interrupted studies
resume.

**6. Evaluation.** `test.py` runs the *autoregressive* rollout — the regime
that matters in production, where errors compound:

$$\hat a_0 = a_0, \qquad \hat a_{t+1} = f_\theta([\tilde p_t, \hat a_t])$$

using the true physical history $p_t$ and only the initial abundances. It
reports per-tracer and per-species MSE in $\log_{10}$ space and plots
best/worst rollouts for a panel of key species.

## File structure

```
lightning_surrogates/
├── config.sh                  # single config: all paths + args for every stage
├── samplers/
│   ├── flatten_dataset.py     # raw HDF5 → flattened trajectory bundle
│   ├── samplers.py            # random / density / qr_pivot / svd_fps / similarity
│   ├── util.py                # flatten, reformat, split-saving (csv | npy)
│   ├── main.py                # run selected samplers on a bundle
│   ├── run_samplers.py        # flatten-if-needed + sample (pipeline entry)
│   ├── benchmark_samplers.py  # split similarity / leakage benchmark
│   ├── run_sampler_benchmark.py / run_benchmark.sh
│   ├── path_utils.py          # env-var-aware default paths
│   └── *.slurm                # TACC jobs (source ../config.sh)
├── models/mlp/
│   ├── src/
│   │   ├── settings.py        # typed env-var defaults; results paths
│   │   ├── data.py            # split loading (csv|npy), one-step pair dataset
│   │   ├── model.py           # MLP LightningModule, L1 loss, phys normalization
│   │   ├── optimize.py        # sequential Optuna study
│   │   ├── train.py           # final training run
│   │   ├── test.py            # autoregressive rollout evaluation
│   │   └── callbacks.py       # epoch printer, relative-improvement early stop
│   └── slurm/                 # common.sh + optimize/train/test/run jobs
└── results/                   # results/{dataset}/{sampler}/{architecture}/ (gitignored)

../datasets/                                    # sibling of this repo
├── grav_collapse/baseline/*.h5                 # raw postprocessed chemistry
└── sampled_datasets/{dataset}/{sampler}/{fmt}/ # train/val/test splits
```

Split storage formats: `csv` (one file per split) or `npy`
(float64 `train/val/test.npy` in long format + `columns.json` with the column
order). The data loader accepts either.

## How to run

### Configure once

All paths and stage arguments live in [`config.sh`](config.sh) — a plain
shell file of `export KEY=value` lines. SLURM scripts source it directly;
the Python scripts read the same variables from the environment, so for
local runs either `source config.sh` first or rely on the built-in defaults,
which derive from the repository location (raw data in the sibling
`datasets/` directory) and match the standard layout. Key knobs:
`DATASET_NAME`, `SAMPLING_PROCEDURE`, `STORAGE_FORMAT`, `MODEL_*`,
`N_TRIALS`, `TRAIN_EPOCHS`. Point `LS_CONFIG` at an alternative file to
switch configurations without editing the repo.

### Local

```bash
# 1. Flatten (if needed) and generate splits, e.g. density sampler, npy format
cd samplers
python run_samplers.py --samplers density --storage-format npy

# 2. Hyperparameter search  →  results/{dataset}/{sampler}/mlp/optimization/
cd ../models/mlp/src
python optimize.py

# 3. Train with the best parameters  →  results/{dataset}/{sampler}/mlp/
python train.py --config-file ../../../results/grav_collapse/density/mlp/optimization/best_params.json
#    (or quick run with config defaults: python train.py --use-defaults)

# 4. Rollout evaluation  →  results/{dataset}/{sampler}/mlp/test_results/
python test.py
```

Every script also accepts an explicit split directory
(`python train.py /path/to/split`) and `--results-dir` to override the
derived experiment directory.

### TACC / SLURM

```bash
cd samplers          && sbatch sample_datasets.slurm   # generate splits
cd models/mlp        && sbatch slurm/optimize.slurm    # Optuna search
                        sbatch slurm/train.slurm       # final training
                        sbatch slurm/test.slurm        # rollout evaluation
                        sbatch slurm/run.slurm         # all three in sequence
```

The SLURM scripts source `config.sh` through `slurm/common.sh`; job logs go
to `logs/` in the submit directory.
