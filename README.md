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

The default dataset is the GOW17 chemistry network
(`DATASET_NAME=gow17_R0.05_M6.0`); the MLP and the
[PINN variant](#pinn-physics-informed-variant) train on the same sampled
splits so their results are directly comparable.

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
checkpoint + loss curves        models/mlp/results/{dataset}/{sampler}/
   │  test.py                   autoregressive rollout on test tracers
   ▼
error summaries + rollout plots models/mlp/results/{dataset}/{sampler}/test_results/
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
$[\tilde p_t, a_t]$ and 333 outputs $a_{t+1}$, trained on the log abundances
with a robust per-element loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{(t,i)} \big\| f_\theta([\tilde p_t^i, a_t^i]) - a_{t+1}^i \big\|$$

The default loss is **smooth-L1 (Huber)** (`MODEL_LOSS_FUNCTION`; `l1`, `mse`,
and `smooth_l1` are the options). L1/Huber in $\log_{10}$ space is the mean
absolute error **in dex**, weighting a factor-of-ten error equally for abundant
and trace species; Huber's vanishing gradient near the optimum settles the
weights instead of dithering (smoother val curve than plain L1). Optimizer:
AdamW with gradient clipping and a cosine learning-rate schedule
(`MODEL_LR_SCHEDULER`); early stopping when validation loss stops improving by a
relative threshold.

**5. Hyperparameter search.** A sequential Optuna study (TPE sampler, median
pruner) searches hidden layers (2–8), hidden units (128–1024, step 128),
learning rate ($10^{-5}$–$10^{-2}$, log scale), and batch size
(256/512/1024/2048). The loss function is **not** tuned — it is fixed to
`MODEL_LOSS_FUNCTION` so trials are comparable; the objective is validation
MSE (a fixed metric independent of the chosen loss). The journal is a SQLite
file. By default `JOURNAL_MODE=fresh` wipes it each run so config changes take
effect; set `JOURNAL_MODE=resume` to continue an interrupted study.

**6. Evaluation.** `test.py` runs the *autoregressive* rollout — the regime
that matters in production, where errors compound:

$$\hat a_0 = a_0, \qquad \hat a_{t+1} = f_\theta([\tilde p_t, \hat a_t])$$

using the true physical history $p_t$ and only the initial abundances. It
reports per-tracer and per-species MSE in $\log_{10}$ space and plots
best/worst rollouts for a panel of key species.

### Implementation notes

- **Multi-step rollout training.** Beyond the one-step loss, training unrolls
  the model over short windows of consecutive steps and weights each step-$j$
  error by $0.5^{\,j}$, with a curriculum that grows the horizon
  ($1\to2\to4\dots$) as training proceeds — so the network is penalized for the
  error compounding it will face at rollout, not just single-step accuracy.
- **Trace-species handling.** Abundances are clipped at `MODEL_ABUND_FLOOR`
  ($10^{-25}$) before the $\log_{10}$ transform, and targets at or below
  `MODEL_TRACE_THRESHOLD` ($10^{-20}$) are downweighted by `MODEL_TRACE_WEIGHT`
  ($0.1$) so unresolved trace species stop dominating the dex loss (set the
  weight to 1 to disable).
- **Optuna journal.** `JOURNAL_MODE=fresh` (default) re-runs all trials so
  search-space/config edits take effect; `resume` continues an interrupted
  study.

## File structure

```
lightning_surrogates/
├── config.sh                  # single config: all paths + args for every stage
├── agent-orchestrator.yaml    # dev agent rules + runtime settings
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
│   ├── slurm/                 # common.sh + optimize/train/test/pipeline jobs
│   └── results/               # {dataset}/{sampler}/ experiment outputs (gitignored)
├── models/PINN/               # physics-informed variant (see its own README.md)
│   ├── README.md              # PINN model, loss, RHS, zeta calibration
│   ├── src/                   # settings/data/model/optimize/train/test
│   │   ├── gow17_network.py   # parse network/*.dat → ODE listing
│   │   ├── gow17_rates.py     # differentiable torch RHS (50 reactions)
│   │   └── validate_rhs.py    # RHS vs finite-diff check, zeta calibration
│   ├── network/{species,reactions}.dat  # GOW17 network (54 species, 50 rxns)
│   ├── GOW17_ODES.txt         # full ODE listing (reference)
│   ├── slurm/                 # optimize/train/test/pipeline jobs
│   └── results/               # {dataset}/{sampler}/pinn/ (at repo-root results/)

../datasets/                                    # sibling of this repo
├── mbon_impl/GOW2017_network/*.h5              # raw postprocessed chemistry
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

# 2. Hyperparameter search  →  models/mlp/results/{dataset}/{sampler}/optimization/
cd ../models/mlp/src
python optimize.py

# 3. Train with the best parameters  →  models/mlp/results/{dataset}/{sampler}/
python train.py --config-file ../results/gow17_R0.05_M6.0/density/optimization/best_params.json
#    (or quick run with config defaults: python train.py --use-defaults)

# 4. Rollout evaluation  →  models/mlp/results/{dataset}/{sampler}/test_results/
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
                        sbatch slurm/pipeline.slurm    # sampling + all three in sequence

# Full pipeline on a different raw .h5 (no config edits):
sbatch --export=ALL,DATASET_NAME=my_dataset,SAMPLERS_RAW_H5=/path/to/file.h5 slurm/pipeline.slurm
```

The SLURM scripts source `config.sh` through `slurm/common.sh`; job logs go
to `logs/` in the submit directory.

## PINN: physics-informed variant

`models/PINN/` is a second surrogate for the same GOW17 data, sharing the
sampler splits, normalization, and Optuna/SLURM machinery as the MLP but adding
two things: **elapsed time $\Delta t$ is a model input**, and the **chemical ODE
system enters the loss as a physics residual**. Its own
[`models/PINN/README.md`](models/PINN/README.md) carries the full derivation,
the conservation laws, and the RHS-validation findings; this is the summary.

**Model.** A hard initial-condition ansatz makes $\hat a(0)=a_0$ exact:

$$\hat a(\Delta t) = a_0 + \tfrac{\Delta t}{t_{\text{ref}}}\,
g_\theta\big([\tilde p,\; a_0,\; \tfrac{\Delta t}{t_{\text{ref}}}]\big)$$

Training pairs span $1..$`PINN_MAX_HORIZON` (default 4) consecutive snapshots, so
the network sees $\Delta t \in \{253, 506, 759, 1012\}$ yr — making time a real
input rather than a fixed step ($t_{\text{ref}}=$ `PINN_DT_REF_YEARS`).

**Loss.** Data L1 (dex error, as in the MLP), plus a physics term and a
conservation term:

$$\mathcal{L} = \mathcal{L}_{\text{data}}
+ w_{\text{phys}}\,\mathbb{E}\,|r|
+ w_{\text{cons}}\,\mathbb{E}\,|\text{invariant drift}|$$

with `PINN_PHYSICS_WEIGHT` ($0.1$) and `PINN_CONSERVATION_WEIGHT` ($0.01$). The
residual $r$ compares the autograd time-derivative of the prediction (double-vjp,
one collocation $\tau$ per sample, random in $(0,\Delta t]$ by default) against
the GOW17 RHS, symmetrically normalized by production + destruction so
$|r|\le 1$ per species — necessary because the network is stiff. Conservation is
measured on 6 linear invariants (charge + elemental H/He/C/O/Si). The objective
for Optuna is still `val_mse` (loss-weight-independent), and the search space
additionally tunes `physics_weight` and `conservation_weight`.

**Chemistry RHS.** `gow17_network.py` parses
`network/{species,reactions}.dat` (54 species, 50 reactions) into stoichiometry;
`gow17_rates.py` is the differentiable torch RHS (two-body, cosmic-ray, photo,
grain-assisted), with `R @ νᵀ` conserving the invariants to machine precision.
`validate_rhs.py` checks the RHS against finite-difference $da/dt$ from the data
and calibrates the cosmic-ray ionization unit from the H₂⁺ quasi-steady-state
balance ($\zeta_{\text{phys}} = 2.024\times10^{-17}\,\text{s}^{-1}$;
`PINN_ZETA_UNIT` default $1.657\times10^{-17}$).

**Run it** (same CLI shape as the MLP; results go to
`results/{dataset}/{sampler}/pinn/` at the repo root, *not* under
`models/PINN/`):

```bash
source config.sh
cd models/PINN/src
python validate_rhs.py        # optional: RHS / zeta sanity check vs the data
python optimize.py            # → optimization/best_params.json
python train.py --config-file <best_params.json>
python test.py --rollout-stride 1   # k-snapshot jumps exercise the Δt input
```

SLURM mirrors the MLP: `models/PINN/slurm/{optimize,train,test,pipeline}.slurm`.
Note the PINN Optuna study defaults to `JOURNAL_MODE=resume` (the MLP defaults to
`fresh`).
