# xLSTM Surrogate (`xlstm`)

xLSTM (extended LSTM) recurrent surrogate variant (Beck et al. 2024). Owns
hyperparameter optimization, final training, and autoregressive rollout testing.
Keeps the shared `[physical params, log10 abundances]_t -> log10 abundances_{t+1}`
interface, rollout curriculum, trace-species downweighting, and physical-input
normalization used by the other surrogates; only the recurrent core differs.

The xLSTM cells are a dependency-free pure-PyTorch reimplementation in
`src/xlstm_cells.py` (recurrent step form, so the hidden state threads across a
trajectory exactly like the LSTM/GRU `rnn` variant). No `xlstm` pip package is
required, and it runs on CPU / MPS / CUDA.

## Configuration

Path-specific settings and pipeline arguments live in the shared repo config
at `lightning_surrogates/config.sh`. SLURM scripts source it directly
(override: `LS_CONFIG`); Python entry points read the resulting environment
variables through `src/settings.py`. For local runs, `source config.sh` first.

xLSTM-specific knobs (defaults in parentheses):

| Env var | Meaning |
| --- | --- |
| `MODEL_XLSTM_CELL_TYPE` | block kind: `slstm` \| `mlstm` \| `mixed` (`slstm`) |
| `MODEL_XLSTM_NUM_BLOCKS` | number of stacked xLSTM blocks (`2`) |
| `MODEL_XLSTM_HIDDEN_DIM` | hidden width; must be divisible by num heads (`256`) |
| `MODEL_XLSTM_NUM_HEADS` | per-cell head count (`4`) |
| `MODEL_XLSTM_DROPOUT` | residual sublayer dropout (`0.0`) |

`mixed` alternates sLSTM and mLSTM blocks (sLSTM first). Optuna tunes
`xlstm_num_blocks` and `xlstm_hidden_dim` (plus learning rate / batch size);
`cell_type` and `num_heads` are fixed per study so a run isolates one variant.

## Usage

Local (after `source config.sh`):

```bash
cd models/xlstm/src
python optimize.py "$LS_DATA_DIR" --num-trials 25 --tune-epochs 50
python train.py    "$LS_DATA_DIR" --config-file ../results/<dataset>/<sampler>/optimization/best_params.json
python test.py     "$LS_DATA_DIR"
```

SLURM (submit from `models/xlstm`):

```bash
sbatch slurm/optimize.slurm
sbatch slurm/train.slurm
sbatch slurm/test.slurm
sbatch slurm/pipeline.slurm   # sampling -> optimize -> train -> test
```

Unit tests (fast, no data):

```bash
cd models/xlstm/src && python -m pytest ../tests/test_xlstm.py -q
```

## Layout

| Dir | Purpose |
| --- | --- |
| `src/` | model, xLSTM cells, data, training, optimization, testing code |
| `tests/` | unit / integration tests |
| `slurm/` | cluster job scripts |
| `logs/` | run logs |
