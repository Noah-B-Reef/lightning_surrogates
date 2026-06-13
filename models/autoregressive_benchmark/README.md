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
