# RNN Surrogate (`rnn`)

Vanilla RNN recurrent surrogate variant. Owns hyperparameter optimization,
final training, and autoregressive rollout testing.

## Configuration

Path-specific settings and pipeline arguments live in the shared repo config
at `lightning_surrogates/config.sh`. SLURM scripts source it directly
(override: `LS_CONFIG`); Python entry points read the resulting environment
variables through `src/settings.py`. For local runs, `source config.sh` first.

## Layout

| Dir | Purpose |
| --- | --- |
| `src/` | model, data, training, optimization code |
| `tests/` | unit / integration tests |
| `slurm/` | cluster job scripts |
| `logs/` | run logs |
