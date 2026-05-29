# Lightning MLP Surrogate

This directory contains a PyTorch Lightning MLP for learning chemical evolution
from sampled GravCollapse tracer split CSVs.

Expected split directory:

```text
train.csv
val.csv
test.csv
```

Train:

```bash
python src/train.py --data-dir /path/to/split --results-dir /path/to/results
```

Optimize:

```bash
python src/optimize.py --data-dir /path/to/split --results-dir /path/to/optuna
```

Test:

```bash
python src/test.py \
  --model_checkpoint /path/to/results/mlp_grav_collapse.ckpt \
  --test_dir /path/to/split/test.csv \
  --output_dir /path/to/results/test_results
```

SLURM:

```bash
sbatch run.slurm
```

Optional overrides:

```bash
sbatch --export=ALL,DATA_DIR=/path/to/split,RESULTS_DIR=/path/to/results run.slurm
```
