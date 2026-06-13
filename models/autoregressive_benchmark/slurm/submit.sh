#!/bin/bash
# Chain the autoregressive benchmark: random sampling first, then the
# train+test job array gated on it with --dependency=afterok.
#
#   cd models/autoregressive_benchmark && ./slurm/submit.sh
#
# Override the dataset / raw H5 at submit time (passed to both jobs):
#   ./slurm/submit.sh --export=ALL,DATASET_NAME=grav_collapse,SAMPLERS_RAW_H5=/path/to/file.h5
set -euo pipefail
cd "$(dirname "$0")"

jid=$(sbatch --parsable "$@" sample.slurm)
echo "sampling job: ${jid}"

bench_jid=$(sbatch --parsable --dependency=afterok:"${jid}" "$@" benchmark.slurm)
echo "benchmark array: ${bench_jid} (starts after ${jid} succeeds)"
