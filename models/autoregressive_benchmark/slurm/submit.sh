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

# Vista's sbatch wrapper prints a multi-line welcome/verification banner to
# STDOUT alongside the job id. Extract only the numeric SLURM job id: match the
# leading number on each line (--parsable may emit 'jobid;cluster') and take the
# last such match, which is the actual id. A polluted id would produce a
# malformed --dependency and the 'Job dependency problem' error.
jid=$(sbatch --parsable "$@" sample.slurm | grep -oE '^[0-9]+' | tail -1)
if [[ -z "${jid}" ]]; then
  echo "error: failed to parse sampling job id from sbatch output" >&2
  exit 1
fi
echo "sampling job: ${jid}"

bench_jid=$(sbatch --parsable --dependency=afterok:"${jid}" "$@" benchmark.slurm | grep -oE '^[0-9]+' | tail -1)
if [[ -z "${bench_jid}" ]]; then
  echo "error: failed to parse benchmark array job id from sbatch output" >&2
  exit 1
fi
echo "benchmark array: ${bench_jid} (starts after ${jid} succeeds)"
