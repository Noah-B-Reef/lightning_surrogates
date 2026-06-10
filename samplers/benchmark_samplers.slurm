#!/bin/bash
#SBATCH --job-name=gc_benchmark
#SBATCH --output=output/benchmark_output_%j.txt
#SBATCH --error=error/benchmark_error_%j.txt
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nreef@utexas.edu

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${SCRIPT_DIR}"

# Allow overrides at submission time, for example:
#   sbatch --export=ALL,SAMPLERS_DATA_DIR=/path/to/sampled_datasets,SAMPLERS_RESULTS_DIR=/path/to/results,N_SAMPLES=6000 benchmark_samplers.slurm
DATA_DIR="${SAMPLERS_DATA_DIR:-${DATA_DIR:-${SCRIPT_DIR}/../sampled_datasets}}"
RESULTS_DIR="${SAMPLERS_RESULTS_DIR:-${RESULTS_DIR:-${SCRIPT_DIR}/results}}"
BEST_SAMPLER_DIR="${BEST_SAMPLER_DIR:-${DATA_DIR}/best_sampler}"
N_SAMPLES="${N_SAMPLES:-6000}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SAMPLER_NAMES="${SAMPLER_NAMES:-random density qr_pivot svd_fps}"
THRESHOLDS="${THRESHOLDS:-0.95 0.99 0.999 0.9999}"
PRIMARY_THRESHOLD="${PRIMARY_THRESHOLD:-0.9999}"
MAX_REFERENCE="${MAX_REFERENCE:-500}"
MAX_CANDIDATE="${MAX_CANDIDATE:-1000}"
MAX_PAIRWISE="${MAX_PAIRWISE:-500}"

mkdir -p output error "${RESULTS_DIR}"

# Load necessary modules or activate an environment if needed.
# module load module_name
# conda activate sampler_env

ARGS=()
ARGS+=(--data-dir "${DATA_DIR}")
ARGS+=(--results-dir "${RESULTS_DIR}")
ARGS+=(--best-sampler-dir "${BEST_SAMPLER_DIR}")
ARGS+=(--n-samples "${N_SAMPLES}")
ARGS+=(--random-state "${RANDOM_STATE}")
ARGS+=(--samplers ${SAMPLER_NAMES})
ARGS+=(--thresholds ${THRESHOLDS})
ARGS+=(--primary-threshold "${PRIMARY_THRESHOLD}")
ARGS+=(--max-reference "${MAX_REFERENCE}")
ARGS+=(--max-candidate "${MAX_CANDIDATE}")
ARGS+=(--max-pairwise "${MAX_PAIRWISE}")

if [ "${FORCE_BENCHMARK:-0}" -eq 1 ]; then
    ARGS+=(--force-benchmark)
fi

python run_sampler_benchmark.py "${ARGS[@]}"
