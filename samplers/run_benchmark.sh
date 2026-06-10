#!/bin/bash
# Sampler pipeline wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Set default arguments or allow override via env variables
RAW_H5="${SAMPLERS_RAW_H5:-${SCRIPT_DIR}/../grav_collapse/baseline/grav_collapse_postprocessed_chemistry_uclchem.h5}"
DATA_DIR="${SAMPLERS_DATA_DIR:-${SCRIPT_DIR}/../sampled_datasets}"
RESULTS_DIR="${SAMPLERS_RESULTS_DIR:-${SCRIPT_DIR}/results}"
N_SAMPLES="${N_SAMPLES:-6000}"
MAX_SIMILARITY_TRACERS="${MAX_SIMILARITY_TRACERS:-500}"

SAMPLER_ARGS=()
SAMPLER_ARGS+=(--raw-h5 "${RAW_H5}")
SAMPLER_ARGS+=(--data-dir "${DATA_DIR}")
SAMPLER_ARGS+=(--n-samples "${N_SAMPLES}")
SAMPLER_ARGS+=(--max-similarity-tracers "${MAX_SIMILARITY_TRACERS}")

BENCHMARK_ARGS=()
BENCHMARK_ARGS+=(--data-dir "${DATA_DIR}")
BENCHMARK_ARGS+=(--results-dir "${RESULTS_DIR}")
BENCHMARK_ARGS+=(--n-samples "${N_SAMPLES}")

if [ "${FORCE_FLATTEN:-0}" -eq 1 ]; then
    SAMPLER_ARGS+=(--force-flatten)
fi

if [ "${FORCE_SAMPLE:-0}" -eq 1 ]; then
    SAMPLER_ARGS+=(--force-sample)
fi

if [ "${FORCE_BENCHMARK:-0}" -eq 1 ]; then
    BENCHMARK_ARGS+=(--force-benchmark)
fi

echo "========================================================="
echo "Executing sampler pipeline wrapper script..."
echo "========================================================="
python run_samplers.py "${SAMPLER_ARGS[@]}"
python run_sampler_benchmark.py "${BENCHMARK_ARGS[@]}"
