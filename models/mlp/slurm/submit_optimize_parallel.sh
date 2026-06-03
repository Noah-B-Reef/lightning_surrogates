#!/bin/bash

set -Eeuo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLP_DIR="$(cd "${SLURM_DIR}/.." && pwd)"
MLP_SLURM_CONFIG="${MLP_SLURM_CONFIG:-${MLP_CONFIG:-${MLP_DIR}/config.sh}}"

if [ ! -f "${MLP_SLURM_CONFIG}" ]; then
    echo "Missing Slurm config: ${MLP_SLURM_CONFIG}" >&2
    exit 2
fi

source "${MLP_SLURM_CONFIG}"

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR must be set in ${MLP_SLURM_CONFIG}}"
OPTUNA_PARALLEL_RESULTS_DIR="${OPTUNA_PARALLEL_RESULTS_DIR:-${RESULTS_DIR}/optimization_parallel}"

if [ -z "${PARALLEL_OPTUNA_STORAGE:-}" ]; then
    PARALLEL_OPTUNA_STORAGE="sqlite:///${OPTUNA_PARALLEL_RESULTS_DIR}/optuna.sqlite3"
    echo "PARALLEL_OPTUNA_STORAGE is unset; defaulting to ${PARALLEL_OPTUNA_STORAGE}"
    echo "WARNING: SQLite is not recommended for multi-node Optuna workers."
fi

mkdir -p \
    "${RESULTS_DIR}/output" \
    "${RESULTS_DIR}/error" \
    "${OPTUNA_PARALLEL_RESULTS_DIR}"

echo "Submitting optimize_parallel.slurm"
echo "MLP_SLURM_CONFIG=${MLP_SLURM_CONFIG}"
echo "PARALLEL_OPTUNA_STORAGE=${PARALLEL_OPTUNA_STORAGE}"
echo "Output: ${RESULTS_DIR}/output/optimize_parallel_<jobid>.out"
echo "Error:  ${RESULTS_DIR}/error/optimize_parallel_<jobid>.err"

sbatch --export=ALL,MLP_SLURM_CONFIG="${MLP_SLURM_CONFIG}" "${SLURM_DIR}/optimize_parallel.slurm"
