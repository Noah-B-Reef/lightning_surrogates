#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=arb_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

# Stage 0 of the autoregressive benchmark: produce/validate a RANDOM sampling of
# the dataset split. This must finish before the train/test array runs;
# submit.sh chains the array with --dependency=afterok on this job.
#
# All paths and arguments come from the repo config (config.sh), loaded by
# common.sh, which forces SAMPLING_PROCEDURE=random. Submit from
# models/autoregressive_benchmark: sbatch slurm/sample.slurm
#
# Locate common.sh whether submitted from models/autoregressive_benchmark/ or
# from its slurm/ dir. sbatch copies this script to the slurmd spool dir, so its
# own path (BASH_SOURCE) is only usable when run outside SLURM.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/common.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/common.sh"
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
mkdir -p logs
print_config_summary
activate_conda

echo "=== sampling (${SAMPLING_PROCEDURE}, ${STORAGE_FORMAT}) ==="
# run_samplers.py flattens the raw H5 if needed and skips sampling when the
# split already exists.
python "${REPO_DIR}/samplers/run_samplers.py" \
    --raw-h5 "${SAMPLERS_RAW_H5}" \
    --data-dir "${SAMPLERS_DATA_DIR}" \
    --n-samples "${N_SAMPLES}" \
    --max-similarity-tracers "${MAX_SIMILARITY_TRACERS}" \
    --storage-format "${STORAGE_FORMAT}" \
    --samplers "${SAMPLING_PROCEDURE}"

validate_split_dir
echo "=== sampling complete: split ready at ${LS_DATA_DIR} ==="
