#!/bin/bash
#SBATCH --job-name=gc_sample
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=nreef@utexas.edu

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${SCRIPT_DIR}"

# All paths and sampling arguments come from the repo config (config.sh).
# Override the config location with LS_CONFIG.
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LS_CONFIG="${LS_CONFIG:-${REPO_DIR}/config.sh}"
if [ ! -f "${LS_CONFIG}" ]; then
    echo "Missing config: ${LS_CONFIG}" >&2
    exit 2
fi
source "${LS_CONFIG}"

mkdir -p logs "${SAMPLERS_DATA_DIR}"

# Load necessary modules or activate an environment if needed.
# module load module_name
# conda activate sampler_env

python run_samplers.py \
    --raw-h5 "${SAMPLERS_RAW_H5}" \
    --data-dir "${SAMPLERS_DATA_DIR}" \
    --n-samples "${N_SAMPLES}" \
    --max-similarity-tracers "${MAX_SIMILARITY_TRACERS}" \
    --storage-format "${STORAGE_FORMAT}" \
    --samplers ${SAMPLING_PROCEDURE}
