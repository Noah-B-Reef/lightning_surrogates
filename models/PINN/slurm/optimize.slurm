#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=pinn_optimize
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

# All paths and pipeline arguments come from the repo config (config.sh),
# loaded by common.sh. Submit from models/PINN: sbatch slurm/<name>.slurm
# Locate common.sh whether the job was submitted from models/PINN/ or from
# models/PINN/slurm/. sbatch copies this script to the slurmd spool directory,
# so its own path (BASH_SOURCE) is only usable when run outside SLURM.
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
validate_split_dir
print_config_summary
activate_conda

if [ "${OPTUNA_STORAGE:-auto}" = "auto" ]; then
    OPTUNA_STORAGE_ARG=()
else
    OPTUNA_STORAGE_ARG=(--storage "${OPTUNA_STORAGE}")
fi

python "${SCRIPT_DIR}/src/optimize.py" \
    "${LS_DATA_DIR}" \
    --results-dir "${OPTUNA_RESULTS_DIR}" \
    --study-name "${PINN_STUDY_NAME}" \
    --journal-mode "${JOURNAL_MODE}" \
    "${OPTUNA_STORAGE_ARG[@]}" \
    --num-trials "${N_TRIALS}" \
    --tune-epochs "${TUNE_EPOCHS}" \
    --max-horizon "${PINN_MAX_HORIZON}" \
    --patience "${PRUNER_PATIENCE}" \
    --min-relative-improvement "${MIN_RELATIVE_IMPROVEMENT}" \
    --no-logger \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"
