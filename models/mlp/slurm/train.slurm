#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_train
#SBATCH --output=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/output/train_%j.out
#SBATCH --error=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/error/train_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    if [ -f "${SLURM_SUBMIT_DIR}/slurm/common.sh" ]; then
        source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
    else
        source "${SLURM_SUBMIT_DIR}/common.sh"
    fi
else
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
fi
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
validate_split_dir
print_config_summary
activate_conda

python "${SCRIPT_DIR}/src/train.py" \
    "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${TRAIN_CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --rollout-steps "${MODEL_ROLLOUT_STEPS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"
