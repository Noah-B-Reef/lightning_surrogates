#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
prepare_logging "train"
validate_split_dir
print_config_summary
activate_conda

python "${SCRIPT_DIR}/src/train.py" \
    "${DATA_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-file "${TRAIN_CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT_NAME}" \
    --epochs "${TRAIN_EPOCHS}" \
    --accelerator "${ACCELERATOR}" \
    --devices "${DEVICES}" \
    --precision "${PRECISION}" \
    --num-workers "${NUM_WORKERS}"
