#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nreef@utexas.edu

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
trap 'status=$?; echo "FAILED at line ${LINENO} with exit code ${status}"; exit ${status}' ERR

cd "${SCRIPT_DIR}"
prepare_logging "test"
validate_split_dir
print_config_summary
activate_conda

python "${SCRIPT_DIR}/src/test.py" \
    "${DATA_DIR}" \
    --model-checkpoint "${TEST_MODEL_CHECKPOINT}" \
    --output-dir "${TEST_OUTPUT_DIR}"
