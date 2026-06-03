#!/bin/bash
#SBATCH -A AST24021
#SBATCH --job-name=gc_test
#SBATCH --output=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/output/test_%j.out
#SBATCH --error=/work/10252/nbr525/vista/lightning_surrogates/models/mlp/results/error/test_%j.err
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
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
prepare_logging "test"
validate_split_dir
print_config_summary
activate_conda

python "${SCRIPT_DIR}/src/test.py" \
    "${DATA_DIR}" \
    --model-checkpoint "${TEST_MODEL_CHECKPOINT}" \
    --output-dir "${TEST_OUTPUT_DIR}"
