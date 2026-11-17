#!/bin/bash
#SBATCH --job-name=input_normalization_stats
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=08:00:00
#SBATCH --output=logs_slurm/input_normalization_stats_%j.out

# Full train-split statistics for a PF dataset stored on Tallinn.
# Example:
#   sbatch scripts/tallinn/input_normalization_stats.sh cms 2.8.0 /local/joosep/mlpf/tensorflow_datasets/cms/2.8.0

set -euo pipefail

DATASET=${1:?dataset is required}
VERSION=${2:?version is required}
DATA_DIR=${3:?TFDS data directory is required}
WORK_DIR=logs_slurm/input_normalization_stats_${SLURM_JOB_ID}

export PYTHONPATH="$PWD"
export MPLCONFIGDIR=/tmp

srun --kill-on-bad-exit=1 .venv/bin/python scripts/compute_input_normalization_stats.py map "$DATASET" "$VERSION" --data-dir "$DATA_DIR" --work-dir "$WORK_DIR" "${@:4}"
.venv/bin/python scripts/compute_input_normalization_stats.py reduce "$DATASET" "$VERSION" --work-dir "$WORK_DIR"
