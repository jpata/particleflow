#!/bin/bash
#SBATCH --job-name=gh200_batch_sweep
#SBATCH --partition=grace
#SBATCH --account=cuny
#SBATCH --qos=burst
#SBATCH --reservation=ghtest
#SBATCH --constraint=gracehopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs_slurm/gh200_batch_sweep_%j.log
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
# Compatibility worker. Prefer submit_batch_sweep.py to read resources from the profile.
exec srun --ntasks=1 --cpu-bind=cores \
  bash scripts/empireai/batch_python.sh -m scripts.training.batch_sweep \
  --platform configs/training/platforms/empireai_gh200.yaml "$@"
