#!/bin/bash
#SBATCH --job-name=h100_batch_sweep
#SBATCH --partition=gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs_slurm/h100_batch_sweep_%j.log
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
# Compatibility worker. Prefer submit_batch_sweep.py to read resources from the profile.
exec srun --ntasks=1 --cpu-bind=cores \
  bash scripts/flatiron/batch_python.sh -m scripts.training.batch_sweep \
  --platform configs/training/platforms/flatiron_h100_single.yaml "$@"
