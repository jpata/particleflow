#!/bin/bash
#SBATCH --job-name=lumi_batch_sweep
#SBATCH --partition=small-g
#SBATCH --account=project_465001293
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs_slurm/lumi_batch_sweep_%j.log
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
# Compatibility worker. Prefer submit_batch_sweep.py to read resources from the profile.
exec srun --ntasks=1 --cpu-bind=cores \
  bash scripts/lumi/batch_python.sh -m scripts.training.batch_sweep \
  --platform configs/training/platforms/lumi_mi250x_single.yaml "$@"
