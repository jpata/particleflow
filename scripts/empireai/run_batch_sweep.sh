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
source "$REPO_ROOT/scripts/empireai/environment.sh"
require_arm_environment
cd "$REPO_ROOT"
OUTPUT=${MLPF_BATCH_SWEEP_OUTPUT:-$REPO_ROOT/experiments/gh200_batch_sweep_${SLURM_JOB_ID:-manual}}
exec python3 scripts/training/batch_sweep.py \
  --scenario configs/training/scenarios/cld_gh200_backbone_benchmark.yaml \
  --platform configs/training/platforms/empireai_gh200.yaml \
  --output "$OUTPUT" "$@"
