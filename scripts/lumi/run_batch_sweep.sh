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
OUTPUT=${MLPF_BATCH_SWEEP_OUTPUT:-${LUMI_PROJECT_ROOT:-/scratch/project_465001293}/${USER}/particleflow/experiments/lumi_batch_sweep_${SLURM_JOB_ID}}
echo "Batch sweep output: $OUTPUT"
# srun establishes the one-device visibility mask before entering the container.
exec srun --ntasks=1 --gpus-per-task=1 --cpu-bind=cores \
  bash scripts/lumi/batch_python.sh scripts/training/monitor_batch_memory.py \
  --diagnostics "$OUTPUT/memory-diagnostics.jsonl" -- scripts/training/batch_sweep.py \
  --scenario configs/training/scenarios/cld_gh200_backbone_benchmark.yaml \
  --platform configs/training/platforms/lumi_mi250x_single.yaml \
  --output "$OUTPUT" "$@"
