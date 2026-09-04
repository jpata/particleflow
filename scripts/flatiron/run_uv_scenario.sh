#!/bin/bash
set -euo pipefail

SCENARIO_FILE=${1:?scenario file is required}
PLATFORM_FILE=${2:?platform profile is required}
shift 2

TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
SEED_OVERRIDE=${SEED:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-index)
      TASK_INDEX=${2:?--task-index requires a value}
      shift 2
      ;;
    --seed)
      SEED_OVERRIDE=${2:?--seed requires a value}
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

module --force purge
module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer

nvidia-smi
export PYTHONPATH="$REPO_ROOT"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export RAY_USAGE_STATS_DISABLE=1
export RAY_TRAIN_V2_ENABLED=1

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-unknown}"
echo "scenario=$SCENARIO_FILE platform=$PLATFORM_FILE task_index=$TASK_INDEX"

RUN_ARGS=(
  --scenario "$SCENARIO_FILE"
  --platform "$PLATFORM_FILE"
  --task-index "$TASK_INDEX"
)
if [[ -n "$SEED_OVERRIDE" ]]; then
  RUN_ARGS+=(--seed "$SEED_OVERRIDE")
fi

uv run python3 scripts/training/run_scenario.py "${RUN_ARGS[@]}"
