#!/bin/bash
set -euo pipefail

SCENARIO_FILE=${1:?scenario file is required}
PLATFORM_FILE=${2:?platform profile is required}
shift 2

TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
SEED_OVERRIDE=${SEED:-}
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
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
    --repo-root)
      REPO_ROOT=${2:?--repo-root requires a value}
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$REPO_ROOT/scripts/training/run_scenario.py" ]]; then
  echo "Invalid repository root '$REPO_ROOT': scripts/training/run_scenario.py is missing" >&2
  exit 2
fi
cd "$REPO_ROOT"

export PF_SITE=tallinn
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=1
export PYTHONPATH="$REPO_ROOT"

nvidia-smi topo -m
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "scenario=$SCENARIO_FILE platform=$PLATFORM_FILE task_index=$TASK_INDEX"

RUN_ARGS=(
  --scenario "$SCENARIO_FILE"
  --platform "$PLATFORM_FILE"
  --task-index "$TASK_INDEX"
)
if [[ -n "$SEED_OVERRIDE" ]]; then
  RUN_ARGS+=(--seed "$SEED_OVERRIDE")
fi

exec uv run python3 scripts/training/run_scenario.py "${RUN_ARGS[@]}"
