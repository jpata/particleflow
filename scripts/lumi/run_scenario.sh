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

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load aws-ofi-rccl

export IMG=${IMG:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export KERAS_BACKEND=torch
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export PYTHONPATH="$REPO_ROOT"

rocm-smi --showdriverversion
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "scenario=$SCENARIO_FILE platform=$PLATFORM_FILE task_index=$TASK_INDEX"

RUN_ARGS=(
  "$REPO_ROOT/scripts/training/run_scenario.py"
  --scenario "$SCENARIO_FILE"
  --platform "$PLATFORM_FILE"
  --task-index "$TASK_INDEX"
)
if [[ -n "$SEED_OVERRIDE" ]]; then
  RUN_ARGS+=(--seed "$SEED_OVERRIDE")
fi

singularity exec \
  -B /scratch/project_465001293 \
  -B /tmp \
  "$IMG" \
  bash -lc 'source "$1/particleflow-env/bin/activate"; shift; exec python3 "$@"' \
  bash "$REPO_ROOT" "${RUN_ARGS[@]}"
