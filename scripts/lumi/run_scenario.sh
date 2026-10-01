#!/bin/bash
set -euo pipefail

SCENARIO_FILE=${1:?scenario file is required}
PLATFORM_FILE=${2:?platform profile is required}
shift 2

TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
SEED_OVERRIDE=${SEED:-}
CONTINUE_RUN=0
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
    --continue)
      CONTINUE_RUN=1
      shift
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

module purge
module load Local-LAIF lumi-aif-singularity-bindings
export PYTHONNOUSERSITE=1

export IMG=${IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/tmp/${USER}-${SLURM_JOB_ID}-miopen-cache}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export KERAS_BACKEND=torch
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-3}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export PYTHONPATH="$REPO_ROOT"

# A standard-g allocation is a complete node and this profile requests all
# eight MI250X GCDs for one task. Do not inherit a login-shell or Slurm binding
# that would hide devices from the eight-process torch DDP launcher.
unset ROCR_VISIBLE_DEVICES

if [[ ! -r "$IMG" ]]; then
  echo "LUMI PyTorch container is not readable at '$IMG'; set IMG" >&2
  exit 2
fi
# The old particleflow-env points to the previous container's Conda Python.
# Use an overlay built in LAIF with access to its GPU packages.
export LUMI_VENV=${LUMI_VENV:-$REPO_ROOT/particleflow-laif-env}
if [[ -n "$LUMI_VENV" && ! -f "$LUMI_VENV/bin/activate" ]]; then
  echo "Python environment activation script is missing at '$LUMI_VENV/bin/activate'" >&2
  echo "Run bash scripts/lumi/setup_env.sh to create the LAIF overlay" >&2
  exit 2
fi
# Keep the same release if the latest symlink changes while jobs are queued.
export IMG=$(readlink -f "$IMG")
mkdir -p "$MIOPEN_USER_DB_PATH"

rocm-smi --showdriverversion
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-all} OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
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
if [[ "$CONTINUE_RUN" == 1 ]]; then
  RUN_ARGS+=(--continue)
fi

singularity run \
  -B /scratch/project_465001293 \
  -B /tmp \
  "$IMG" \
  bash -c 'set -e; if [[ -n "${LUMI_VENV:-}" ]]; then source "$LUMI_VENV/bin/activate"; fi; exec python3 "$@"' \
  bash "${RUN_ARGS[@]}"
