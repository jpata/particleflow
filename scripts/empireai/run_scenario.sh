#!/bin/bash
set -euo pipefail
SCENARIO_FILE=${1:?scenario file is required}
PLATFORM_FILE=${2:?platform profile is required}
shift 2
# Slurm may spool this script elsewhere; use the submission working directory.
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ${1:-} == --repo-root ]]; then
  REPO_ROOT=${2:?--repo-root requires a value}
  shift 2
fi
source "$REPO_ROOT/scripts/empireai/environment.sh"
require_arm_environment
cd "$REPO_ROOT"
nvidia-smi
python3 -c 'import platform, torch; print("Architecture:", platform.machine(), "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
RUN_ARGS=(--scenario "$SCENARIO_FILE" --platform "$PLATFORM_FILE"
  --task-index "${SLURM_ARRAY_TASK_ID:-0}")
exec python3 scripts/empireai/preflight_and_run.py "${RUN_ARGS[@]}" "$@"
