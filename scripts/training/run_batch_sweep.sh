#!/bin/bash
# Resources come from submit_batch_sweep.py and the platform YAML, not this worker.
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_WRAPPER=$1
shift
if [[ "$PYTHON_WRAPPER" == - ]]; then
  exec srun --ntasks=1 --cpu-bind=cores python3 -m scripts.training.batch_sweep "$@"
fi
exec srun --ntasks=1 --cpu-bind=cores bash "$PYTHON_WRAPPER" -m scripts.training.batch_sweep "$@"
