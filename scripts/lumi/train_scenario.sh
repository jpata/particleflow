#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

module purge
module load Local-LAIF lumi-aif-singularity-bindings
export PYTHONNOUSERSITE=1

export IMG=${IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
export KERAS_BACKEND=torch
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

# Resolve configs with container Python, then submit using the host Slurm client.
SUBMISSION_SCRIPT=$(mktemp /tmp/mlpf-lumi-submit.XXXXXX)
trap 'rm -f "$SUBMISSION_SCRIPT"' EXIT
singularity run \
  -B /scratch/project_465001293 \
  -B /tmp \
  "$IMG" \
  bash -c 'set -e; if [[ -n "${LUMI_VENV:-}" ]]; then source "$LUMI_VENV/bin/activate"; fi; exec "${PYTHON_EXECUTABLE:-python3}" "$@"' \
  bash "$REPO_ROOT/scripts/lumi/submit_scenario.py" \
  "$@" --submission-script "$SUBMISSION_SCRIPT"

# --dry-run and --list do not write a submission script.
if [[ -s "$SUBMISSION_SCRIPT" ]]; then
  bash "$SUBMISSION_SCRIPT"
fi
