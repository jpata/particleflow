#!/bin/bash
# Execute Python inside the same LAIF overlay for setup checks, dry-runs and jobs.
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LUMI_PROJECT_ROOT=${LUMI_PROJECT_ROOT:-/scratch/project_465001293}
export IMG=${IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
export LUMI_VENV=${LUMI_VENV:-$REPO_ROOT/particleflow-laif-env}
export MLPF_CLD_DATA_DIR=${MLPF_CLD_DATA_DIR:-$LUMI_PROJECT_ROOT/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export KERAS_BACKEND=torch
export OMP_NUM_THREADS=1
export MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-/tmp/${USER}-${SLURM_JOB_ID:-manual}-batch-miopen}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}
if [[ ! -r "$IMG" || ! -f "$LUMI_VENV/bin/activate" ]]; then
  echo "Set IMG to a readable LAIF image and run bash scripts/lumi/setup_env.sh with the same IMG/LUMI_VENV." >&2
  exit 2
fi
export IMG=$(readlink -f "$IMG")
mkdir -p "$MIOPEN_USER_DB_PATH"
# Preserve Slurm's ROCR/HIP visibility mask for this single-GCD job.
exec singularity run -B "$LUMI_PROJECT_ROOT" -B /tmp "$IMG" \
  bash -c 'set -euo pipefail; source "$LUMI_VENV/bin/activate"; exec python3 "$@"' bash "$@"
