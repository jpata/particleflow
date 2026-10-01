#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export PYTHONNOUSERSITE=1
export IMG=${IMG:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}
export LUMI_VENV=${LUMI_VENV:-$REPO_ROOT/particleflow-laif-env}
singularity run -B /scratch/project_465001293 -B /tmp "$IMG" bash -c '
  set -euo pipefail
  if [[ ! -f "$LUMI_VENV/bin/activate" ]]; then
    python -m venv --system-site-packages "$LUMI_VENV"
  fi
  source "$LUMI_VENV/bin/activate"
  python -m pip install -r scripts/lumi/requirements-laif.txt
'
