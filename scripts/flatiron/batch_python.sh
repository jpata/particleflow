#!/bin/bash
set -euo pipefail
REPO_ROOT=${MLPF_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
module --force purge
module load modules/2.4-20250724
module load slurm gcc cmake cuda/12.8.0 cudnn/9.2.0.82-12 nccl openmpi apptainer
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MLPF_CLD_DATA_DIR=${MLPF_CLD_DATA_DIR:-/mnt/ceph/users/${USER}/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds}
# Install/sync once before submission. Never resolve a new environment per case.
exec uv run --no-sync python3 "$@"
