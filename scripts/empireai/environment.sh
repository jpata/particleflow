#!/bin/bash
# Source from a launcher or setup script. Keep binaries and caches ARM-specific.
EMPIREAI_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export MLPF_GH200_ENV=${MLPF_GH200_ENV:-$EMPIREAI_REPO_ROOT/envs/empireai-gh200/.venv}
export MLPF_CLD_DATA_DIR=${MLPF_CLD_DATA_DIR:-$EMPIREAI_REPO_ROOT/data/empireai-smoke/tfds}
export UV_CACHE_DIR="$EMPIREAI_REPO_ROOT/.cache/uv-aarch64"
export UV_PYTHON_INSTALL_DIR="$EMPIREAI_REPO_ROOT/.cache/python-aarch64"
export PYTHONPATH="$EMPIREAI_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PATH="$MLPF_GH200_ENV/bin:$PATH"

require_arm_environment() {
  if [[ $(uname -m) != aarch64 ]]; then
    echo "Run this command on an ARM64 compute node (use reservation ghtest)." >&2
    return 1
  fi
  if [[ ! -x "$MLPF_GH200_ENV/bin/python3" ]]; then
    echo "Missing ARM environment. Run scripts/empireai/setup_env.sh on alphagh01 first." >&2
    return 1
  fi
}
