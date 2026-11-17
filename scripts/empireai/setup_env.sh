#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
if [[ $(uname -m) != aarch64 ]]; then
  echo "Set up the environment on an ARM64 compute node, not the x86 login node." >&2
  exit 1
fi
TOOLS_DIR="$EMPIREAI_REPO_ROOT/.cache/uv-bootstrap-aarch64"
UV_BIN="$TOOLS_DIR/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
  /usr/bin/python3 -m pip install --target "$TOOLS_DIR" uv==0.13.6
fi
export UV_PROJECT_ENVIRONMENT="$MLPF_GH200_ENV"
cd "$EMPIREAI_REPO_ROOT"
SYNC_ARGS=(sync --project envs/empireai-gh200 --python 3.11)
if [[ ${MLPF_UPDATE_LOCK:-0} == 1 ]]; then
  "$UV_BIN" lock --project envs/empireai-gh200 --python 3.11
elif [[ -f envs/empireai-gh200/uv.lock ]]; then
  SYNC_ARGS+=(--locked)
fi
"$UV_BIN" "${SYNC_ARGS[@]}"
"$MLPF_GH200_ENV/bin/python3" -c 'import mlpf.pipeline; import torch; from torch.nn.attention.varlen import varlen_attn; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "native varlen attention import passed")'
