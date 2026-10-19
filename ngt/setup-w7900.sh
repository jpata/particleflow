#!/usr/bin/env bash
# Run INSIDE the NGT GPU session. Installs into /tmp, not the repository env.
set -euo pipefail
: "${NGT_VENV:=/tmp/sdpa-w7900-venv}"
: "${ROCM_PATH:=/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$ROCM_PATH/bin/rocminfo" ]]; then
    echo "Set ROCM_PATH to the installed ROCm 7.2.x directory (inspect /opt/rocm*)." >&2
    exit 1
fi
"$ROCM_PATH/bin/rocminfo" > /tmp/sdpa-rocminfo.txt
if ! rg_output=$(grep -E 'gfx1100' /tmp/sdpa-rocminfo.txt); then
    echo "Expected a W7900 (gfx1100); inspect /tmp/sdpa-rocminfo.txt." >&2
    exit 1
fi
printf '%s\n' "$rg_output"
if ! command -v python3.12 >/dev/null; then
    echo "Python 3.12 is required. In the NGT Alma9 image: sudo dnf install -y python3.12 python3.12-pip" >&2
    exit 1
fi
python3.12 -m venv "$NGT_VENV"
"$NGT_VENV/bin/python" -m pip install --upgrade pip
"$NGT_VENV/bin/python" -m pip install 'numpy==1.26.4' 'onnx==1.19.1' psutil
"$NGT_VENV/bin/python" -m pip install \
    'torch==2.9.1+rocm7.2.1.lw.gitff65f5bc' 'onnxruntime-migraphx==1.23.2' \
    --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
"$NGT_VENV/bin/python" -m pip check
"$NGT_VENV/bin/python" - <<'PY'
import torch
import onnxruntime as ort
assert torch.version.hip, 'Installed torch is not a ROCm build'
assert torch.cuda.is_available(), 'ROCm cannot access the GPU'
print('GPU:', torch.cuda.get_device_name(0), 'HIP:', torch.version.hip)
assert 'MIGraphXExecutionProvider' in ort.get_available_providers(), ort.get_available_providers()
x = torch.randn(32, 32, device='cuda')
print('GPU matmul:', (x @ x).sum().item())
print('ORT providers:', ort.get_available_providers())
PY
printf '\nFor this shell, run:\nexport ROCM_PATH=%q\nexport LD_LIBRARY_PATH=%q\nsource %q/bin/activate\n' "$ROCM_PATH" "$LD_LIBRARY_PATH" "$NGT_VENV"
