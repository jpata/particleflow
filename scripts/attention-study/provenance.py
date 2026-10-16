"""Capture reproducibility information without collecting credentials."""

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def command(argv):
    if not shutil.which(argv[0]):
        return {"status": "unavailable", "command": argv}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
        return {"command": argv, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, AttributeError, subprocess.TimeoutExpired) as e:
        return {"command": argv, "error": str(e)}


def collect():
    import onnxruntime as ort
    import torch

    commands = {
        "cpu": ["lscpu"],
        "os": ["cat", "/etc/os-release"],
        "nvidia": ["nvidia-smi", "-q"],
        "cuda_toolkit": ["nvcc", "--version"],
        "amdgpu_module": ["modinfo", "amdgpu"],
        "hip_toolkit": ["hipconfig", "--full"],
        "rocm_smi": ["rocm-smi", "--showdriverversion", "--showproductname", "--showmeminfo", "vram"],
        "amd_smi": ["amd-smi", "version"],
        "rocminfo": ["rocminfo"],
        "rocm_packages": ["rpm", "-qa", "*rocm*", "*migraphx*", "*hip*", "*rocblas*", "*miopen*", "*hipblas*", "*amdgpu*"],
        "cuda_packages": ["rpm", "-qa", "*cuda*", "*cudnn*", "*nvidia*"],
        "rocprof": ["rocprofv3", "--version"],
        "nsys": ["nsys", "--version"],
        "ncu": ["ncu", "--version"],
        "pip_check": [sys.executable, "-m", "pip", "check"],
    }
    data = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "python_packages": [
            {"name": n, "version": v}
            for n, v in sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()}, key=lambda x: x[0].lower())
        ],
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "hip_build": torch.version.hip,
            "cudnn_version": torch.backends.cudnn.version() if not torch.version.hip else None,
            "miopen_version_via_torch": torch.backends.cudnn.version() if torch.version.hip else None,
            "config": torch.__config__.show(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "ort": {"version": ort.__version__, "providers": ort.get_available_providers(), "build": ort.get_build_info()},
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MIGRAPHX_", "ORT_", "ROCBLAS_", "HIPBLASLT_", "TORCH_ROCM_"))
            or k
            in [
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "ROCM_PATH",
                "LD_LIBRARY_PATH",
            ]
        },
        "commands": {k: command(v) for k, v in commands.items()},
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        data["torch"]["device"] = {"name": p.name, "total_memory": p.total_memory, "capability": torch.cuda.get_device_capability(0)}
        torch.empty(1, device="cuda")
    # Query the driver API separately from toolkit/build versions.
    if torch.version.cuda:
        try:
            lib = ctypes.CDLL("libcuda.so.1")
            version = ctypes.c_int()
            lib.cuInit(0)
            status = lib.cuDriverGetVersion(ctypes.byref(version))
            data["cuda_driver_api"] = {"status": status, "version_integer": version.value}
        except (OSError, AttributeError, subprocess.TimeoutExpired) as e:
            data["cuda_driver_api"] = {"error": str(e)}
    if torch.version.cuda:
        try:
            lib = ctypes.CDLL("libcudart.so.12")
            version = ctypes.c_int()
            status = lib.cudaRuntimeGetVersion(ctypes.byref(version))
            data["cuda_runtime_api"] = {"status": status, "version_integer": version.value}
        except (OSError, AttributeError) as exc:
            data["cuda_runtime_api"] = {"error": str(exc)}
    if torch.version.hip:
        try:
            lib = ctypes.CDLL("libamdhip64.so")
            version = ctypes.c_int()
            status = lib.hipRuntimeGetVersion(ctypes.byref(version))
            data["hip_runtime_api"] = {"status": status, "version_integer": version.value}
        except (OSError, AttributeError, subprocess.TimeoutExpired) as e:
            data["hip_runtime_api"] = {"error": str(e)}
    for name, path in [
        ("nvidia_kernel", "/proc/driver/nvidia/version"),
        ("amdgpu_module", "/sys/module/amdgpu/version"),
        ("kernel", "/proc/version"),
    ]:
        try:
            data[name] = Path(path).read_text()
        except OSError as e:
            data[name] = {"unavailable": str(e)}
    data["auxiliary_cxx_runtime_packages"] = []
    for metadata in Path("/tmp/sdpa-cxx-runtime/conda-meta").glob("*.json"):
        record = json.loads(metadata.read_text())
        data["auxiliary_cxx_runtime_packages"].append({k: record.get(k) for k in ["name", "version", "build", "sha256"]})
    # Record actual loaded libraries as well as package metadata.
    data["mapped_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line
            and any(s in line.lower() for s in ["torch", "onnxruntime", "cuda", "cudnn", "hip", "migraph", "rocblas", "miopen", "stdc++"])
        }
    )
    data["source_hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    return data


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(collect(), indent=2) + "\n")
