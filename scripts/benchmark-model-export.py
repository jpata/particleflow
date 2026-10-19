"""Measure one static full-model export case in a fresh process.

Run once per case/provider. Timing has no ORT profiling; use --placement in a
separate process to collect strict placement evidence. Inputs/outputs are on
host, so session.run latency includes transfers and output allocation.
"""

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import resource
import threading
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("qualification", Path(__file__).with_name("qualify-heptv2-onnx.py"))
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def vram_reader(provider):
    if provider == "cpu":
        return lambda: None, "not applicable"
    if provider == "migraphx":
        cards = sorted(Path("/dev/dri").glob("card[0-9]*"))
        if len(cards) != 1:
            raise RuntimeError(f"Expected one allocated DRM card, found {cards}")
        counter = Path("/sys/class/drm") / cards[0].name / "device/mem_info_vram_used"
        return lambda: int(counter.read_text()), str(counter)
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("NVML init failed")
    handle = ctypes.c_void_p()
    if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
        raise RuntimeError("NVML device lookup failed")

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    def read():
        info = Memory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
            raise RuntimeError("NVML memory query failed")
        return info.used

    return read, "NVML device 0 used bytes"


def rss():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--case", required=True, help="NPZ filename from manifest")
    p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--placement", action="store_true")
    p.add_argument("--flash", action="store_true", help="Request CUDA sdpa_kernel=1 for Attention-24")
    a = p.parse_args()
    if a.flash and a.provider != "cuda":
        p.error("--flash requires CUDA")
    a.outdir.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    from mlpf.model.onnx_runtime import create_session, finish_profile

    manifest = json.loads((a.bundle / "manifest.json").read_text())
    assert qualification.digest(a.bundle / "model.onnx") == manifest["model_sha256"]
    case = next(c for c in manifest["cases"] if c["file"] == a.case)
    assert qualification.digest(a.bundle / a.case) == case["sha256"]
    data = np.load(a.bundle / a.case)
    feeds = {n: data[n] for n in ["Xfeat_normed", "mask"]}
    valid = feeds["mask"].astype(bool)
    path = a.outdir / "static.onnx"
    qualification.specialize_model(a.bundle / "model.onnx", path, feeds)
    read_vram, source = vram_reader(a.provider)
    baseline = {"rss_bytes": rss(), "vram_bytes": read_vram()}
    samples, failures = [], []
    stop = threading.Event()
    phase = ["session"]

    def sample():
        while not stop.is_set():
            try:
                samples.append({"time": time.perf_counter(), "phase": phase[0], "rss_bytes": rss(), "vram_bytes": read_vram()})
            except Exception as exc:
                failures.append(str(exc))
                return
            stop.wait(0.01)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        start = time.perf_counter()
        session = create_session(
            path,
            a.provider,
            threads=a.threads,
            profile_prefix=a.outdir / "placement" if a.placement else None,
            require_full_gpu=a.provider != "cpu",
            flash=a.flash,
        )
        creation = time.perf_counter() - start
        phase[0] = "first_run"
        start = time.perf_counter()
        values = session.run(qualification.OUTPUTS, feeds)
        first = time.perf_counter() - start
        phase[0] = "warmup"
        for _ in range(a.warmup):
            values = session.run(qualification.OUTPUTS, feeds)
        phase[0] = "timing"
        durations = []
        for _ in range(a.iters):
            start = time.perf_counter_ns()
            values = session.run(qualification.OUTPUTS, feeds)
            durations.append((time.perf_counter_ns() - start) / 1e6)
        phase[0] = "validation"
        errors = {}
        for name, value in zip(qualification.OUTPUTS, values, strict=True):
            target = data[name]
            delta = value[valid].astype(np.float64) - target[valid]
            errors[name] = {
                "max_abs": float(np.max(abs(delta))),
                "mae": float(np.mean(abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(target[valid]), 1e-12)),
                "finite": bool(np.isfinite(value).all()),
                "padding_zero": bool(np.all(value[~valid] == 0)),
                "close": bool(np.allclose(value[valid], target[valid], atol=1e-3, rtol=3e-3)),
            }
        placement = finish_profile(session, a.provider, require_full_gpu=a.provider != "cpu") if a.placement else None
    finally:
        stop.set()
        sampler.join()
    if failures:
        raise RuntimeError(f"Memory sampler failed: {failures}")

    def peak(key, phases):
        vals = [s[key] for s in samples if s["phase"] in phases and s[key] is not None]
        return max(vals) if vals else None

    result = {
        "case": case,
        "provider": a.provider,
        "model_sha256": manifest["model_sha256"],
        "scope": manifest["scope"],
        "ort": ort.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "threads": a.threads,
        "compiler_environment": {k: v for k, v in os.environ.items() if k.startswith("MIGRAPHX_")},
        "session_seconds": creation,
        "first_run_seconds": first,
        "warmup": a.warmup,
        "latency_ms": {"samples": durations, **{f"p{q}": float(np.percentile(durations, q)) for q in [50, 90, 95, 99]}},
        "outputs": errors,
        "placement": placement,
        "profiling_enabled": a.placement,
        "flash_requested": a.flash,
        "memory": {
            "baseline": baseline,
            "vram_source": source,
            "sampling_interval_seconds": 0.01,
            "rss_lifetime_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "session_peak_rss_bytes": peak("rss_bytes", ["session", "first_run"]),
            "session_peak_vram_bytes": peak("vram_bytes", ["session", "first_run"]),
            "inference_peak_rss_bytes": peak("rss_bytes", ["warmup", "timing"]),
            "inference_peak_vram_bytes": peak("vram_bytes", ["warmup", "timing"]),
        },
        "passed": all(e["finite"] and e["padding_zero"] and e["close"] for e in errors.values()),
    }
    (a.outdir / "memory-samples.json").write_text(json.dumps(samples))
    (a.outdir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": a.case, "provider": a.provider, "passed": result["passed"], "p50_ms": result["latency_ms"]["p50"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
