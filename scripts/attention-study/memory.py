"""Separate memory experiment; driver-global GPU deltas, never PyTorch-only ORT memory."""

import argparse
import json
import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model", type=Path)
p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.provider == "cuda":
    ort.preload_dlls()
provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[a.provider]
process = psutil.Process()
if a.provider != "cpu":
    torch.cuda.init()
    torch.cuda.synchronize()


def usage():
    return process.memory_info().rss if a.provider == "cpu" else torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]


baseline = usage()
samples = []
stop = threading.Event()


def poll():
    while not stop.is_set():
        samples.append(usage())
        stop.wait(0.002)


thread = threading.Thread(target=poll, daemon=True)
thread.start()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
if a.provider != "cpu":
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
s = ort.InferenceSession(str(a.model), so, providers=[provider])
s.disable_fallback()
after_session = usage()
feeds = {}
for i in s.get_inputs():
    dt = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32}[i.type]
    feeds[i.name] = np.ones(i.shape, dt)
    if i.name == "mask":
        feeds[i.name][:, i.shape[-1] // 2 :] = 0
out = s.run(None, feeds)[0]
after_first = usage()
compile_peak = max(samples + [after_first])
samples.clear()
for _ in range(20):
    out = s.run(None, feeds)[0]
if a.provider != "cpu":
    torch.cuda.synchronize()
samples.append(usage())
stop.set()
thread.join()
result = {
    "model": a.model.name,
    "provider": a.provider,
    "metric": "process_rss" if a.provider == "cpu" else "driver_global_device_used",
    "baseline_bytes": baseline,
    "after_session_bytes": after_session,
    "after_first_bytes": after_first,
    "compile_peak_delta_bytes": compile_peak - baseline,
    "inference_peak_delta_bytes": max(samples) - baseline,
    "sample_interval_s": 0.002,
    "samples": len(samples),
    "environment": {k: v for k, v in os.environ.items() if k.startswith("MIGRAPHX_")},
    "finite": bool(np.isfinite(out).all()),
    "limitations": "GPU deltas include context/provider caches and all device users; sampled peak is a lower bound, not an allocator exact peak. Timings are collected in a separate experiment.",
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(result, indent=2))
