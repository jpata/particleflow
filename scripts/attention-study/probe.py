"""ORT-only profiling and dynamic-shape compilation probe."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model", type=Path)
p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--allow-cpu", action="store_true")
p.add_argument("--sizes", type=int, nargs="+")
p.add_argument("--cache", type=Path)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[a.provider]
if a.provider == "cuda":
    import torch  # noqa: F401 - load CUDA wheel libraries without executing GPU operations.

    ort.preload_dlls()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.enable_profiling = True
so.profile_file_prefix = str(a.out / "ort")
if a.provider != "cpu" and not a.allow_cpu:
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
opts = {}
if a.provider == "cuda":
    opts["use_tf32"] = "0"
if a.cache:
    opts["migraphx_model_cache_dir"] = str(a.cache)
    a.cache.mkdir(parents=True, exist_ok=True)
records = []
for session_index in range(2 if a.sizes else 1):
    t = time.perf_counter()
    s = ort.InferenceSession(str(a.model), so, providers=[(provider, opts)] + (["CPUExecutionProvider"] if a.allow_cpu else []))
    s.disable_fallback()
    records.append({"session": session_index, "session_ms": (time.perf_counter() - t) * 1000})
    for seq in a.sizes or [None] * 3:
        rng = np.random.default_rng(12345)
        feeds = {}
        for i in s.get_inputs():
            shape = [(4 if d == "batch" else seq if isinstance(d, str) else d) for d in i.shape]
            dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32}[i.type]
            if i.name == "mask":
                v = np.ones(shape, dtype)
                v[:, shape[-1] // 2 :] = 0
            else:
                v = rng.standard_normal(shape).astype(dtype)
            feeds[i.name] = v
        t = time.perf_counter()
        out = s.run(None, feeds)[0]
        elapsed = (time.perf_counter() - t) * 1000
        records.append({"session": session_index, "seq": seq or out.shape[-2], "run_ms": elapsed, "finite": bool(np.isfinite(out).all())})
        assert np.isfinite(out).all()
    profile = Path(s.end_profiling())
    events = json.loads(profile.read_text())
    records.append(
        {
            "session": session_index,
            "placements": [
                {"name": e["name"], "op": e["args"].get("op_name"), "provider": e["args"]["provider"]}
                for e in events
                if e.get("args", {}).get("provider")
            ],
        }
    )
    del s
(a.out / "probe.json").write_text(json.dumps(records, indent=2))
