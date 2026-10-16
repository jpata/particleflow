"""Capture versions and loaded libraries after executing a real provider session."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch  # noqa: F401 - load vendor libraries in the same order as the timing harness.
from provenance import collect

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
if a.provider == "cuda":
    ort.preload_dlls()
provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[a.provider]
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
session = ort.InferenceSession(str(a.model), so, providers=[provider])
feeds = {}
for item in session.get_inputs():
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32}[item.type]
    feeds[item.name] = np.ones(item.shape, dtype=dtype)
assert np.isfinite(session.run(None, feeds)[0]).all()
a.out.write_text(json.dumps(collect(), indent=2))
