"""Run only ORT inference so rocprof traces exclude PyTorch reference kernels."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("--iters", type=int, default=3)
args = parser.parse_args()
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
session = ort.InferenceSession(str(args.model), options, providers=["MIGraphXExecutionProvider"])
session.disable_fallback()
assert "MIGraphXExecutionProvider" in session.get_providers()
rng = np.random.default_rng(12345)
feeds = {}
for inp in session.get_inputs():
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32}[inp.type]
    assert all(isinstance(dim, int) for dim in inp.shape), inp.shape
    if inp.name == "mask":
        value = np.ones(inp.shape, dtype=dtype)
        value[:, inp.shape[1] // 2 :] = 0
    else:
        value = rng.standard_normal(inp.shape).astype(dtype)
    feeds[inp.name] = value
for _ in range(args.iters):
    output = session.run(None, feeds)[0]
    assert np.isfinite(output).all()
print("Model:", args.model, "providers:", session.get_providers(), "output:", output.shape, output.dtype)
