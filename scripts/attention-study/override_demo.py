"""Exercise the repository's actual scoped Attention-24 symbolic override."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from study import dump, errors, inputs, native

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--wrapper", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("mlpf_onnx_attention", a.wrapper)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class Block(torch.nn.Module):
    def forward(self, query, key, value, mask):
        out = wrapper.export_attention(query, key, value, mask.bool())
        return out * mask[:, None, :, None].to(out.dtype)


arrays = inputs(2, 128, 4, 16, np.float16 if a.dtype == "fp16" else np.float32, "padding")
tensors = tuple(torch.from_numpy(x) for x in arrays)
# Validate the real deployment contract rather than silently changing semantics.
rejections = {}
for mode in ["empty", "holes"]:
    try:
        wrapper.validate_padding_mask(torch.from_numpy(inputs(2, 128, 4, 16, np.float32, mode)[3]))
        rejections[mode] = False
    except ValueError as exc:
        rejections[mode] = str(exc)
placeholder = a.out / "override-placeholder.onnx"
torch.onnx.export(
    Block().eval(),
    tensors,
    str(placeholder),
    opset_version=20,
    custom_opsets={"mlpf": 1},
    dynamo=False,
    input_names=["query", "key", "value", "mask"],
    output_names=["output"],
)
m = onnx.version_converter.convert_version(onnx.load(placeholder), 24)
for n in m.graph.node:
    if n.domain == "mlpf" and n.op_type == "Attention24":
        n.domain = ""
        n.op_type = "Attention"
imports = [o for o in m.opset_import if o.domain != "mlpf"]
del m.opset_import[:]
m.opset_import.extend(imports)
onnx.checker.check_model(m)
final = a.out / "override-attention24.onnx"
onnx.save(m, final)
for path in [placeholder, final]:
    dump(path)
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
s = ort.InferenceSession(str(final), so, providers=["CPUExecutionProvider"])
out = s.run(None, dict(zip(["query", "key", "value", "mask"], arrays)))[0]
ref = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(arrays)]).numpy()
result = {
    "wrapper_sha256": hashlib.sha256(a.wrapper.read_bytes()).hexdigest(),
    "dtype": a.dtype,
    "correctness": errors(out, ref, a.dtype == "fp16"),
    "contract_rejections": rejections,
    "final_nodes": dump(final),
    "placeholder_nodes": dump(placeholder),
}
(a.out / "validation.json").write_text(json.dumps(result, indent=2))
assert result["correctness"]["correct"]
assert all(rejections.values())
