"""Separate Attention-23 softmax precision from input/output dtype and runtime version."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper
from study import errors, inputs, native

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
arrays = inputs(4, 512, 16, 16, np.float16, "unmasked")
reference = native([x.astype(np.float32) for x in arrays]).numpy()
rows = []
for precision in [0, 1]:
    model = onnx.load(a.model)
    for node in model.graph.node:
        if node.op_type == "Attention":
            attrs = [attr for attr in node.attribute if attr.name != "softmax_precision"]
            del node.attribute[:]
            node.attribute.extend(attrs)
            node.attribute.append(helper.make_attribute("softmax_precision", precision))
    path = a.out / f"attention23-softmax-{precision}.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = session.run(None, dict(zip(["query", "key", "value"], arrays)))[0]
    rows.append({"ort": ort.__version__, "softmax_precision": precision, **errors(out, reference, True)})
(a.out / "results.json").write_text(json.dumps(rows, indent=2))
