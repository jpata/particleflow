"""Materialize the dense mask query axis required by ORT, preserving raw exporter output."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from study import attention24, dump, inputs

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
for r in list(manifest):
    if r["kind"] == "attention24_dense":
        a = inputs(r["batch"], r["seq"], r["heads"], r["dim"], np.float32 if r["dtype"] == "fp32" else np.float16, r["mode"])
        path = root / r["model"]
        attention24(path, a, False, r["dynamic"])
        r.update(nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if r["kind"] == "sdpa_dynamo" and r["mode"] == "padding" and r["status"] == "ok":
        m = onnx.load(root / r["model"])
        nodes = []
        for n in m.graph.node:
            if n.op_type == "Attention":
                shape = [r["batch"], 1, r["seq"], r["seq"]]
                nodes += [
                    h.make_node("Constant", [], ["expanded_mask_shape"], value=h.make_tensor("expanded_mask_shape", T.INT64, [4], shape)),
                    h.make_node("Expand", [n.input[3], "expanded_mask_shape"], ["expanded_attention_mask"]),
                ]
                n.input[3] = "expanded_attention_mask"
            nodes.append(n)
        del m.graph.node[:]
        m.graph.node.extend(nodes)
        rr = {**r, "kind": "sdpa_dynamo_expanded"}
        path = root / r["model"].replace("sdpa_dynamo_", "sdpa_dynamo_expanded_")
        rr["model"] = path.name
        onnx.checker.check_model(m)
        onnx.save(m, path)
        rr.update(nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest.append(rr)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
