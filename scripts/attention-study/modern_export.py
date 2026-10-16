"""Retry modern exports at their supported opset, retaining the failed opset-20 evidence."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from study import Attention, dump, inputs
from torch.nn.attention import SDPBackend, sdpa_kernel

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
(root / "previous-modern-export-attempts.json").write_text(json.dumps([r for r in manifest if r["kind"] == "sdpa_dynamo"], indent=2))
manifest = [r for r in manifest if r["kind"] != "sdpa_dynamo"]
for dtype in ["fp32", "fp16"]:
    for mode in ["unmasked", "padding"]:
        arrays = inputs(4, 512, 16, 16, np.float32 if dtype == "fp32" else np.float16, mode)
        r = {"kind": "sdpa_dynamo", "dtype": dtype, "mode": mode, "batch": 4, "seq": 512, "heads": 16, "dim": 16, "dynamic": False, "opset": 23}
        path = root / f"sdpa_dynamo_{dtype}_{mode}_b4_s512.onnx"
        r["model"] = path.name
        try:
            with sdpa_kernel(SDPBackend.MATH):
                torch.onnx.export(
                    Attention(mode),
                    tuple(torch.from_numpy(x) for x in arrays),
                    str(path),
                    input_names=["query", "key", "value"] + (["mask"] if len(arrays) == 4 else []),
                    output_names=["output"],
                    opset_version=23,
                    dynamo=True,
                    external_data=False,
                )
            r.update(status="ok", nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as e:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
            r.update(status="failed", error=str(e))
        manifest.append(r)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
