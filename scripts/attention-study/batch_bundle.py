"""Balanced right-padding batch sweep, FP16, fixed average valid fraction."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from study import Attention, attention24, dump, inputs
from torch.nn.attention import SDPBackend, sdpa_kernel

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
manifest = []
for batch in [1, 2, 4, 8, 16]:
    for seq in [512, 1280]:
        arrays = inputs(batch, seq, 16, 16, np.float16, "padding", True)
        for kind in ["explicit_safe", "attention24_lengths"]:
            r = {
                "kind": kind,
                "dtype": "fp16",
                "mode": "padding",
                "batch": batch,
                "seq": seq,
                "heads": 16,
                "dim": 16,
                "dynamic": False,
                "length_policy": "balanced",
            }
            path = root / f"{kind}_fp16_padding_b{batch}_s{seq}.onnx"
            r["model"] = path.name
            try:
                if kind == "attention24_lengths":
                    attention24(path, arrays, True)
                else:
                    with sdpa_kernel(SDPBackend.MATH):
                        torch.onnx.export(
                            Attention("padding", True),
                            tuple(torch.from_numpy(x) for x in arrays),
                            str(path),
                            input_names=["query", "key", "value", "mask"],
                            output_names=["output"],
                            opset_version=20,
                            dynamo=False,
                        )
                r.update(status="ok", nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception as exc:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
                r.update(status="failed", error=str(exc))
            manifest.append(r)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2))
