"""Small causal correctness/representation diagnostic, separate from the scaling workload."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from study import attention24, dump, errors, inputs
from torch.nn.attention import SDPBackend, sdpa_kernel

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("action", choices=["export", "check"])
p.add_argument("--bundle", type=Path, required=True)
p.add_argument("--out", type=Path)
p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"])
a = p.parse_args()


class Causal(torch.nn.Module):
    def __init__(self, explicit):
        super().__init__()
        self.explicit = explicit

    def forward(self, q, k, v):
        if not self.explicit:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        scores = (q * q.shape[-1] ** -0.25) @ (k.transpose(-1, -2) * k.shape[-1] ** -0.25)
        allowed = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
        return scores.masked_fill(~allowed, float("-inf")).float().softmax(-1).to(q.dtype) @ v


if a.action == "export":
    a.bundle.mkdir(parents=True, exist_ok=True)
    manifest = []
    for dtype in ["fp32", "fp16"]:
        arr = inputs(2, 128, 4, 16, np.float32 if dtype == "fp32" else np.float16, "unmasked")
        for kind in ["sdpa_legacy", "explicit", "attention24"]:
            path = a.bundle / f"causal_{kind}_{dtype}.onnx"
            if kind == "attention24":
                attention24(path, arr, False)
                m = onnx.load(path)
                for n in m.graph.node:
                    if n.op_type == "Attention":
                        for attr in n.attribute:
                            if attr.name == "is_causal":
                                attr.i = 1
                onnx.save(m, path)
            else:
                with sdpa_kernel(SDPBackend.MATH):
                    torch.onnx.export(
                        Causal(kind == "explicit"),
                        tuple(torch.from_numpy(x) for x in arr),
                        str(path),
                        opset_version=20,
                        dynamo=False,
                        input_names=["query", "key", "value"],
                        output_names=["output"],
                    )
            manifest.append(
                {"kind": kind, "dtype": dtype, "model": path.name, "nodes": dump(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
    (a.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
else:
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[a.provider]
    if a.provider == "cuda":
        ort.preload_dlls()
    torch.set_num_threads(1)
    for r in json.loads((a.bundle / "manifest.json").read_text()):
        arr = inputs(2, 128, 4, 16, np.float32 if r["dtype"] == "fp32" else np.float16, "unmasked")
        with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
            ref = F.scaled_dot_product_attention(*(torch.from_numpy(x.astype(np.float32)) for x in arr), is_causal=True).numpy()
        row = {**r, "provider": a.provider, "provider_options": {"use_tf32": "0"} if a.provider == "cuda" else {}}
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            if a.provider != "cpu":
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            s = ort.InferenceSession(
                str(a.bundle / r["model"]), so, providers=[(provider, {"use_tf32": "0"})] if a.provider == "cuda" else [provider]
            )
            s.disable_fallback()
            out = s.run(None, dict(zip(["query", "key", "value"], arr)))[0]
            row.update(errors(out, ref, r["dtype"] == "fp16"))
            row["status"] = "ok" if row["correct"] else "incorrect"
        except Exception as exc:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
            row.update(status="failed", error=str(exc))
        rows.append(row)
    (a.out / "results.json").write_text(json.dumps(rows, indent=2))
