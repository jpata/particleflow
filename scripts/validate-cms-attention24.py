"""Compare valid CMS outputs against the original model, one event per input NPZ."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def compare(a, b):
    r = {}
    for name, y, z in zip(["bid", "id", "momentum", "pu"], a, b):
        d = np.abs(y - z)
        r[name] = {
            "max_abs": float(d.max()),
            "mae": float(d.mean()),
            "p99_abs": float(np.quantile(d, 0.99)),
            "p999_abs": float(np.quantile(d, 0.999)),
            "finite": bool(np.isfinite(z).all()),
        }
        if name != "momentum":
            r[name]["argmax_changes"] = int(np.sum(y.argmax(-1) != z.argmax(-1)))
    active = (a[0].argmax(-1) == 1) & (b[0].argmax(-1) == 1)
    r["active_pid_changes"] = int(np.sum((a[1].argmax(-1) != b[1].argmax(-1)) & active))
    r["active_elements"] = int(active.sum())
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.device == "cuda":
        ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    providers = [("CUDAExecutionProvider", {"use_tf32": "0", "sdpa_kernel": "1"})] if args.device == "cuda" else ["CPUExecutionProvider"]
    original = ort.InferenceSession(str(args.original), opts, providers=providers)
    patched = ort.InferenceSession(str(args.model), opts, providers=providers)
    assert args.device != "cuda" or "CUDAExecutionProvider" in patched.get_providers()
    paths = sorted(args.inputs.glob("*.npz"))[: args.limit]
    xs = [np.load(p)["Xfeat_normed"] for p in paths]
    assert xs, "No input events"
    for x in xs:
        assert x.ndim == 3 and x.shape[0] == 1 and x.shape[2] == 55 and np.isfinite(x).all()
    ref = []
    for i, x in enumerate(xs):
        ref.append(original.run(None, {"Xfeat_normed": x, "mask": np.ones(x.shape[:2], np.float32)}))
        print("reference", i, flush=True)

    results = {}
    for batch, pad in [(1, 0), (1, 512), (2, 512), (4, 512), (4, 6144)]:
        rows = []
        for start in range(0, len(xs), batch):
            vals = xs[start : start + batch]
            length = max(x.shape[1] for x in vals)
            if pad:
                length = ((length + pad - 1) // pad) * pad
            x = np.zeros((len(vals), length, 55), np.float32)
            mask = np.zeros(x.shape[:2], np.float32)
            for j, v in enumerate(vals):
                n = v.shape[1]
                x[j, :n] = v[0]
                mask[j, :n] = 1
            out = patched.run(None, {"Xfeat_normed": x, "mask": mask})
            for j, v in enumerate(vals):
                rows.append(compare([r[0] for r in ref[start + j]], [o[j, : v.shape[1]] for o in out]))
        results[f"B{batch}_pad{pad}"] = rows
        print(
            "case",
            batch,
            pad,
            "pidchanges",
            sum(r["active_pid_changes"] for r in rows),
            "max",
            max(r["id"]["max_abs"] for r in rows),
            flush=True,
        )
    report = {
        "ort": ort.__version__,
        "device": args.device,
        "providers": providers,
        "source_sha256": hashlib.sha256(args.original.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "inputs": [
            {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "length": x.shape[1],
            }
            for p, x in zip(paths, xs)
        ],
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert all(r[n]["finite"] for rows in results.values() for r in rows for n in ["bid", "id", "momentum", "pu"])


if __name__ == "__main__":
    main()
