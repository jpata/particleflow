"""Canonical attention exports, provider validation, timing, and graph evidence."""

import argparse
import collections
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto as T
from onnx import helper as h
from provenance import collect
from torch.nn.attention import SDPBackend, sdpa_kernel


class Attention(torch.nn.Module):
    def __init__(self, mode, explicit=False):
        super().__init__()
        self.mode = mode
        self.explicit = explicit

    def forward(self, query, key, value, mask=None):
        valid = None if mask is None else mask.bool()[:, None, None, :]
        if not self.explicit:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=valid)
        else:
            # Keep the attention matmuls in the input dtype, softmax in FP32.
            scores = (query * query.shape[-1] ** -0.25) @ (key.transpose(-1, -2) * key.shape[-1] ** -0.25)
            if valid is not None:
                has_keys = valid.any(dim=-1, keepdim=True)
                first = torch.arange(key.shape[-2], device=key.device) == 0
                safe = valid | (~has_keys & first)
                scores = scores.masked_fill(~safe, float("-inf"))
            probs = scores.float().softmax(dim=-1).to(query.dtype)
            out = probs @ value
            if valid is not None:
                out = out * has_keys.to(out.dtype)
        if mask is not None:
            out = out * mask[:, None, :, None].to(out.dtype)
        return out


def inputs(batch, seq, heads, dim, dtype, mode, balanced=False):
    rng = np.random.default_rng(12345 + seq)
    qkv = [rng.standard_normal((batch, heads, seq, dim)).astype(dtype) for _ in range(3)]
    if mode == "unmasked":
        return qkv
    mask = np.ones((batch, seq), np.int32)
    for i in range(batch):
        mask[i, max(1, seq - int(seq * 0.5 * (i + (0.5 if balanced else 1)) / batch)) :] = 0
    if mode == "empty":
        mask[0] = 0
    if mode == "holes":
        mask[:, 1::3] = 0
    return qkv + [mask]


def native(arrays, explicit=False, backend=SDPBackend.MATH, device="cpu"):
    module = Attention("masked" if len(arrays) == 4 else "unmasked", explicit).to(device)
    args = [torch.from_numpy(x).to(device) for x in arrays]
    with torch.no_grad(), sdpa_kernel(backend):
        return module(*args)


def attention24(path, arrays, lengths, dynamic=False):
    q = arrays[0]
    batch, heads, seq, dim = q.shape
    dt = T.FLOAT16 if q.dtype == np.float16 else T.FLOAT
    shape = ["batch", heads, "seq", dim] if dynamic else list(q.shape)
    items = [h.make_tensor_value_info(n, dt, shape) for n in ["query", "key", "value"]]
    nodes = []
    names = ["query", "key", "value"]
    if len(arrays) == 4:
        ms = ["batch", "seq"] if dynamic else [batch, seq]
        items.append(h.make_tensor_value_info("mask", T.INT32, ms))

        def const(name, values):
            nodes.append(h.make_node("Constant", [], [name], value=h.make_tensor(name, T.INT64, [len(values)], values)))

        if lengths:
            nodes.append(h.make_node("Cast", ["mask"], ["mask64"], to=T.INT64))
            const("sum_axis", [1])
            nodes.append(h.make_node("ReduceSum", ["mask64", "sum_axis"], ["lengths"], keepdims=0))
            names += ["", "", "", "lengths"]
        else:
            nodes.append(h.make_node("Cast", ["mask"], ["valid"], to=T.BOOL))
            const("mask_axes", [1, 2])
            nodes.append(h.make_node("Unsqueeze", ["valid", "mask_axes"], ["attn_mask"]))
            # ORT's Attention kernel requires the query axis to equal Q length.
            nodes.append(h.make_node("Shape", ["query"], ["qshape"]))
            const("batch_axis", [0])
            const("seq_axis", [2])
            const("one", [1])
            nodes.append(h.make_node("Gather", ["qshape", "batch_axis"], ["batch_dim"], axis=0))
            nodes.append(h.make_node("Gather", ["qshape", "seq_axis"], ["seq_dim"], axis=0))
            nodes.append(h.make_node("Concat", ["batch_dim", "one", "seq_dim", "seq_dim"], ["dense_shape"], axis=0))
            nodes.append(h.make_node("Expand", ["attn_mask", "dense_shape"], ["dense_mask"]))
            names += ["dense_mask"]
    nodes.append(
        h.make_node("Attention", names, ["raw" if len(arrays) == 4 else "output"], is_causal=0, softmax_precision=T.FLOAT, name="Attention24")
    )
    if len(arrays) == 4:
        nodes.append(h.make_node("Cast", ["mask"], ["mask_float"], to=dt))
        const("query_axes", [1, 3])
        nodes.append(h.make_node("Unsqueeze", ["mask_float", "query_axes"], ["query_mask"]))
        nodes.append(h.make_node("Mul", ["raw", "query_mask"], ["output"]))
    model = h.make_model(
        h.make_graph(nodes, "attention24_lengths" if lengths else "attention24_dense", items, [h.make_tensor_value_info("output", dt, shape)]),
        opset_imports=[h.make_opsetid("", 24)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def dump(path):
    model = onnx.load(path)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except onnx.shape_inference.InferenceError as exc:
        print(f"Shape inference unavailable for {path}: {exc}")
    path.with_suffix(".onnx.txt").write_text(h.printable_graph(model.graph))
    return dict(collections.Counter((n.domain + "::" if n.domain else "") + n.op_type for n in model.graph.node))


def export_bundle(args):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "export-provenance.json").write_text(json.dumps(collect(), indent=2))
    records = []
    for seq in args.seq:
        for dtype in ["fp32", "fp16"]:
            for mode in ["unmasked", "padding"]:
                arrays = inputs(args.batch, seq, 16, 16, np.float32 if dtype == "fp32" else np.float16, mode)
                for kind in ["sdpa_legacy", "explicit_safe", "attention24_dense", "attention24_lengths"]:
                    if kind == "attention24_lengths" and mode == "unmasked":
                        continue
                    variants = [False, True] if seq == 512 else [False]
                    for dynamic in variants:
                        record = {
                            "kind": kind,
                            "dtype": dtype,
                            "mode": mode,
                            "batch": args.batch,
                            "seq": seq,
                            "heads": 16,
                            "dim": 16,
                            "dynamic": dynamic,
                        }
                        path = args.out / (f"{kind}_{dtype}_{mode}_b{args.batch}_s{seq}" + ("_dynamic" if dynamic else "") + ".onnx")
                        record["model"] = path.name
                        try:
                            if kind.startswith("attention24"):
                                attention24(path, arrays, kind.endswith("lengths"), dynamic)
                            else:
                                module = Attention(mode, kind == "explicit_safe").eval()
                                names = ["query", "key", "value"] + (["mask"] if len(arrays) == 4 else [])
                                axes = {n: {0: "batch", 2: "seq"} for n in names[:3]}
                                axes["output"] = {0: "batch", 2: "seq"}
                                if "mask" in names:
                                    axes["mask"] = {0: "batch", 1: "seq"}
                                with sdpa_kernel(SDPBackend.MATH):
                                    torch.onnx.export(
                                        module,
                                        tuple(torch.from_numpy(x) for x in arrays),
                                        str(path),
                                        input_names=names,
                                        output_names=["output"],
                                        opset_version=20,
                                        dynamo=False,
                                        dynamic_axes=axes if dynamic else None,
                                    )
                            record.update(status="ok", nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        except Exception as e:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
                            record.update(status="failed", error=f"{type(e).__name__}: {e}")
                        records.append(record)
                # Modern exporter is an explicitly separate diagnostic, not a silent replacement.
                if seq == 512:
                    record = {
                        "kind": "sdpa_dynamo",
                        "dtype": dtype,
                        "mode": mode,
                        "batch": args.batch,
                        "seq": seq,
                        "heads": 16,
                        "dim": 16,
                        "dynamic": False,
                    }
                    path = args.out / f"sdpa_dynamo_{dtype}_{mode}_b{args.batch}_s{seq}.onnx"
                    record["model"] = path.name
                    try:
                        names = ["query", "key", "value"] + (["mask"] if len(arrays) == 4 else [])
                        with sdpa_kernel(SDPBackend.MATH):
                            torch.onnx.export(
                                Attention(mode),
                                tuple(torch.from_numpy(x) for x in arrays),
                                str(path),
                                input_names=names,
                                output_names=["output"],
                                opset_version=23,
                                dynamo=True,
                                external_data=False,
                            )
                        record.update(status="ok", nodes=dump(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    except Exception as e:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
                        record.update(status="failed", error=f"{type(e).__name__}: {e}")
                    records.append(record)
    (args.out / "manifest.json").write_text(json.dumps(records, indent=2))


def timing(fn, device, warmup, iters):
    first = time.perf_counter()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    first = time.perf_counter() - first
    for _ in range(warmup):
        out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    trials = []
    for trial in range(3):
        for _ in range(warmup):
            out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        group = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn()
            if device == "cuda":
                torch.cuda.synchronize()
            group.append((time.perf_counter() - t) * 1000)
        samples.extend(group)
        trials.append(group)
    return out, {
        "first_call_ms": first * 1000,
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
        "trial_samples_ms": trials,
        "trial_medians_ms": [statistics.median(x) for x in trials],
    }


def errors(out, ref, low_precision=False):
    input_dtype = np.asarray(out).dtype
    out = np.asarray(out, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    finite = bool(np.isfinite(out).all())
    diff = np.abs(out - ref)
    tol = 2e-3 if low_precision or input_dtype == np.float16 else 1e-4
    return {
        "finite": finite,
        "mae": float(np.mean(diff)) if finite else None,
        "max_abs": float(np.max(diff)) if finite else None,
        "atol": tol,
        "rtol": tol,
        "correct": bool(finite and np.allclose(out, ref, atol=tol, rtol=tol)),
    }


def check_edges(sess, r):
    result = {}
    for mode in ["empty", "holes"]:
        if mode == "holes" and r["kind"] == "attention24_lengths":
            result[mode] = {"status": "outside_contract"}
            continue
        dtype = np.float32 if r["dtype"] == "fp32" else np.float16
        arrays = inputs(r["batch"], r["seq"], r["heads"], r["dim"], dtype, mode)
        reference = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(arrays)]).numpy()
        try:
            output = sess.run(None, {i.name: a for i, a in zip(sess.get_inputs(), arrays)})[0]
            result[mode] = errors(output, reference, r["dtype"] == "fp16")
        except Exception as exc:  # noqa: BLE001 - retain unsupported runtime cases as evidence.
            result[mode] = {"error": str(exc)}
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[args.provider]
    device = "cpu" if args.provider == "cpu" else "cuda"
    if args.provider == "cuda":
        ort.preload_dlls()
    (args.out / "sources").mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, args.out / "sources" / source.name)
    (args.out / "provenance.json").write_text(json.dumps(collect(), indent=2))
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    rows = []
    seen = set()
    for r in manifest:
        if r["status"] != "ok":
            continue
        if args.quick and (r["seq"] != 512 or r["dynamic"]):
            continue
        arrays = inputs(
            r["batch"],
            r["seq"],
            r["heads"],
            r["dim"],
            np.float32 if r["dtype"] == "fp32" else np.float16,
            r["mode"],
            r.get("length_policy") == "balanced",
        )
        r = {**r, "lengths": arrays[3].sum(-1).tolist() if len(arrays) == 4 else [r["seq"]] * r["batch"]}
        r["valid_tokens"] = sum(r["lengths"])
        ref = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(arrays)]).numpy()
        key = (r["batch"], r["seq"], r["dtype"], r["mode"])
        if key not in seen:
            seen.add(key)
            for backend in [SDPBackend.MATH] if device == "cpu" else [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]:
                row = {
                    **r,
                    "kind": "pytorch_" + backend.name.lower(),
                    "dynamic": False,
                    "provider": args.provider,
                    "status": "ok",
                    "timing": "native",
                }
                for field in ["model", "sha256", "nodes"]:
                    row.pop(field, None)
                print("NATIVE", key, backend.name, flush=True)
                try:
                    tensors = [torch.from_numpy(x).to(device) for x in arrays]
                    module = Attention(r["mode"]).to(device)

                    def fn(module=module, tensors=tensors, backend=backend):
                        with torch.no_grad(), sdpa_kernel(backend):
                            return module(*tensors)

                    out, ts = timing(fn, device, args.warmup, args.iters)
                    row.update(ts)
                    row.update(errors(out.cpu().numpy(), ref, r["dtype"] == "fp16"))
                    if not row["correct"]:
                        row["status"] = "incorrect"
                except Exception as e:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
                    row.update(status="failed", error=f"{type(e).__name__}: {e}")
                rows.append(row)
        print("ORT", r["model"], flush=True)
        row = {**r, "provider": args.provider, "status": "ok", "timing": "host_io"}
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            if args.provider != "cpu":
                so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            else:
                so.optimized_model_filepath = str(args.out / (Path(r["model"]).stem + ".optimized.onnx"))
            opts = {}
            if args.provider == "cuda":
                opts["use_tf32"] = "0"
            t = time.perf_counter()
            sess = ort.InferenceSession(str(args.bundle / r["model"]), so, providers=[(provider, opts)])
            row["session_ms"] = (time.perf_counter() - t) * 1000
            sess.disable_fallback()
            row["session_providers"] = sess.get_providers()
            assert provider in sess.get_providers()
            row["output_type"] = sess.get_outputs()[0].type
            feeds = {i.name: a for i, a in zip(sess.get_inputs(), arrays)}
            out, ts = timing(lambda sess=sess, feeds=feeds: sess.run(None, feeds)[0], device, args.warmup, args.iters)
            row.update(ts)
            row.update(errors(out, ref, r["dtype"] == "fp16"))
            if not row["correct"]:
                row["status"] = "incorrect"
            if args.provider != "cpu":
                binding = sess.io_binding()
                values = []
                for name, a in feeds.items():
                    v = ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)
                    values.append(v)
                    binding.bind_ortvalue_input(name, v)
                output_dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}[sess.get_outputs()[0].type]
                vout = ort.OrtValue.ortvalue_from_shape_and_type(list(arrays[0].shape), output_dtype, "cuda", 0)
                binding.bind_ortvalue_output("output", vout)

                def bound(sess=sess, binding=binding):
                    sess.run_with_iobinding(binding)
                    binding.synchronize_outputs()

                _, ts = timing(bound, device, args.warmup, args.iters)
                row["device_io"] = ts
                row["device_io_errors"] = errors(vout.numpy(), ref, r["dtype"] == "fp16")
                if not row["device_io_errors"]["correct"]:
                    row["status"] = "incorrect"
            if r["seq"] == 512 and r["mode"] == "padding":
                row["edge_cases"] = check_edges(sess, r)
            # Verify representative graph placement separately, outside timed samples.
            if r["seq"] == 512:
                so.enable_profiling = True
                so.profile_file_prefix = str(args.out / Path(r["model"]).stem)
                if args.provider == "cuda":
                    so.optimized_model_filepath = str(args.out / (Path(r["model"]).stem + ".optimized.onnx"))
                prof = ort.InferenceSession(str(args.bundle / r["model"]), so, providers=[(provider, opts)])
                prof.disable_fallback()
                prof.run(None, feeds)
                pf = prof.end_profiling()
                row["profile_file"] = str(Path(pf).name)
                events = json.loads(Path(pf).read_text())
                row["node_placement"] = [
                    {"name": e["name"], "op": e["args"].get("op_name"), "provider": e["args"]["provider"]}
                    for e in events
                    if e.get("args", {}).get("provider")
                ]
                del prof
            del sess
        except Exception as e:  # noqa: BLE001 - preserve per-variant exporter/runtime failures.
            row.update(status="failed", error=f"{type(e).__name__}: {e}")
        rows.append(row)
        (args.out / "results.json").write_text(json.dumps(rows, indent=2))
    # Edge cases are correctness tests only and excluded from scaling curves.
    edge = []
    for mode in ["empty", "holes"]:
        for dt in [np.float32, np.float16]:
            a = inputs(2, 16, 2, 16, dt, mode)
            ref = native([x.astype(np.float32) if i < 3 else x for i, x in enumerate(a)]).numpy()
            out = native(a, explicit=True).float().numpy()
            edge.append({"mode": mode, "dtype": str(dt), "explicit_safe": errors(out, ref, dt == np.float16)})
    (args.out / "edge-correctness.json").write_text(json.dumps(edge, indent=2))
    (args.out / "provenance-after.json").write_text(json.dumps(collect(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["export", "run"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], default="cpu")
    p.add_argument("--seq", nargs="+", type=int, default=[128, 512, 1280, 2560])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    export_bundle(a) if a.action == "export" else run(a)
