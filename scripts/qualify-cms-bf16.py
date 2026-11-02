"""GPU PyTorch BF16 references for the paired CMS ONNX qualification.

Attention uses forced SDPA Flash with production jagged packing; HEPTv2 uses
its compiled FlexAttention with FP32 LSE. Timings include synchronous CPU/GPU
transfers and FP32 output conversion, and exclude file I/O and warmup.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time


spec = importlib.util.spec_from_file_location("cms_qualification", Path(__file__).with_name("qualify-cms-physics.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


@contextmanager
def record_attention_calls(model_type, records):
    import torch
    import mlpf.model.heptv2 as hept

    target = torch.nn.functional if model_type == "attention" else hept
    name = "scaled_dot_product_attention" if model_type == "attention" else "flex_attention"
    original = getattr(target, name)

    def observed(query, key, value, *args, **kwargs):
        assert all(t.dtype == torch.bfloat16 and t.is_cuda for t in (query, key, value))
        records.append(
            {
                "qkv_dtype": "torch.bfloat16",
                "device": str(query.device),
                "returns_lse": kwargs.get("return_lse", False) or getattr(kwargs.get("return_aux"), "lse", False),
            }
        )
        return original(query, key, value, *args, **kwargs)

    setattr(target, name, observed)
    try:
        yield
    finally:
        setattr(target, name, original)


def reference(args):
    import torch
    from mlpf.model.onnx_attention import fp32_layer_norms

    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, _ = q.load_model(args.root, args.model, args.threads, flash=True)
    model = model.cuda().eval()
    dest = args.outdir or args.root / "reference" / args.model
    dest.mkdir(parents=True, exist_ok=True)
    cases = q.selected_cases(args)
    metadata = {
        "model": args.model,
        "reference": "PyTorch BF16 SDPA Flash (jagged)" if args.model == "attention" else "PyTorch BF16 compiled FlexAttention + FP32 LSE",
        "precision": "full-model GPU BF16 autocast; FP32 LayerNorm; original FP32 input geometry; FP32 external outputs",
        "checkpoint_sha256": json.loads((args.root / "release.json").read_text())["models"][args.model]["checkpoint_sha256"],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python": sys.version,
        "host": platform.node(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threads": args.threads,
        "warmup": [],
        "persistent_autocast_weight_cache": True,
        "timing_scope": "batch-one synchronized CPU inputs to CPU FP32 outputs; includes transfers; excludes file I/O, compilation and physics",
    }

    def infer(feeds):
        x = torch.from_numpy(feeds["Xfeat_normed"]).cuda()
        mask = torch.from_numpy(feeds["mask"]).cuda()
        values = model(x, mask)
        return [v.float().cpu().numpy() for v in values]

    with fp32_layer_norms(model), torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        # Warm every static bucket before measuring any event. Compilation and
        # first-use costs are reported separately rather than amortized away.
        for slots in sorted({c["slots"] for c in cases}):
            feeds = q.feeds_for(args.root, next(c for c in cases if c["slots"] == slots))
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(6):
                infer(feeds)
            torch.cuda.synchronize()
            metadata["warmup"].append({"slots": slots, "calls": 6, "seconds_including_compile": time.perf_counter() - start})
            print(args.model, "PyTorch BF16 warmup complete", slots, flush=True)
        calls = []
        feeds = q.feeds_for(args.root, cases[0])
        with (
            record_attention_calls(args.model, calls),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
            ) as prof,
        ):
            infer(feeds)
        assert len(calls) == 3, calls
        trace = dest / "kernel-trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = sorted({e["name"] for e in events if e.get("cat") == "kernel"})
        if not kernels:
            raise RuntimeError("No GPU kernels retained for the PyTorch baseline")
        if args.model == "attention" and not any("flash" in e.key.lower() for e in prof.key_averages()):
            raise RuntimeError("Forced Flash baseline did not record a Flash operation")
        metadata["kernel_proof"] = {"attention_calls": calls, "gpu_kernels": kernels, "trace_sha256": q.digest(trace)}
        results = []
        for case in cases:
            feeds = q.feeds_for(args.root, case)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = infer(feeds)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            record = q.retain_outputs(dest, case, outputs, feeds, enforce_padding=False)
            record["inference_ms"] = elapsed_ms
            results.append(record)
            q.write_json(dest / "progress.json", {"completed": len(results), "expected": len(cases)})
            if len(results) % 32 == 0:
                print(args.model, metadata["reference"], len(results), "/", len(cases), flush=True)
        metadata["cases"] = results
        q.write_json(dest / "manifest.json", metadata)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["attention", "heptv2"], required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--bucket", type=int)
    p.add_argument("--case")
    p.add_argument("--outdir", type=Path)
    args = p.parse_args()
    args.root = args.root.resolve()
    reference(args)


if __name__ == "__main__":
    main()
