"""Batch throughput on retained real CMS inputs, in isolated shape processes."""

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def qualification():
    spec = importlib.util.spec_from_file_location("qualification", Path(__file__).with_name("qualify-cms-physics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    q = qualification()
    profile = "attention24-bf16" if args.provider == "cuda" else "decomposed-bf16"
    if args.provider == "migraphx" and args.engine == "onnx":
        os.environ.update(q.migraphx_environment(args.model, profile, "on"))
    manifest = json.loads((args.root / "input-manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c["slots"] == args.slots][:64]
    if len(cases) < 64:
        raise ValueError("Need 64 distinct events in the selected bucket")
    inputs = [q.feeds_for(args.root, c) for c in cases]
    batches = [{k: np.concatenate([x[k] for x in inputs[i : i + args.batch]], axis=0) for k in inputs[0]} for i in range(0, len(inputs), args.batch)]
    prefix = args.outdir / f"{args.model}-{args.engine}-s{args.slots}-b{args.batch}"
    metadata = {
        "model": args.model,
        "engine": args.engine,
        "provider": args.provider,
        "slots": args.slots,
        "batch": args.batch,
        "host": platform.node(),
        "python": sys.version,
        "source_sha256": q.digest(Path(__file__)),
        "events": [{"case": c["file"], "input_sha256": c["sha256"], "length": c["length"]} for c in cases],
        "scope": "synchronous CPU FP32 inputs to CPU FP32 outputs, includes transfers/conversion; excludes input preparation, compilation, warmup, validation",
        "threads": 4,
        "warmup_calls": 6,
        "rounds": args.rounds,
    }
    guard = contextlib.ExitStack()
    with guard:
        if args.engine == "onnx":
            import onnxruntime as ort

            from mlpf.model.onnx_runtime import create_session, finish_profile

            source = args.root / "models" / args.model / profile / "model.onnx"
            static = prefix.with_suffix(".onnx")
            q.specialize(source, static, batches[0])
            start = time.perf_counter()
            session = create_session(
                static,
                args.provider,
                threads=4,
                require_full_gpu=True,
                profile_prefix=prefix,
                flash=args.provider == "cuda",
            )
            session.run(list(q.HEADS), batches[0])
            metadata.update(
                ort=ort.__version__,
                model_sha256=q.digest(source),
                static_sha256=q.digest(static),
            )
            metadata["placement"] = finish_profile(session, args.provider, require_full_gpu=True)
            metadata["session_and_first_run_seconds"] = time.perf_counter() - start

            def infer(feeds):
                return session.run(list(q.HEADS), feeds)

            baseline_dir = args.root / "results" / args.model / args.provider / profile / "on"
        else:
            import torch

            from mlpf.model.onnx_attention import fp32_layer_norms

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model, _ = q.load_model(args.root, args.model, 4, flash=True)
            model = model.cuda().eval()
            guard.enter_context(fp32_layer_norms(model))
            guard.enter_context(torch.inference_mode())
            guard.enter_context(torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True))
            metadata.update(
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),
                hip=torch.version.hip,
                cuda=torch.version.cuda,
            )
            metadata["checkpoint_sha256"] = json.loads((args.root / "release.json").read_text())["models"][args.model]["checkpoint_sha256"]

            def infer(feeds):
                values = model(
                    torch.from_numpy(feeds["Xfeat_normed"]).cuda(),
                    torch.from_numpy(feeds["mask"]).cuda(),
                )
                outputs = [v.float().cpu().numpy() for v in values]
                torch.cuda.synchronize()
                return outputs

            baseline_dir = args.root / ("reference" if args.provider == "cuda" else "reference-amd") / args.model

        start = time.perf_counter()
        for _ in range(6):
            outputs = infer(batches[0])
        metadata["warmup_seconds"] = time.perf_counter() - start
        # Retain numerical diagnostics against the original batch-one outputs.
        # BF16 kernels can change rounding with batch size; do not claim equivalence.
        differences = {head: 0.0 for head in q.HEADS}
        baseline_available = True
        for i, case in enumerate(cases[: args.batch]):
            path = baseline_dir / case["file"]
            if not path.exists():
                baseline_available = False
                continue
            with np.load(path) as reference:
                for head, output in zip(q.HEADS, outputs):
                    actual = output[i, : case["length"]]
                    if actual.shape != reference[head].shape:
                        raise ValueError(f"Output shape mismatch for {head}")
                    differences[head] = max(
                        differences[head],
                        float(np.max(np.abs(actual - reference[head]))),
                    )
        metadata["batch_one_comparison"] = {
            "all_baselines_available": baseline_available,
            "max_abs_by_head": differences,
        }
        samples, round_rates = [], []
        for round_index in range(args.rounds):
            elapsed = []
            # Rotate batch order between rounds, without changing membership.
            for j in range(len(batches)):
                feeds = batches[(j + round_index) % len(batches)]
                start = time.perf_counter()
                outputs = infer(feeds)
                ms = (time.perf_counter() - start) * 1000
                elapsed.append(ms)
                samples.append(
                    {
                        "round": round_index,
                        "batch_index": (j + round_index) % len(batches),
                        "ms": ms,
                    }
                )
                if not all(np.isfinite(v).all() for v in outputs):
                    raise ValueError("Nonfinite outputs")
                if args.engine == "onnx" and not all(np.all(v[feeds["mask"] == 0] == 0) for v in outputs):
                    raise ValueError("Nonzero padded ONNX output")
            round_rates.append(1000 * len(cases) / sum(elapsed))
        latencies = np.array([s["ms"] for s in samples])
        metadata.update(
            status="ok",
            timed_batches=len(samples),
            timed_events=len(samples) * args.batch,
            events_per_second=1000 * args.batch / float(latencies.mean()),
            mean_batch_ms=float(latencies.mean()),
            median_batch_ms=float(np.median(latencies)),
            p95_batch_ms=float(np.quantile(latencies, 0.95)),
            amortized_ms_per_event=float(latencies.mean()) / args.batch,
            round_events_per_second=round_rates,
            samples=samples,
        )
        q.write_json(prefix.with_suffix(".json"), metadata)
        print(
            json.dumps(
                {
                    k: metadata[k]
                    for k in [
                        "model",
                        "engine",
                        "slots",
                        "batch",
                        "events_per_second",
                        "median_batch_ms",
                    ]
                }
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--provider", choices=["cuda", "migraphx"], required=True)
    p.add_argument("--models", nargs="+", default=["attention", "heptv2"])
    p.add_argument("--engines", nargs="+", default=["onnx", "native"])
    p.add_argument("--lengths", nargs="+", type=int, default=[4000, 5000])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--model")
    p.add_argument("--engine")
    p.add_argument("--slots", type=int)
    p.add_argument("--batch", type=int)
    args = p.parse_args()
    if args.rounds < 1 or any(b not in [1, 2, 4, 8, 16, 32, 64] for b in args.batches):
        p.error("rounds must be positive; batch sizes must divide 64")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    results = []
    for model in args.models:
        for engine in args.engines:
            for slots in args.lengths:
                for batch in args.batches:
                    prefix = args.outdir / f"{model}-{engine}-s{slots}-b{batch}"
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--root",
                        str(args.root),
                        "--outdir",
                        str(args.outdir),
                        "--provider",
                        args.provider,
                        "--model",
                        model,
                        "--engine",
                        engine,
                        "--slots",
                        str(slots),
                        "--batch",
                        str(batch),
                        "--rounds",
                        str(args.rounds),
                    ]
                    with prefix.with_suffix(".log").open("w") as log:
                        result = subprocess.run(command, check=False, stdout=log, stderr=subprocess.STDOUT)
                    row = {
                        "model": model,
                        "engine": engine,
                        "slots": slots,
                        "batch": batch,
                        "exit_code": result.returncode,
                    }
                    if result.returncode == 0:
                        row.update(json.loads(prefix.with_suffix(".json").read_text()))
                        row.pop("samples", None)
                        row.pop("events", None)
                    else:
                        row.update(status="failed", log=str(prefix.with_suffix(".log")))
                    results.append(row)
                    qualification().write_json(args.outdir / "summary.json", results)
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
