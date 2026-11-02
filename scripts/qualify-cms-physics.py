"""Prepare a unique CMS test sample and retain reference/ONNX outputs for physics closure.

Run prepare, export, reference and run in that order. Each static runtime shape
runs in a fresh subprocess. Physics analysis consumes the retained raw heads,
uses production particle decoding, and clusters every collection identically.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

HEADS = ("bid", "id", "momentum", "pu")
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def config_for(root, model_type):
    from mlpf.conf import AttentionType, MLPFConfig

    value = json.loads((root / "checkpoints" / model_type / "hyperparameters.json").read_text())
    value.pop("num_mlpf_params", None)
    config = MLPFConfig.model_validate(value)
    if config.dataset.value != "cms" or config.model.output_mode.value != "elementwise" or config.model.type.value != model_type:
        raise ValueError("This qualification requires the selected elementwise CMS model")
    if config.model.attention:
        config.model.attention.attention_type = AttentionType.MATH
        config.model.attention.use_jagged_attention = False
        config.model.attention.use_flash_attn_varlen = False
    return config


def load_model(root, model_type, threads, *, flash=False):
    import torch
    from mlpf.model.mlpf import MLPF

    torch.set_num_threads(threads)
    config = config_for(root, model_type)
    if flash and model_type == "attention":
        from mlpf.conf import AttentionType

        config.model.attention.attention_type = AttentionType.FLASH
        config.model.attention.use_jagged_attention = True
        config.model.attention.export_onnx_fused = False
    release = json.loads((root / "release.json").read_text())["models"][model_type]
    checkpoint = root / "checkpoints" / model_type / Path(release["checkpoint"]).name
    assert digest(checkpoint) == release["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    model = MLPF(config).float().eval()
    model.load_state_dict({k: v for k, v in state.items() if not k.startswith("task_loss_weighter.")}, strict=True)
    return model, config


def prepare(args):
    import tensorflow_datasets as tfds
    from mlpf.model.PFDataset import TFDSDataSource

    directory = args.data_dir / "cms_pf_ttbar/1/2.8.0"
    builder = tfds.builder_from_directory(str(directory))
    raw = builder.as_data_source(split="test")
    configs = [config_for(args.root, name) for name in ["attention", "heptv2"]]
    assert configs[0].input_dim == configs[1].input_dim
    assert configs[0].sort_data == configs[1].sort_data
    source = TFDSDataSource(raw, sort=configs[0].sort_data, pad_to_multiple=100, feature_dim=configs[0].input_dim, max_open_readers=8)
    destination = args.root / "inputs"
    destination.mkdir(parents=True, exist_ok=True)
    seen_records, seen_events, cases = set(), set(), []
    for index in np.random.default_rng(args.seed).permutation(len(raw)):
        record = raw.data_source.__getitems__([int(index)])[0]
        record_hash = hashlib.sha256(record).hexdigest()
        if record_hash in seen_records:
            continue
        seen_records.add(record_hash)
        event = source[int(index)]
        valid = event["X"][:, 0] != 0
        x = event["X"][valid].astype(np.float32)
        # Content identity also rejects alternate serializations of the same event.
        identity = hashlib.sha256()
        for key in ["X", "ytarget", "ycand", "genjets", "genmet"]:
            array = np.ascontiguousarray(event[key])
            identity.update(key.encode() + str(array.shape).encode() + str(array.dtype).encode() + array.tobytes())
        event_hash = identity.hexdigest()
        if event_hash in seen_events:
            continue
        seen_events.add(event_hash)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f"Invalid input event {index}")
        target = event["ytarget"][valid].copy()
        target[:, 2] = event["ytarget_pt_orig"][valid]
        target[:, 6] = event["ytarget_e_orig"][valid]
        file = f"event_{len(cases):04d}.npz"
        np.savez(destination / file, X=x, target=target, cand=event["ycand"][valid], genjets=event["genjets"], genmet=event["genmet"])
        cases.append(
            {
                "file": file,
                "index": int(index),
                "record_sha256": record_hash,
                "event_sha256": event_hash,
                "sha256": digest(destination / file),
                "length": len(x),
                "slots": int(np.ceil(len(x) / args.bucket_width) * args.bucket_width),
            }
        )
        if len(cases) % 64 == 0:
            print("Prepared", len(cases), "unique events", flush=True)
        if len(cases) == args.events:
            break
    if len(cases) != args.events:
        raise RuntimeError(f"Only {len(cases)} unique events available")
    manifest = {
        "dataset": "cms_pf_ttbar/1:2.8.0",
        "split": "test",
        "dataset_entries": len(raw),
        "dataset_info_sha256": digest(directory / "dataset_info.json"),
        "seed": args.seed,
        "preprocessing": "Production TFDSDataSource including HO correction and CMS target mapping",
        "sort_data": configs[0].sort_data,
        "feature_dim": configs[0].input_dim,
        "bucket_width": args.bucket_width,
        "cases": cases,
    }
    write_json(args.root / "inputs/manifest.json", manifest)
    print("Prepared", len(cases), "events; buckets", sorted({c["slots"] for c in cases}), flush=True)


def feeds_for(root, case):
    path = root / "inputs" / case["file"]
    assert digest(path) == case["sha256"]
    with np.load(path) as data:
        x = data["X"].copy()
    shape = (1, case["slots"], x.shape[1])
    features = np.zeros(shape, dtype=np.float32)
    features[0, : len(x)] = x
    mask = np.zeros(shape[:2], dtype=np.float32)
    mask[0, : len(x)] = 1
    return {"Xfeat_normed": features, "mask": mask}


def selected_cases(args):
    cases = json.loads((args.root / "inputs/manifest.json").read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["file"] == args.case]
    if args.limit:
        cases = cases[: args.limit]
    return [c for c in cases if args.bucket is None or c["slots"] == args.bucket]


def export(args):
    import torch
    from mlpf.model.onnx_attention import export_mlpf

    model, config = load_model(args.root, args.model, args.threads)
    feeds = feeds_for(args.root, selected_cases(args)[0])
    dest = args.root / "models" / args.model / args.profile
    dest.mkdir(parents=True, exist_ok=True)
    format = "attention24" if args.profile.startswith("attention24-") else "decomposed"
    bf16 = args.profile.endswith("bf16")
    if args.model != "attention" and args.profile != "fp32" and not bf16:
        raise ValueError("HEPTv2 qualification uses FP32")
    device = "cuda" if bf16 else "cpu"
    model = model.to(device)
    export_mlpf(
        model,
        torch.from_numpy(feeds["Xfeat_normed"]).to(device),
        torch.from_numpy(feeds["mask"]).to(device),
        dest / "model.onnx",
        attention_format=format,
        attention_dtype=torch.bfloat16 if bf16 else torch.float16 if "fp16" in args.profile else None,
        compute_dtype=torch.bfloat16 if bf16 else None,
    )
    write_json(
        dest / "manifest.json",
        {
            "model": args.model,
            "profile": args.profile,
            "model_sha256": digest(dest / "model.onnx"),
            "config": config.model_dump(mode="json"),
            "checkpoint_sha256": json.loads((args.root / "release.json").read_text())["models"][args.model]["checkpoint_sha256"],
            "compute_dtype": "BF16 CUDA autocast, FP32 LayerNorm, FP32 external I/O" if bf16 else "existing precision policy",
            "export_device": device,
        },
    )
    print("Exported", args.model, args.profile, flush=True)


def retain_outputs(dest, case, outputs, feeds, enforce_padding=True):
    dest.mkdir(parents=True, exist_ok=True)
    length = case["length"]
    zero = all(np.all(value[:, length:] == 0) for value in outputs)
    finite = all(np.isfinite(value).all() for value in outputs)
    if not finite or (enforce_padding and not zero):
        raise RuntimeError(f"Invalid outputs in {case['file']}: finite={finite}, padding_zero={zero}")
    np.savez(dest / case["file"], **{k: value[0, :length] for k, value in zip(HEADS, outputs, strict=True)})
    return {
        "case": case["file"],
        "input_sha256": case["sha256"],
        "output_sha256": digest(dest / case["file"]),
        "finite": finite,
        "padding_zero": zero,
        "slots": feeds["mask"].shape[1],
    }


def reference(args):
    import torch

    model, _ = load_model(args.root, args.model, args.threads)
    dest = args.outdir or args.root / "reference" / args.model
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for case in selected_cases(args):
        feeds = feeds_for(args.root, case)
        start = time.monotonic()
        with torch.inference_mode():
            values = model(torch.from_numpy(feeds["Xfeat_normed"]), torch.from_numpy(feeds["mask"]))
        results.append(retain_outputs(dest, case, [v.numpy() for v in values], feeds, enforce_padding=False))
        if len(results) % 16 == 0:
            print(args.model, "CPU FP32 reference", len(results), "last seconds", round(time.monotonic() - start, 2), flush=True)
        write_json(dest / "progress.json", {"completed": len(results), "expected": len(selected_cases(args))})
    write_json(
        dest / "manifest.json",
        {
            "model": args.model,
            "reference": "Independent PyTorch CPU FP32 MATH",
            "checkpoint_sha256": json.loads((args.root / "release.json").read_text())["models"][args.model]["checkpoint_sha256"],
            "torch": torch.__version__,
            "python": sys.version,
            "host": platform.node(),
            "script_sha256": SOURCE_SHA256,
            "threads": args.threads,
            "cases": results,
        },
    )


def specialize(source, dest, feeds):
    import onnx

    graph = onnx.load(source)
    for value in graph.graph.input:
        for dim, size in zip(value.type.tensor_type.shape.dim, feeds[value.name].shape, strict=True):
            dim.dim_value = size
    for value in graph.graph.output:
        for dim, size in zip(value.type.tensor_type.shape.dim[:2], feeds["mask"].shape, strict=True):
            dim.dim_value = size
    del graph.graph.value_info[:]
    onnx.save(onnx.shape_inference.infer_shapes(graph), dest)


def migraphx_environment(model, profile, fusion):
    # Reduced-precision profiles explicitly enable or disable attention fusion.
    settings = {
        "MIGRAPHX_DISABLE_MLIR": "0",
        "MIGRAPHX_MLIR_USE_SPECIFIC_OPS": "!attention",
        "MIGRAPHX_DISABLE_PASSES": "",
        "MIGRAPHX_GPU_HIP_FLAGS": "",
    }
    if model == "heptv2":
        settings.update(
            MIGRAPHX_DISABLE_PASSES="eliminate_data_type",
            MIGRAPHX_GPU_COMPILE_PARALLEL="4",
            MIGRAPHX_GPU_OPTIMIZE="3",
            MIGRAPHX_GPU_HIP_FLAGS="-mllvm -enable-gvn-memdep=false",
        )
    if profile != "fp32" and fusion == "on":
        settings["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = "attention"
    return settings


def run(args):
    cases = selected_cases(args)
    dest = args.outdir or args.root / "results" / args.model / args.provider / args.profile / args.fusion
    dest.mkdir(parents=True, exist_ok=True)
    if args.bucket is None:
        for bucket in sorted({c["slots"] for c in cases}):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--root",
                str(args.root),
                "--model",
                args.model,
                "--profile",
                args.profile,
                "--provider",
                args.provider,
                "--fusion",
                args.fusion,
                "--bucket",
                str(bucket),
                "--outdir",
                str(dest),
                "--threads",
                str(args.threads),
            ]
            if args.limit:
                command += ["--limit", str(args.limit)]
            if args.case:
                command += ["--case", args.case]
            if args.cpu_basic:
                command += ["--cpu-basic"]
            with (dest / f"bucket_{bucket}.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            write_json(dest / f"bucket_{bucket}.exit.json", {"exit_code": result.returncode})
            if result.returncode:
                raise RuntimeError(f"Bucket {bucket} failed with code {result.returncode}; see {dest}/bucket_{bucket}.log")
            print(args.model, args.provider, args.profile, args.fusion, "completed bucket", bucket, flush=True)
        return
    if args.provider == "migraphx":
        # MIGraphX caches some environment settings when its libraries load.
        os.environ.update(migraphx_environment(args.model, args.profile, args.fusion))
    import onnxruntime as ort
    from mlpf.model.onnx_runtime import create_session, finish_profile

    source = args.root / "models" / args.model / args.profile
    graph_manifest = json.loads((source / "manifest.json").read_text())
    assert digest(source / "model.onnx") == graph_manifest["model_sha256"]
    feeds = feeds_for(args.root, cases[0])
    static = dest / f"bucket_{args.bucket}.onnx"
    specialization_start = time.perf_counter()
    specialize(source / "model.onnx", static, feeds)
    specialization_seconds = time.perf_counter() - specialization_start
    session_start = time.perf_counter()
    if args.provider == "cpu" and args.cpu_basic:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = args.threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_mem_pattern = False
        options.enable_profiling = True
        options.profile_file_prefix = str(dest / f"bucket_{args.bucket}")
        session = ort.InferenceSession(str(static), options, providers=["CPUExecutionProvider"])
        session.disable_fallback()
    elif args.provider == "cuda" and args.profile.startswith("attention24-") and args.fusion == "off":
        ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = args.threads
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.enable_profiling = True
        options.profile_file_prefix = str(dest / f"bucket_{args.bucket}")
        session = ort.InferenceSession(str(static), options, providers=[("CUDAExecutionProvider", {"use_tf32": "0", "sdpa_kernel": "16"})])
        session.disable_fallback()
    else:
        session = create_session(
            static,
            args.provider,
            threads=args.threads,
            require_full_gpu=args.provider != "cpu",
            profile_prefix=dest / f"bucket_{args.bucket}",
            flash=args.provider == "cuda" and args.profile.startswith("attention24-"),
        )
    session_creation_seconds = time.perf_counter() - session_start
    first_run_start = time.perf_counter()
    session.run(list(HEADS), feeds)
    first_run_seconds = time.perf_counter() - first_run_start
    placement = finish_profile(session, args.provider, require_full_gpu=args.provider != "cpu")
    warmup_calls = 1
    warmup_start = time.perf_counter()
    if args.profile.endswith("bf16"):
        for _ in range(5):
            session.run(list(HEADS), feeds)
        warmup_calls += 5
    extra_warmup_seconds = time.perf_counter() - warmup_start
    results = []
    for case in cases:
        feeds = feeds_for(args.root, case)
        start = time.perf_counter()
        outputs = session.run(list(HEADS), feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000
        results.append(retain_outputs(dest, case, outputs, feeds))
        results[-1]["inference_ms"] = elapsed_ms
        if len(results) % 32 == 0:
            print(args.model, args.provider, args.bucket, len(results), "/", len(cases), flush=True)
    write_json(
        dest / f"bucket_{args.bucket}.json",
        {
            "provider": args.provider,
            "ort": ort.__version__,
            "graph_optimization_level": str(session.get_session_options().graph_optimization_level),
            "memory_pattern_enabled": session.get_session_options().enable_mem_pattern,
            "python": sys.version,
            "numpy": np.__version__,
            "threads": args.threads,
            "host": platform.node(),
            "script_sha256": SOURCE_SHA256,
            "model_sha256": graph_manifest["model_sha256"],
            "static_sha256": digest(static),
            "placement": placement,
            "cases": results,
            "timing": {
                "scope": "batch-one synchronous session.run, CPU inputs/outputs; includes transfers; excludes loading, saving, compilation and physics",
                "warmup_calls": warmup_calls,
                "profile_disabled_during_timing": True,
                "specialization_seconds": specialization_seconds,
                "session_creation_seconds": session_creation_seconds,
                "first_run_seconds": first_run_seconds,
                "extra_warmup_seconds": extra_warmup_seconds,
            },
            "environment": {k: v for k, v in os.environ.items() if k.startswith(("MIGRAPHX_", "ORT_"))},
            "loaded_libraries": sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/" in line and any(key in line for key in ["migraphx", "rocblas", "hipblas", "amdhip", "amd_comgr"])
                }
            ),
        },
    )
    del session
    gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "export", "reference", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--events", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--bucket-width", type=int, default=1000)
    p.add_argument("--bucket", type=int)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--case", help="Select one retained event for diagnosis")
    p.add_argument("--cpu-basic", action="store_true", help="Use basic CPU graph optimization without memory-pattern reuse")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--model", choices=["attention", "heptv2"], default="attention")
    p.add_argument("--profile", choices=["fp32", "decomposed-fp16", "attention24-fp16", "decomposed-bf16", "attention24-bf16"], default="fp32")
    p.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], default="cpu")
    p.add_argument("--fusion", choices=["on", "off"], default="on")
    p.add_argument("--outdir", type=Path)
    args = p.parse_args()
    if args.events < 1 or args.bucket_width < 100 or args.bucket_width % 100:
        p.error("events must be positive; bucket width must be a positive multiple of 100")
    args.root = args.root.resolve()
    if args.command == "prepare" and args.data_dir is None:
        p.error("prepare requires --data-dir")
    if args.threads < 1 or args.limit < 0:
        p.error("threads must be positive and limit must be nonnegative")
    if args.command != "prepare" and not selected_cases(args):
        p.error("No retained events match the requested selection")
    {"prepare": prepare, "export": export, "reference": reference, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
