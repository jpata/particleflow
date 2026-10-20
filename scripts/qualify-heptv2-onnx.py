"""Export full-model reference bundles and qualify identical inputs on explicit providers."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

OUTPUTS = ["bid", "id", "momentum", "pu"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_bundle(args):
    import torch
    from mlpf.conf import MLPFConfig
    from mlpf.model.mlpf import MLPF
    from mlpf.model.onnx_attention import export_mlpf

    torch.set_num_threads(1)
    torch.manual_seed(42)
    block = 8 if args.small else 100
    config = MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": args.model_type,
            "model": {
                "type": args.model_type,
                "attention": {"num_convs": 3, "num_heads": 8, "head_dim": 16, "attention_type": "math"},
                "heptv2": {
                    "num_convs": 1 if args.small else 3,
                    "num_heads": 2 if args.small else 8,
                    "embedding_dim": 16 if args.small else 128,
                    "width": 16 if args.small else 512,
                    "block_size": block,
                    "n_hashes": 2 if args.small else 3,
                },
            },
        }
    )
    model = MLPF(config).eval()
    cases = []
    for batch, size in [(2, 2 * block), (1, 3 * block), (3, 4 * block)]:
        x = torch.randn(batch, size, config.input_dim)
        x[..., 0] = config.elemtypes_nonzero[0]
        x[..., 1] = x[..., 1].exp()
        x[..., 5] = x[..., 5].exp()
        lengths = [size - 1 - i * (block // 2) for i in range(batch)]
        mask = torch.arange(size)[None, :] < torch.tensor(lengths)[:, None]
        x[~mask] = 0
        cases.append((f"synthetic_b{batch}_s{size}", x, mask.float()))
    if args.inputs_bundle:
        cases = []
        source = json.loads((args.inputs_bundle / "manifest.json").read_text())
        for case in source["cases"]:
            with np.load(args.inputs_bundle / case["file"]) as data:
                cases.append((Path(case["file"]).stem, torch.from_numpy(data["Xfeat_normed"].copy()), torch.from_numpy(data["mask"].copy())))
    if args.data_dir:
        import tensorflow_datasets as tfds

        ds = tfds.builder(args.dataset, data_dir=args.data_dir).as_data_source(split="train")
        for index in range(args.num_events):
            x = torch.as_tensor(np.array(ds[index]["X"], copy=True), dtype=torch.float32)
            x = x[x[:, 0] != 0]
            if config.sort_data:
                x = x[torch.argsort(x[:, 1], descending=True)]
            if x.shape[1] > config.input_dim:
                raise ValueError("Dataset feature width exceeds model input dimension")
            length = len(x)
            size = ((length + block - 1) // block) * block
            x = torch.nn.functional.pad(x, (0, config.input_dim - x.shape[1], 0, size - length)).unsqueeze(0).clamp(-60000, 60000)
            mask = (torch.arange(size)[None, :] < length).float()
            cases.append((f"event_{index:04d}", x, mask))
    if args.sweep:
        source_x, source_mask = cases[-1][1:]
        pool = source_x[source_mask.bool()]
        cases = []
        for batch, size in [(1, n) for n in args.sweep] + [(b, args.batch_length) for b in args.batches]:
            valid_length = max(1, int(size * 0.9))
            x = pool[torch.arange(batch * size) % len(pool)].reshape(batch, size, -1).clone()
            mask = (torch.arange(size)[None, :] < valid_length).expand(batch, -1).clone()
            x[~mask] = 0
            cases.append((f"sweep_b{batch}_s{size}", x, mask.float()))
    export_mlpf(
        model,
        cases[0][1],
        cases[0][2],
        args.bundle / "model.onnx",
        attention_format=args.attention_format,
        attention_dtype=torch.float16 if args.attention_dtype == "float16" else None,
    )
    manifest = {
        "scope": f"Export correctness with seeded untrained {args.model_type} weights; not physics performance",
        "seed": 42,
        "torch": torch.__version__,
        "export_profile": {"attention_format": args.attention_format, "attention_dtype": args.attention_dtype},
        "config": config.model_dump(mode="json"),
        "model_sha256": digest(args.bundle / "model.onnx"),
        "cases": [],
    }
    for name, x, mask in cases:
        with torch.no_grad():
            reference = model(x, mask)
        path = args.bundle / f"{name}.npz"
        np.savez(path, Xfeat_normed=x.numpy(), mask=mask.numpy(), **{key: value.numpy() for key, value in zip(OUTPUTS, reference, strict=True)})
        manifest["cases"].append({"file": path.name, "sha256": digest(path), "shape": list(x.shape), "lengths": mask.sum(-1).int().tolist()})
    (args.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def specialize_model(source, destination, feeds):
    import onnx

    graph = onnx.load(source)
    for value in graph.graph.input:
        for dim, size in zip(value.type.tensor_type.shape.dim, feeds[value.name].shape, strict=True):
            dim.dim_value = size
    for value in graph.graph.output:
        for dim, size in zip(value.type.tensor_type.shape.dim[:2], feeds["mask"].shape, strict=True):
            dim.dim_value = size
    del graph.graph.value_info[:]
    onnx.save(onnx.shape_inference.infer_shapes(graph), destination)


def run_bundle(args):
    import onnxruntime as ort
    from mlpf.model.onnx_runtime import create_session, finish_profile

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    assert digest(args.bundle / "model.onnx") == manifest["model_sha256"]
    report = {
        "provider": args.provider,
        "ort": ort.__version__,
        "python": platform.python_version(),
        "model_sha256": manifest["model_sha256"],
        "scope": manifest["scope"],
        "cases": [],
        "passed": True,
        "compiler_environment": {
            key: os.environ.get(key)
            for key in (
                "MIGRAPHX_DISABLE_PASSES",
                "MIGRAPHX_GPU_COMPILE_PARALLEL",
                "MIGRAPHX_GPU_OPTIMIZE",
                "MIGRAPHX_GPU_HIP_FLAGS",
                "MIGRAPHX_MLIR_USE_SPECIFIC_OPS",
            )
        },
    }
    session = None
    profiles = []
    for case in manifest["cases"]:
        path = args.bundle / case["file"]
        assert digest(path) == case["sha256"]
        with np.load(path) as data:
            feeds = {name: data[name] for name in ["Xfeat_normed", "mask"]}
            valid = feeds["mask"].astype(bool)
            if session is None:
                model_path = args.bundle / "model.onnx"
                if args.static_shapes:
                    model_path = args.outdir / (path.stem + ".onnx")
                    specialize_model(args.bundle / "model.onnx", model_path, feeds)
                session = create_session(
                    model_path,
                    args.provider,
                    profile_prefix=args.outdir / path.stem,
                    require_full_gpu=args.provider != "cpu" and not args.allow_cpu_fallback,
                )
            start = time.perf_counter()
            session.run(OUTPUTS, feeds)
            cold = time.perf_counter() - start
            times = []
            for _ in range(3):
                start = time.perf_counter()
                values = session.run(OUTPUTS, feeds)
                times.append(time.perf_counter() - start)
            errors = {}
            for name, value in zip(OUTPUTS, values, strict=True):
                target = data[name]
                close = np.isfinite(value).all() and np.allclose(value[valid], target[valid], atol=1e-3, rtol=3e-3)
                close = bool(close and np.all(value[~valid] == 0))
                errors[name] = {
                    "close": close,
                    "max_abs": float(np.max(abs(value[valid] - target[valid]))),
                    "mae": float(np.mean(abs(value[valid] - target[valid]))),
                }
                report["passed"] &= close
            report["cases"].append({**case, "first_seconds": cold, "warm_median_seconds": float(np.median(times)), "outputs": errors})
            print(case["file"], errors, flush=True)
            if args.static_shapes:
                profiles.append(finish_profile(session, args.provider, require_full_gpu=args.provider != "cpu" and not args.allow_cpu_fallback))
                session = None
    if session is not None:
        profiles.append(finish_profile(session, args.provider, require_full_gpu=args.provider != "cpu" and not args.allow_cpu_fallback))
    report["placement"] = profiles
    report["static_shapes"] = args.static_shapes
    report["qualified"] = report["passed"] and (args.provider == "cpu" or all(p["full_provider_placement"] for p in profiles))
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "run"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], default="cpu")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Diagnostic placement run; does not establish full GPU qualification")
    parser.add_argument("--static-shapes", action="store_true", help="Specialize the common graph to each recorded input shape")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--model-type", choices=["heptv2", "attention"], default="heptv2")
    parser.add_argument("--attention-format", choices=["decomposed", "attention24"], default="decomposed")
    parser.add_argument("--attention-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--sweep", type=int, nargs="+")
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--batch-length", type=int, default=1000)
    parser.add_argument("--inputs-bundle", type=Path, help="Reuse recorded inputs while regenerating model/reference outputs")
    parser.add_argument("--data-dir")
    parser.add_argument("--dataset", default="cms_pf_ttbar/1:3.0.0")
    parser.add_argument("--num-events", type=int, default=3)
    args = parser.parse_args()
    if args.model_type == "heptv2" and (args.attention_format != "decomposed" or args.attention_dtype != "float32"):
        parser.error("HEPTv2 qualification uses decomposed FP32; attention profiles apply to the attention model")
    if args.sweep and any(size <= 0 for size in [*args.sweep, *args.batches, args.batch_length]):
        parser.error("Sweep dimensions must be positive")
    args.bundle.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        export_bundle(args)
    else:
        if args.outdir is None:
            parser.error("run requires --outdir")
        run_bundle(args)


if __name__ == "__main__":
    main()
