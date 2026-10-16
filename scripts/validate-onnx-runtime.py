"""Replay exported MLPF batches on CPU, CUDA, or MIGraphX without PyTorch/TFDS."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import onnxruntime as ort


def configure_migraphx(options, provider_options):
    """Register AMD's plugin and explicitly select its first available device."""
    try:
        import onnxruntime_ep_migraphx as plugin

        names, paths = plugin.get_ep_names(), plugin.get_library_paths()
        if not names or len(names) != len(paths):
            raise RuntimeError("MIGraphX plugin package has missing provider libraries")
        for name, path in zip(names, paths, strict=True):
            ort.register_execution_provider_library(name, path)
        devices = [device for device in ort.get_ep_devices() if device.ep_name == "MIGraphXExecutionProvider"]
        if not devices:
            raise RuntimeError("MIGraphX plugin found no supported AMD device")
        options.add_provider_for_devices([devices[0]], provider_options)
    except Exception as error:
        raise RuntimeError(
            "Cannot initialize MIGraphX: use envs/ort-amd on a supported AMD GPU with ROCm 10.1.0 installed and its libraries discoverable"
        ) from error
    return {name: version(name) for name in ("onnxruntime-ep-migraphx", "migraphx", "migraphx-libs")}


def load_inputs(path, session):
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(bool)": np.bool_}
    with np.load(path, allow_pickle=False) as data:
        if not np.all((data["mask"] == 0) | (data["mask"] == 1)):
            raise ValueError(f"{path}: expected a binary mask")
        feeds = {i.name: data[i.name].astype(dtypes[i.type]) for i in session.get_inputs()}
    features, mask = feeds["Xfeat_normed"], feeds["mask"]
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError(f"{path}: expected features[B,L,F] and mask[B,L]")
    valid = mask.astype(bool)
    if not np.all((mask == 0) | (mask == 1)) or not np.all(valid.any(axis=-1)) or np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError(f"{path}: expected nonempty events with binary right padding")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite features")
    return feeds, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="NPZ batches from cms-validate-onnx.py --save-inputs")
    parser.add_argument("--provider", choices=["cpu", "cuda", "migraphx"], default="cpu")
    parser.add_argument("--provider-option", action="append", default=[], help="Execution-provider option key=value")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=3e-3)
    parser.add_argument("--require-full-gpu", action="store_true", help="Fail if the profile contains any CPU compute nodes")
    args = parser.parse_args()
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[args.provider]
    if args.provider == "cuda":
        ort.preload_dlls()
    if args.provider != "migraphx" and provider not in ort.get_available_providers():
        raise RuntimeError(f"{provider} is unavailable: {ort.get_available_providers()}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(args.outdir / "ort_profile")
    provider_options = dict(item.split("=", 1) for item in args.provider_option)
    plugin_versions = {}
    if args.provider == "migraphx":
        plugin_versions = configure_migraphx(options, provider_options)
        # Passing providers= here would override the plugin device selection.
        session = ort.InferenceSession(str(args.model), options)
    else:
        session = ort.InferenceSession(str(args.model), options, providers=[(provider, provider_options)])
    if provider not in session.get_providers():
        raise RuntimeError(f"Requested {provider}, session fell back to {session.get_providers()}")
    cpu_options = ort.SessionOptions()
    cpu_options.intra_op_num_threads = 1
    reference = ort.InferenceSession(str(args.model), cpu_options, providers=["CPUExecutionProvider"])
    results = []
    passed = True
    for path in args.inputs:
        feeds, valid = load_inputs(path, session)
        expected = reference.run(None, feeds)
        start = time.perf_counter()
        session.run(None, feeds)  # includes compilation for a new MIGraphX shape
        cold = time.perf_counter() - start
        start = time.perf_counter()
        actual = session.run(None, feeds)
        elapsed = time.perf_counter() - start
        errors = {}
        for output, target, value in zip(session.get_outputs(), expected, actual):
            target, value = target[valid].astype(np.float32), value[valid].astype(np.float32)
            finite = bool(np.isfinite(target).all() and np.isfinite(value).all())
            close = finite and bool(np.allclose(target, value, atol=args.atol, rtol=args.rtol))
            errors[output.name] = {"finite": finite, "close": close, "max_abs_error": float(np.max(np.abs(target - value))) if finite else None}
            passed &= close
        results.append(
            {
                "inputs": str(path),
                "shape": list(feeds["Xfeat_normed"].shape),
                "lengths": valid.sum(-1).tolist(),
                "first_call_seconds": cold,
                "warm_seconds": elapsed,
                "outputs": errors,
            }
        )
    profile = session.end_profiling()
    nodes = Counter(
        e.get("args", {}).get("provider")
        for e in json.loads(Path(profile).read_text())
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    )
    if args.require_full_gpu and (args.provider == "cpu" or nodes.get("CPUExecutionProvider", 0)):
        passed = False
    report = {
        "onnxruntime": ort.__version__,
        "plugin_versions": plugin_versions,
        "model": str(args.model),
        "provider": provider,
        "provider_options": session.get_provider_options(),
        "profile": profile,
        "profile_node_calls": dict(nodes),
        "passed": passed,
        "results": results,
    }
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
