"""Compare identical FP16 ONNX graphs with attention fusion enabled and disabled.

Each worker runs in a fresh process. CUDA selects sdpa_kernel=1 or 16;
MIGraphX enables or rejects MLIR attention while leaving MLIR GEMMs enabled.
Retain compiler logs and kernel traces to verify actual execution before
attributing errors to fusion.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

OUTPUTS = ["bid", "id", "momentum", "pu"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(actual, expected, valid):
    result = {}
    for name in OUTPUTS:
        value, target = actual[name], expected[name]
        a, b = value[valid].astype(np.float64), target[valid].astype(np.float64)
        delta = abs(a - b)
        result[name] = {
            "max_abs": float(delta.max()),
            "mae": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            "close": bool(np.allclose(a, b, atol=1e-3, rtol=3e-3)),
            "violating_elements": int(np.count_nonzero(delta > 1e-3 + 3e-3 * abs(b))),
            "finite": bool(np.isfinite(value).all()),
            "padding_zero": bool(np.all(value[~valid] == 0)),
            "argmax_disagreements": int(np.count_nonzero(a.argmax(-1) != b.argmax(-1))) if name in ["bid", "id", "pu"] else None,
        }
    return result


def worker(args):
    import onnx
    import onnxruntime as ort

    bundle = args.root / ("attention24-fp16" if args.provider == "cuda" else "attention-decomposed-fp16")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert digest(bundle / "model.onnx") == manifest["model_sha256"]
    case = next(c for c in manifest["cases"] if c["file"] == args.case)
    assert digest(bundle / args.case) == case["sha256"]
    with np.load(bundle / args.case) as data:
        feeds = {k: data[k].copy() for k in ["Xfeat_normed", "mask"]}
    dest = args.outdir / Path(args.case).stem / args.mode
    dest.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(bundle / "model.onnx")
    for value in graph.graph.input:
        for dim, size in zip(value.type.tensor_type.shape.dim, feeds[value.name].shape, strict=True):
            dim.dim_value = size
    for value in graph.graph.output:
        for dim, size in zip(value.type.tensor_type.shape.dim[:2], feeds["mask"].shape, strict=True):
            dim.dim_value = size
    del graph.graph.value_info[:]
    path = dest / "static.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(graph), path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = True
    options.profile_file_prefix = str(dest / "placement")
    if args.provider == "cuda":
        # Importing torch loads the container's CUDA libraries before ORT.
        import torch  # noqa: F401

        ort.preload_dlls()
        provider = "CUDAExecutionProvider"
        settings = {"use_tf32": "0", "sdpa_kernel": "1" if args.mode == "fused" else "16"}
    else:
        provider = "MIGraphXExecutionProvider"
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        settings = {}
    session = ort.InferenceSession(str(path), options, providers=[(provider, settings)])
    session.disable_fallback()
    values = session.run(OUTPUTS, feeds)
    np.savez(dest / "outputs.npz", **dict(zip(OUTPUTS, values, strict=True)))
    profile = Path(session.end_profiling())
    counts = Counter(e["args"]["provider"] for e in json.loads(profile.read_text()) if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
    assert counts and set(counts) == {provider}, counts
    metadata = {
        "provider": provider,
        "mode": args.mode,
        "case": args.case,
        "ort": ort.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "model_sha256": manifest["model_sha256"],
        "input_sha256": case["sha256"],
        "static_sha256": digest(path),
        "compute_providers": dict(counts),
        "loaded_libraries": sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/" in line and any(k in line for k in ["migraphx", "hipblas", "rocblas", "amdhip", "amd_comgr"])
            }
        ),
        "provider_options": session.get_provider_options(),
        "environment": {
            k: v for k, v in os.environ.items() if k.startswith(("MIGRAPHX_", "ORT_", "HIPBLASLT_", "ROCBLAS_")) or k == "LD_LIBRARY_PATH"
        },
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def verify_execution(dest, provider, mode, expected, size):
    path = dest / "run.log"
    log = path.read_text() if path.exists() else gzip.decompress(path.with_suffix(".log.gz").read_bytes()).decode()
    if provider == "cuda":
        selected = "FLASH_ATTENTION" if mode == "fused" else "MATH"
        dispatches = [line for line in log.splitlines() if "Operator=Attention " in line]
        assert len(dispatches) == expected and all(f"SdpaKernel={selected}" in line for line in dispatches), dispatches
        return {"verified": True, "dispatches": dispatches}
    assert 'module: "main"\n' in log
    final = 'module: "main"\n' + log.rsplit('module: "main"\n', 1)[1].split("\n\n", 1)[0] + "\n"
    (dest / "compiled-program.txt").write_text(final)
    objects = [
        line
        for line in final.splitlines()
        if "gpu::code_object" in line and "symbol_name=mlir_" in line and "reduce_max" in line and "reduce_sum" in line and line.count("dot") >= 2
    ]
    quadratic = [line for line in final.splitlines() if ("allocate" in line or "= load[" in line) and f", {size}, {size}" in line]
    assert len(objects) == (expected if mode == "fused" else 0), objects
    if mode == "fused":
        assert not quadratic, quadratic
    else:
        assert quadratic, "Unfused attention score buffers were not established"
    return {"verified": True, "fused_attention_code_objects": len(objects), "quadratic_buffers": quadratic}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--provider", choices=["cuda", "migraphx"], required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--case", help="Default: all seven retained cases")
    parser.add_argument("--mode", choices=["fused", "unfused"], help=argparse.SUPPRESS)
    parser.add_argument("--summarize-only", action="store_true", help="Verify and compare retained successful workers without rerunning")
    parser.add_argument("--reuse-successful", action="store_true", help="Resume existing successful workers in this output directory")
    parser.add_argument("--compress-logs", action="store_true", help="Compress successful worker logs after verification")
    args = parser.parse_args()
    if args.mode:
        worker(args)
        return
    bundle = args.root / ("attention24-fp16" if args.provider == "cuda" else "attention-decomposed-fp16")
    manifest = json.loads((bundle / "manifest.json").read_text())
    import onnx

    expected = sum(n.op_type == ("Attention" if args.provider == "cuda" else "Softmax") for n in onnx.load(bundle / "model.onnx").graph.node)
    assert expected > 0
    args.outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "provider": args.provider,
        "model_sha256": manifest["model_sha256"],
        "atol": 1e-3,
        "rtol": 3e-3,
        "cases": [],
        "execution_verified": False,
    }
    for case in manifest["cases"]:
        if args.case and args.case != case["file"]:
            continue
        for mode in ["fused", "unfused"]:
            if args.summarize_only:
                continue
            env = os.environ.copy()
            env["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
            if args.provider == "migraphx":
                env.update(
                    MIGRAPHX_DISABLE_MLIR="0",
                    MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention" if mode == "fused" else "!attention",
                    MIGRAPHX_TRACE_COMPILE="1",
                    MIGRAPHX_DISABLE_PASSES="",
                    MIGRAPHX_GPU_HIP_FLAGS="",
                )
            dest = args.outdir / Path(case["file"]).stem / mode
            dest.mkdir(parents=True, exist_ok=True)
            if (
                args.reuse_successful
                and (dest / "metadata.json").exists()
                and (dest / "exit-code.txt").exists()
                and (dest / "exit-code.txt").read_text().strip() == "0"
            ):
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.root),
                "--provider",
                args.provider,
                "--outdir",
                str(args.outdir),
                "--case",
                case["file"],
                "--mode",
                mode,
            ]
            with (dest / "run.log").open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            (dest / "exit-code.txt").write_text(str(completed.returncode) + "\n")
            if completed.returncode:
                raise RuntimeError(f'{case["file"]}/{mode} failed; see {dest}/run.log')
        dest = args.outdir / Path(case["file"]).stem
        metas = [json.loads((dest / mode / "metadata.json").read_text()) for mode in ["fused", "unfused"]]
        assert metas[0]["static_sha256"] == metas[1]["static_sha256"]
        for mode, metadata in zip(["fused", "unfused"], metas, strict=True):
            assert metadata["model_sha256"] == manifest["model_sha256"]
            assert metadata["input_sha256"] == case["sha256"]
            assert metadata["mode"] == mode and metadata["case"] == case["file"]
            assert (dest / mode / "exit-code.txt").read_text().strip() == "0"
            if args.provider == "migraphx":
                assert metadata["environment"]["MIGRAPHX_DISABLE_MLIR"] == "0"
                assert metadata["environment"]["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] == ("attention" if mode == "fused" else "!attention")
        proofs = {mode: verify_execution(dest / mode, args.provider, mode, expected, case["shape"][1]) for mode in ["fused", "unfused"]}
        if args.compress_logs:
            for mode in ["fused", "unfused"]:
                log = dest / mode / "run.log"
                if log.exists():
                    log.with_suffix(".log.gz").write_bytes(gzip.compress(log.read_bytes()))
                    log.unlink()
        with np.load(bundle / case["file"]) as ref, np.load(dest / "fused/outputs.npz") as fused, np.load(dest / "unfused/outputs.npz") as unfused:
            valid = ref["mask"].astype(bool)
            comparisons = {
                "fused_vs_fp32": compare(fused, ref, valid),
                "unfused_vs_fp32": compare(unfused, ref, valid),
                "fused_vs_unfused": compare(fused, unfused, valid),
            }
        report["cases"].append({"case": case["file"], "execution": proofs, "comparisons": comparisons})
        (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(case["file"], {k: max(e["max_abs"] for e in v.values()) for k, v in comparisons.items()}, flush=True)
    if not report["cases"]:
        parser.error("No cases selected")
    report["execution_verified"] = True
    report["numerical_passed_cases"] = {
        comparison: sum(
            all(e["close"] and e["finite"] and e["padding_zero"] for e in case["comparisons"][comparison].values()) for case in report["cases"]
        )
        for comparison in ["fused_vs_fp32", "unfused_vs_fp32", "fused_vs_unfused"]
    }
    (args.outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
