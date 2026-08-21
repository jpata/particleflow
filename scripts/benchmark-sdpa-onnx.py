#!/usr/bin/env python3
"""Benchmark PyTorch SDPA and ONNX Runtime attention variants.

This is a standalone microbenchmark for the attention operation used by the
attention-based MLPF model. It intentionally avoids MLPF checkpoints and TFDS
inputs so sequence length, dtype, masking, export shape, and ORT provider
options can be swept systematically.
"""

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData, create_calibrator
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import QuantizationMode, load_model_with_shape_infer

try:
    import psutil
except ImportError:
    psutil = None


class SdpaModule(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, masked: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.masked = masked

    def _to_bnhd(self, x):
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch, seq_len, _ = query.shape
        q = self._to_bnhd(query)
        k = self._to_bnhd(key)
        v = self._to_bnhd(value)
        attn_mask = None
        if self.masked:
            attn_mask = mask.to(torch.bool).view(batch, 1, 1, seq_len)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        return out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)


@dataclass
class Variant:
    name: str
    engine: str
    dtype: str
    backend: str = ""
    masked: bool = False
    provider_option_sdpa_kernel: int | None = None
    use_tf32: bool | None = None


class SingleBatchCalibrationDataReader(CalibrationDataReader):
    def __init__(self, feeds: dict[str, np.ndarray]):
        self.feeds = feeds
        self.enum_data = iter([feeds])

    def get_next(self):
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = iter([self.feeds])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch SDPA and ONNX Runtime attention variants.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--outdir", type=Path, default=Path("sdpa_onnx_bench"))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 10000])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--memory-sample-interval", type=float, default=0.02, help="Memory sampler interval in seconds.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mask-fraction", type=float, default=0.1)
    parser.add_argument("--modes", choices=["unmasked", "masked", "both"], default="both")
    parser.add_argument("--profile", action="store_true", help="Enable ORT profiling.")
    parser.add_argument("--keep-models", action="store_true", help="Keep exported ONNX files.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variants to run. Defaults depend on --device.",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        help="Extra ORT provider option as key=value. Useful for CUDA, e.g. sdpa_kernel=1.",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "fp16":
        return torch.float16, np.float16, TensorProto.FLOAT16
    if name == "bf16":
        return torch.bfloat16, None, TensorProto.BFLOAT16
    return torch.float32, np.float32, TensorProto.FLOAT


def make_variants(device: str, modes: list[str], selected: set[str] | None, provider_sdpa_kernel: int | None):
    variants = [
        Variant("pt_math_fp32", "pytorch", "fp32", "math"),
        Variant("onnx_sdpa_export_fp32", "onnx_export", "fp32"),
        Variant("ort_mha_fp32", "ort_mha", "fp32", provider_option_sdpa_kernel=provider_sdpa_kernel),
    ]
    if device == "cpu":
        variants.extend(
            [
                Variant("pt_math_fp16", "pytorch", "fp16", "math"),
                Variant("onnx_sdpa_export_fp16", "onnx_export", "fp16"),
                Variant("ort_mha_fp16", "ort_mha", "fp16", provider_option_sdpa_kernel=provider_sdpa_kernel),
                Variant("onnx_sdpa_static_int8", "onnx_static_quant", "int8"),
            ]
        )
    if device == "cuda":
        variants.extend(
            [
                Variant("pt_flash_fp16", "pytorch", "fp16", "flash"),
                Variant("pt_efficient_fp32", "pytorch", "fp32", "efficient"),
                Variant("ort_mha_fp16", "ort_mha", "fp16", provider_option_sdpa_kernel=provider_sdpa_kernel),
                Variant("onnx_sdpa_static_int8", "onnx_static_quant", "int8"),
            ]
        )

    expanded = []
    for variant in variants:
        for mode in modes:
            v = Variant(**asdict(variant))
            v.masked = mode == "masked"
            v.name = f"{variant.name}_{mode}"
            if selected is None or v.name in selected or variant.name in selected:
                expanded.append(v)
    return expanded


def provider_options_from_args(args):
    opts = {}
    for item in args.provider_option:
        if "=" not in item:
            raise ValueError(f"Invalid --provider-option {item!r}; expected key=value")
        key, value = item.split("=", 1)
        opts[key] = value
    return opts


def make_inputs(batch, seq_len, hidden, dtype, device, masked, mask_fraction, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    query = torch.randn(batch, seq_len, hidden, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    key = torch.randn(batch, seq_len, hidden, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    value = torch.randn(batch, seq_len, hidden, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    mask = None
    if masked:
        mask = torch.ones(batch, seq_len, dtype=torch.int32, device=device)
        num_masked = int(seq_len * mask_fraction)
        if num_masked > 0:
            mask[:, seq_len - num_masked :] = 0
    return query, key, value, mask


def run_pytorch_reference(query, key, value, mask, num_heads, head_dim):
    module = SdpaModule(num_heads, head_dim, mask is not None).eval().to(device=query.device)
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = module(query.float(), key.float(), value.float(), mask)
    return out.detach().float().cpu().numpy()


def run_pytorch_variant(variant, query, key, value, mask, num_heads, head_dim):
    module = SdpaModule(num_heads, head_dim, variant.masked).eval().to(device=query.device)
    backend = {
        "math": torch.nn.attention.SDPBackend.MATH,
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    }[variant.backend]
    with torch.no_grad(), torch.nn.attention.sdpa_kernel(backend):
        return module(query, key, value, mask)


def export_sdpa_model(path, query, key, value, mask, num_heads, head_dim, masked):
    module = SdpaModule(num_heads, head_dim, masked).eval().to(device=query.device)
    args = (query, key, value, mask) if masked else (query, key, value)
    input_names = ["query", "key", "value", "mask"] if masked else ["query", "key", "value"]
    dynamic_axes = {name: {0: "batch", 1: "seq"} for name in input_names}
    dynamic_axes["output"] = {0: "batch", 1: "seq"}
    torch.onnx.export(
        module,
        args,
        path,
        opset_version=20,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def make_mha_model(path, batch, seq_len, hidden, dtype_proto, num_heads, masked):
    inputs = ["query", "key", "value"]
    value_infos = [
        helper.make_tensor_value_info("query", dtype_proto, ["batch", "seq", hidden]),
        helper.make_tensor_value_info("key", dtype_proto, ["batch", "seq", hidden]),
        helper.make_tensor_value_info("value", dtype_proto, ["batch", "seq", hidden]),
    ]
    if masked:
        inputs.extend(["", "mask"])
        value_infos.append(helper.make_tensor_value_info("mask", TensorProto.INT32, ["batch", "seq"]))

    node = helper.make_node(
        "MultiHeadAttention",
        inputs,
        ["output"],
        "MultiHeadAttention_0",
        domain="com.microsoft",
        num_heads=num_heads,
        mask_filter_value=float("-inf"),
    )
    graph = helper.make_graph(
        [node],
        "SDPA_MHA_Graph",
        value_infos,
        [helper.make_tensor_value_info("output", dtype_proto, ["batch", "seq", hidden])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, path)


def quantize_sdpa_model(fp32_path, int8_path, feeds):
    quantize_static(
        fp32_path,
        int8_path,
        SingleBatchCalibrationDataReader(feeds),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        extra_options={
            "ActivationSymmetric": True,
            "MatMulConstBOnly": False,
        },
        calibration_providers=["CPUExecutionProvider"],
    )


def quantize_sdpa_model_integer_ops(fp32_path, int8_path, feeds):
    data_reader = SingleBatchCalibrationDataReader(feeds)
    calibrator = create_calibrator(
        fp32_path,
        ["MatMul"],
        calibrate_method=CalibrationMethod.MinMax,
        providers=["CPUExecutionProvider"],
    )
    calibrator.collect_data(data_reader)
    tensors_range = calibrator.compute_data()
    if not isinstance(tensors_range, TensorsData):
        raise TypeError(f"Unexpected calibration data type: {type(tensors_range)}")

    model = load_model_with_shape_infer(fp32_path)
    quantizer = ONNXQuantizer(
        model,
        per_channel=False,
        reduce_range=False,
        mode=QuantizationMode.IntegerOps,
        static=True,
        weight_qType=QuantType.QInt8,
        activation_qType=QuantType.QInt8,
        tensors_range=tensors_range,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=["MatMul"],
        extra_options={
            "ActivationSymmetric": True,
            "MatMulConstBOnly": False,
        },
    )
    quantizer.quantize_model()
    quantizer.model.save_model_to_file(int8_path)


def graph_stats(path):
    model = onnx.load(path)
    counts = {}
    for node in model.graph.node:
        key = f"{node.domain or 'onnx'}::{node.op_type}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def create_session(path, args, variant, model_dir):
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.num_threads
    so.inter_op_num_threads = args.num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(model_dir / f"{Path(path).stem}.optimized.onnx")
    if args.profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(model_dir / Path(path).stem)

    if args.device == "cuda":
        opts = provider_options_from_args(args)
        if variant.provider_option_sdpa_kernel is not None:
            opts.setdefault("sdpa_kernel", str(variant.provider_option_sdpa_kernel))
        providers = [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


def to_numpy_inputs(query, key, value, mask, variant):
    feeds = {
        "query": query.detach().cpu().numpy(),
        "key": key.detach().cpu().numpy(),
        "value": value.detach().cpu().numpy(),
    }
    if variant.masked:
        feeds["mask"] = mask.detach().cpu().numpy().astype(np.int32)
    return feeds


def summarize_times(times):
    if not times:
        return {}
    ordered = sorted(times)
    return {
        "mean_ms": statistics.mean(times) * 1000.0,
        "std_ms": (statistics.stdev(times) * 1000.0) if len(times) > 1 else 0.0,
        "p50_ms": ordered[len(ordered) // 2] * 1000.0,
        "p90_ms": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)] * 1000.0,
        "p99_ms": ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)] * 1000.0,
    }


class MemorySampler:
    def __init__(self, device: str, interval: float):
        self.device = device
        self.interval = interval
        self.pid = os.getpid()
        self.process = psutil.Process() if psutil is not None else None
        self.kind = "process_vram" if device == "cuda" else "rss"
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        value = self._sample_gpu_vram_mib() if self.device == "cuda" else self._sample_cpu_rss_mib()
        if value is not None:
            self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def _sample_cpu_rss_mib(self):
        if self.process is None:
            return None
        return self.process.memory_info().rss / (1024**2)

    def _sample_gpu_vram_mib(self):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            total = 0.0
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2 and parts[0] == str(self.pid):
                    total += float(parts[1])
            if total > 0.0:
                return total
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return torch.cuda.max_memory_allocated() / (1024**2)


def summarize_memory(runs):
    values = [run["memory_mib"] for run in runs if run.get("memory_mib") is not None]
    if not values:
        return {"peak_memory_mib": None, "mean_peak_memory_mib": None}
    return {"peak_memory_mib": max(values), "mean_peak_memory_mib": statistics.mean(values)}


def benchmark_variant(args, variant, seq_len, hidden, model_dir):
    input_dtype_name = "fp32" if variant.engine == "onnx_static_quant" else variant.dtype
    torch_dtype, _, onnx_dtype = dtype_from_name(input_dtype_name)
    device = torch.device(args.device)
    query, key, value, mask = make_inputs(
        args.batch_size,
        seq_len,
        hidden,
        torch_dtype,
        device,
        variant.masked,
        args.mask_fraction,
        args.seed,
    )
    reference = run_pytorch_reference(query, key, value, mask, args.num_heads, args.head_dim)

    result = {
        "variant": variant.name,
        "engine": variant.engine,
        "dtype": variant.dtype,
        "backend": variant.backend,
        "device": args.device,
        "masked": variant.masked,
        "batch_size": args.batch_size,
        "sequence_length": seq_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "hidden": hidden,
        "status": "ok",
        "error": "",
        "graph_nodes": {},
        "optimized_graph_nodes": {},
        "memory_kind": "process_vram" if args.device == "cuda" else "rss",
        "runs": [],
    }

    try:
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        if variant.engine == "pytorch":
            for _ in range(args.warmup):
                _ = run_pytorch_variant(variant, query, key, value, mask, args.num_heads, args.head_dim)
            if args.device == "cuda":
                torch.cuda.synchronize()

            times = []
            output = None
            for _ in range(args.iters):
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                with MemorySampler(args.device, args.memory_sample_interval) as memory:
                    t0 = time.perf_counter()
                    output = run_pytorch_variant(variant, query, key, value, mask, args.num_heads, args.head_dim)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                result["runs"].append({"runtime_s": elapsed, "memory_mib": memory.peak_mib})

            output_np = output.detach().float().cpu().numpy()
        else:
            model_path = model_dir / f"{variant.name}_s{seq_len}.onnx"
            if variant.engine == "onnx_export":
                export_sdpa_model(model_path, query, key, value, mask, args.num_heads, args.head_dim, variant.masked)
            elif variant.engine == "ort_mha":
                make_mha_model(model_path, args.batch_size, seq_len, hidden, onnx_dtype, args.num_heads, variant.masked)
            elif variant.engine == "onnx_static_quant":
                fp32_model_path = model_dir / f"{variant.name}_s{seq_len}.fp32.onnx"
                export_sdpa_model(
                    fp32_model_path,
                    query,
                    key,
                    value,
                    mask,
                    args.num_heads,
                    args.head_dim,
                    variant.masked,
                )
                feeds = to_numpy_inputs(query, key, value, mask, variant)
                result["source_graph_nodes"] = graph_stats(fp32_model_path)
                if args.device == "cuda":
                    quantize_sdpa_model_integer_ops(fp32_model_path, model_path, feeds)
                else:
                    quantize_sdpa_model(fp32_model_path, model_path, feeds)
            else:
                raise ValueError(f"Unknown engine {variant.engine}")

            result["graph_nodes"] = graph_stats(model_path)
            sess = create_session(model_path, args, variant, model_dir)
            optimized_path = Path(sess.get_session_options().optimized_model_filepath)
            if optimized_path.exists():
                result["optimized_graph_nodes"] = graph_stats(optimized_path)
            feeds = to_numpy_inputs(query, key, value, mask, variant)
            for _ in range(args.warmup):
                _ = sess.run(None, feeds)
            if args.device == "cuda":
                torch.cuda.synchronize()

            times = []
            output_np = None
            for _ in range(args.iters):
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                with MemorySampler(args.device, args.memory_sample_interval) as memory:
                    t0 = time.perf_counter()
                    output_np = sess.run(None, feeds)[0]
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                result["runs"].append({"runtime_s": elapsed, "memory_mib": memory.peak_mib})

            if args.profile:
                result["profile_file"] = sess.end_profiling()
            if not args.keep_models:
                try:
                    model_path.unlink()
                except OSError:
                    pass
                if variant.engine == "onnx_static_quant":
                    try:
                        fp32_model_path.unlink()
                    except OSError:
                        pass

        result.update(summarize_times(times))
        result.update(summarize_memory(result["runs"]))
        diff = np.abs(reference - output_np.astype(np.float32))
        result["mae"] = float(np.nanmean(diff))
        result["max_abs_err"] = float(np.nanmax(diff))
        result["num_invalid"] = int(np.sum(~np.isfinite(output_np)))
        if result["num_invalid"] > 0:
            result["status"] = "invalid_output"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if args.device == "cuda":
            torch.cuda.empty_cache()
    finally:
        del query, key, value, mask
        gc.collect()
    return result


def system_info(args):
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "device": args.device,
        "num_threads": args.num_threads,
        "environment": {
            key: os.environ.get(key)
            for key in [
                "ORT_DISABLE_FLASH_ATTENTION",
                "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION",
                "ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO",
                "ORT_ENABLE_CUDNN_FLASH_ATTENTION",
                "ORT_MIN_SEQ_LEN_EFFICIENT_ATTENTION_FP32",
                "ORT_MIN_SEQ_LEN_FLASH_ATTENTION_PACKED_QKV",
            ]
            if os.environ.get(key) is not None
        },
    }
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    else:
        info["cuda_available"] = False
    return info


def write_outputs(outdir, payload):
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    rows = payload["results"]
    csv_path = outdir / "summary.csv"
    fields = [
        "variant",
        "status",
        "device",
        "engine",
        "dtype",
        "backend",
        "masked",
        "sequence_length",
        "num_heads",
        "head_dim",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "memory_kind",
        "peak_memory_mib",
        "mean_peak_memory_mib",
        "mae",
        "max_abs_err",
        "num_invalid",
        "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, csv_path


def print_ranking(results):
    ok = [r for r in results if r.get("status") == "ok" and "mean_ms" in r]
    failed = [r for r in results if r.get("status") != "ok"]
    print("\nRuntime ranking:")
    for row in sorted(ok, key=lambda r: (r["sequence_length"], r["mean_ms"])):
        peak_memory_mib = row.get("peak_memory_mib")
        if peak_memory_mib is None:
            peak_memory_mib = float("nan")
        print(
            f"  S={row['sequence_length']:5d} {row['variant']:<32s} "
            f"{row['mean_ms']:10.3f} ms  "
            f"peak_mem={peak_memory_mib:10.1f} MiB  "
            f"MAE={row.get('mae', float('nan')):.3e}"
        )
    if failed:
        print("\nFailures:")
        for row in failed:
            print(f"  S={row['sequence_length']:5d} {row['variant']:<32s} {row['error']}")


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if args.device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("--device cuda requested, but CUDAExecutionProvider is not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.num_threads)

    modes = ["unmasked", "masked"] if args.modes == "both" else [args.modes]
    provider_opts = provider_options_from_args(args)
    provider_sdpa_kernel = int(provider_opts["sdpa_kernel"]) if "sdpa_kernel" in provider_opts else None
    selected = set(args.variants) if args.variants else None
    variants = make_variants(args.device, modes, selected, provider_sdpa_kernel)
    if not variants:
        raise RuntimeError("No variants selected")

    hidden = args.num_heads * args.head_dim
    model_dir = args.outdir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seq_len in args.seq_lens:
        for variant in variants:
            print(f"Running {variant.name} S={seq_len} on {args.device}")
            results.append(benchmark_variant(args, variant, seq_len, hidden, model_dir))

    payload = {
        "system": system_info(args),
        "config": {
            "seq_lens": args.seq_lens,
            "batch_size": args.batch_size,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "iters": args.iters,
            "warmup": args.warmup,
            "memory_sample_interval": args.memory_sample_interval,
            "mask_fraction": args.mask_fraction,
            "modes": modes,
        },
        "results": results,
    }
    summary_path, csv_path = write_outputs(args.outdir, payload)
    print_ranking(results)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
