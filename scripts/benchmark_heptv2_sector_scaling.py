"""Benchmark HEPTv2 sectorized hit-set inference on synthetic CLD hits."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF


MODEL_NAME = "pyg-cld-hits-heptv2-sector-set-v1"
HIT_COUNTS = (1024, 4096, 16384, 32768, 65536, 100000)
SLOT_COUNTS = (256, 512, 1024, 2048, 4096, 8192, 10000)
CASES = tuple(dict.fromkeys(
    [*((num_hits, 256) for num_hits in HIT_COUNTS)]
    + [*((16384, num_slots) for num_slots in SLOT_COUNTS)]
    + [(32768, 10000), (100000, 4096), (100000, 10000)]
))


def synthetic_hits(num_hits, input_dim, block_size, device):
    padded_hits = ((num_hits + block_size - 1) // block_size) * block_size
    features = torch.zeros(1, padded_hits, input_dim, device=device)
    element_type = torch.where(torch.arange(num_hits, device=device) % 2 == 0, 1.0, 2.0)
    eta = torch.randn(num_hits, device=device).clamp(-3.0, 3.0)
    phi = torch.rand(num_hits, device=device) * (2.0 * torch.pi) - torch.pi
    energy = torch.rand(num_hits, device=device) * 5.0 + 0.1
    radius = torch.where(element_type == 1, 100.0, 2000.0)
    features[0, :num_hits, 0] = element_type
    features[0, :num_hits, 1] = energy / torch.cosh(eta)
    features[0, :num_hits, 2] = eta
    features[0, :num_hits, 3] = torch.sin(phi)
    features[0, :num_hits, 4] = torch.cos(phi)
    features[0, :num_hits, 5] = energy
    features[0, :num_hits, 6] = radius * torch.cos(phi)
    features[0, :num_hits, 7] = radius * torch.sin(phi)
    features[0, :num_hits, 8] = radius * torch.sinh(eta)
    features[0, :num_hits, 10] = torch.where(element_type == 1, 3.0, 4.0)
    mask = torch.arange(padded_hits, device=device).unsqueeze(0) < num_hits
    return features, mask


def measure(model, features, mask):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encode_backbone(features, mask)
        torch.cuda.synchronize()
        backbone_end = time.perf_counter()
        output = model.set_decoder(memory, mask, features)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    if not all(torch.isfinite(prediction).all().item() for prediction in output):
        raise ValueError("Non-finite model output")
    return {
        "backbone_ms": (backbone_end - started) * 1000.0,
        "decoder_ms": (finished - backbone_end) * 1000.0,
        "total_ms": (finished - started) * 1000.0,
    }


def run_case(num_hits, num_slots, repeats):
    device = torch.device("cuda:0")
    config = MLPFConfig.from_spec("particleflow_spec.yaml", MODEL_NAME, "cld")
    config.model.set_decoder.num_slots = num_slots
    torch.manual_seed(12345)
    model = MLPF(config).to(device).eval()
    features, mask = synthetic_hits(num_hits, config.input_dim, config.model.heptv2.block_size, device)
    measure(model, features, mask)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    timings = [measure(model, features, mask) for _ in range(repeats)]
    return {
        "hits": num_hits,
        "padded_hits": features.shape[1],
        "slots": num_slots,
        "status": "ok",
        **{key: round(statistics.median(timing[key] for timing in timings), 2) for key in timings[0]},
        "baseline_allocated_mib": round(baseline_allocated / 2**20, 1),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("experiments/cld_heptv2_sector_scaling/inference_gpu.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--hits", type=int)
    parser.add_argument("--slots", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.worker:
        if args.hits is None or args.slots is None:
            parser.error("--worker requires --hits and --slots")
        try:
            print(json.dumps(run_case(args.hits, args.slots, args.repeats)), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            print(json.dumps({"hits": args.hits, "slots": args.slots, "status": "oom", "error": str(error)}), flush=True)
        return

    result = {
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "batch_size": 1,
        "dtype": "bfloat16 autocast",
        "benchmark": "eval/inference_mode, full backbone and decoder, no Hungarian matching or backward; synchronized wall-clock",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["gpu"] != result["gpu"] or previous["torch"] != result["torch"] or previous["model"] != result["model"]:
            raise ValueError("Existing results use a different GPU, PyTorch version, or model")
        result["rows"] = previous["rows"]
    completed = {(row["hits"], row["slots"]) for row in result["rows"]}
    for num_hits, num_slots in CASES:
        if (num_hits, num_slots) in completed:
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker", "--hits", str(num_hits), "--slots", str(num_slots), "--repeats", str(args.repeats),
        ]
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            if process.returncode:
                row = {"hits": num_hits, "slots": num_slots, "status": "error", "error": process.stderr[-1500:]}
            else:
                row = json.loads(process.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            row = {"hits": num_hits, "slots": num_slots, "status": "timeout", "timeout_seconds": args.timeout}
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
