"""Measure CLD hit-set inference across hit and query-slot counts."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmark_set_object_formation import PLATFORM_PATH, SCENARIO_PATH, VARIANTS, synthetic_batch
from mlpf.model.mlpf import MLPF
from mlpf.training_scenarios import load_platform_profile, load_training_scenario, resolve_scenario_job


HIT_COUNTS = (4096, 16384, 32768, 65536, 100000)
SLOT_COUNTS = (256, 512, 1024, 2048, 4096, 8192, 10000)
CASES = tuple(dict.fromkeys([
    *((hits, 256) for hits in HIT_COUNTS),
    *((4096, slots) for slots in SLOT_COUNTS),
    (32768, 10000),
    (100000, 4096),
    (100000, 8192),
    (100000, 10000),
]))


def measure_forward(model, hits, mask):
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encode_backbone(hits, mask)
        torch.cuda.synchronize()
        backbone_end = time.perf_counter()
        model.set_decoder(memory, mask, hits)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    return {
        "backbone_ms": (backbone_end - started) * 1000,
        "decoder_ms": (finished - backbone_end) * 1000,
        "total_ms": (finished - started) * 1000,
    }


def run_worker(variant, num_hits, num_slots, repeats):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda:0")
    scenario = load_training_scenario(SCENARIO_PATH)
    platform = load_platform_profile(PLATFORM_PATH)
    job = resolve_scenario_job(scenario, platform, variant, scenario.seeds[0], spec_file="particleflow_spec.yaml")
    config = job.resolved_config
    config.model.attention.use_flash_attn_varlen = False
    config.model.set_decoder.num_slots = num_slots
    torch.manual_seed(12345)
    model = MLPF(config).to(device).eval()
    batch = synthetic_batch(num_hits, 1, device)
    measure_forward(model, batch.X, batch.mask)
    torch.cuda.reset_peak_memory_stats(device)
    timings = [measure_forward(model, batch.X, batch.mask) for _ in range(repeats)]
    return {
        "variant": variant,
        "hits": num_hits,
        "slots": num_slots,
        "status": "ok",
        **{key: round(statistics.median(timing[key] for timing in timings), 2) for key in timings[0]},
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=Path, default=Path("experiments/cld_set_hits_object_formation_comparison/inference_scaling_gpu.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--hits", type=int)
    parser.add_argument("--slots", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")
    if args.worker:
        if args.variant is None or args.hits is None or args.slots is None:
            parser.error("--worker requires --variant, --hits, and --slots")
        try:
            print(json.dumps(run_worker(args.variant, args.hits, args.slots, args.repeats)), flush=True)
        except torch.cuda.OutOfMemoryError as error:
            print(json.dumps({"variant": args.variant, "hits": args.hits, "slots": args.slots, "status": "oom", "error": str(error)}), flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "batch_size": 1,
        "dtype": "bfloat16 autocast",
        "benchmark": "eval/inference_mode, full backbone and set decoder, no matching or backward; synchronized wall-clock",
        "configuration_change": "flash_attn_varlen disabled on local NVIDIA GPU; query-slot count varied from the 256-slot scenario configuration",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["gpu"] != result["gpu"] or previous["torch"] != result["torch"]:
            raise ValueError("Existing results use a different GPU or PyTorch version")
        result["rows"] = previous["rows"]
    completed = {(row["variant"], row["hits"], row["slots"]) for row in result["rows"]}
    for variant in VARIANTS:
        for num_hits, num_slots in CASES:
            if (variant, num_hits, num_slots) in completed:
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--variant", variant,
                "--hits", str(num_hits),
                "--slots", str(num_slots),
                "--repeats", str(args.repeats),
            ]
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
                if process.returncode:
                    row = {"variant": variant, "hits": num_hits, "slots": num_slots, "status": "error", "error": process.stderr[-1500:]}
                else:
                    row = json.loads(process.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                row = {"variant": variant, "hits": num_hits, "slots": num_slots, "status": "timeout", "timeout_seconds": args.timeout}
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
