"""Benchmark the three CLD set-prediction reference paths on one local GPU.

Example:
    .venv/bin/python scripts/benchmark_set_reference_inference.py \
        --output experiments/set_reference_inference.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from mlpf.model.mlpf import MLPF
from mlpf.training_scenarios import load_platform_profile, load_training_scenario, resolve_scenario_job


SCENARIO = Path("configs/training/scenarios/cld_set_hits_heptv2_sector_comparison.yaml")
PLATFORM = Path("configs/training/platforms/lumi_mi250x.yaml")
CASES = (
    (512, 256), (2048, 256), (8192, 256), (32768, 256),
    (2048, 64), (2048, 1024), (8192, 1024), (8192, 4096), (32768, 4096),
)


def synthetic_hits(num_hits, device):
    generator = torch.Generator(device=device).manual_seed(1000 + num_hits)
    hits = torch.zeros((1, num_hits, 15), device=device)
    element_type = torch.where(torch.arange(num_hits, device=device) % 2 == 0, 1.0, 2.0)
    eta = torch.randn(num_hits, device=device, generator=generator).clamp(-3, 3)
    phi = torch.rand(num_hits, device=device, generator=generator) * (2 * torch.pi) - torch.pi
    energy = torch.rand(num_hits, device=device, generator=generator) * 5 + 0.1
    radius = torch.where(element_type == 1, 100.0, 2000.0)
    hits[0, :, 0] = element_type
    hits[0, :, 1] = energy / torch.cosh(eta)
    hits[0, :, 2] = eta
    hits[0, :, 3] = torch.sin(phi)
    hits[0, :, 4] = torch.cos(phi)
    hits[0, :, 5] = energy
    hits[0, :, 6] = radius * torch.cos(phi)
    hits[0, :, 7] = radius * torch.sin(phi)
    hits[0, :, 8] = radius * torch.sinh(eta)
    hits[0, :, 10] = torch.where(element_type == 1, 3.0, 4.0)
    return hits


def timed_forward(model, hits, mask):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encode_backbone(hits, mask)
        torch.cuda.synchronize()
        backbone_end = time.perf_counter()
        backbone_peak = torch.cuda.max_memory_allocated()
        backbone_current = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model.set_decoder(memory, mask, hits)
    torch.cuda.synchronize()
    end = time.perf_counter()
    decoder_peak = torch.cuda.max_memory_allocated()
    return {
        "backbone_ms": (backbone_end - start) * 1000,
        "decoder_ms": (end - backbone_end) * 1000,
        "backbone_peak_mib": backbone_peak / 2**20,
        "decoder_extra_peak_mib": (decoder_peak - backbone_current) / 2**20,
        "total_peak_mib": max(backbone_peak, decoder_peak) / 2**20,
    }


def benchmark_variant(variant_name, scenario, platform, repeats, cases):
    job = resolve_scenario_job(scenario, platform, variant_name, scenario.seeds[0], spec_file="particleflow_spec.yaml")
    config = job.resolved_config
    # Local NVIDIA uses PyTorch's jagged Flash SDPA backbone path; LUMI uses
    # the separate flash_attn_varlen backend for that same backbone.
    config.model.attention.use_flash_attn_varlen = False
    torch.manual_seed(12345)
    device = torch.device("cuda:0")
    model = MLPF(config).to(device).eval()
    max_slots = max(num_slots for _, num_slots in cases)
    query_bank = model.set_decoder.queries.detach().new_empty(1, max_slots, model.set_decoder.queries.shape[-1])
    torch.nn.init.trunc_normal_(query_bank, std=0.02)
    original_slots = min(max_slots, model.set_decoder.queries.shape[1])
    query_bank[:, :original_slots] = model.set_decoder.queries.detach()[:, :original_slots]
    rows = []
    for num_hits, num_slots in cases:
        model.set_decoder.num_slots = num_slots
        model.set_decoder.queries = torch.nn.Parameter(query_bank[:, :num_slots])
        hits = synthetic_hits(num_hits, device)
        mask = torch.ones((1, num_hits), dtype=torch.bool, device=device)
        try:
            timed_forward(model, hits, mask)  # kernel compilation and warmup
            samples = [timed_forward(model, hits, mask) for _ in range(repeats)]
            medians = {key: statistics.median(sample[key] for sample in samples) for key in samples[0]}
            row = {
                "variant": variant_name,
                "hits": num_hits,
                "slots": num_slots,
                "status": "ok",
                "backbone_ms": round(medians["backbone_ms"], 2),
                "decoder_ms": round(medians["decoder_ms"], 2),
                "total_ms": round(medians["backbone_ms"] + medians["decoder_ms"], 2),
                "backbone_peak_mib": round(medians["backbone_peak_mib"], 1),
                "decoder_extra_peak_mib": round(medians["decoder_extra_peak_mib"], 1),
                "total_peak_mib": round(medians["total_peak_mib"], 1),
            }
        except torch.cuda.OutOfMemoryError:
            row = {"variant": variant_name, "hits": num_hits, "slots": num_slots, "status": "oom"}
            torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("experiments/set_reference_inference.json"))
    parser.add_argument("--variant", action="append", help="Run only this scenario variant; repeat to select multiple")
    parser.add_argument("--case", nargs=2, action="append", type=int, metavar=("HITS", "SLOTS"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA or ROCm GPU is required")
    scenario = load_training_scenario(SCENARIO)
    platform = load_platform_profile(PLATFORM)
    variants = args.variant or list(scenario.variants)
    if any(variant not in scenario.variants for variant in variants):
        parser.error("Unknown variant")
    cases = tuple(map(tuple, args.case)) if args.case else CASES
    if any(hits < 1 or slots < 1 for hits, slots in cases):
        parser.error("Hits and slots must be positive")
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "batch_size": 1,
        "dtype": "bfloat16 autocast",
        "cases": cases,
        "measurement": "eval inference; backbone and decoder separately synchronized; no matching or backward",
        "rows": [],
    }
    for variant in variants:
        result["rows"].extend(benchmark_variant(variant, scenario, platform, args.repeats, cases))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        torch.cuda.empty_cache()
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
