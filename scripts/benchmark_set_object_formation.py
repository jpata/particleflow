"""Benchmark the CLD hit-set object-formation variants on synthetic events."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from mlpf.model.PFDataset import PFBatch
from mlpf.model.losses import REGRESSION_FEATURES
from mlpf.model.mlpf import MLPF
from mlpf.model.set_losses import SetMatcherWeights, set_mlpf_loss
from mlpf.model.utils import unpack_predictions, unpack_target
from mlpf.training_scenarios import load_platform_profile, load_training_scenario, resolve_scenario_job


SCENARIO_PATH = Path("configs/training/scenarios/cld_set_hits_object_formation_comparison.yaml")
PLATFORM_PATH = Path("configs/training/platforms/lumi_mi250x.yaml")
VARIANTS = ("baseline_topk_absolute", "grid_diverse_aggregate_anchors")
CASES = (
    (512, 64),
    (1024, 64),
    (2048, 64),
    (4096, 64),
    (8192, 64),
    (16384, 64),
    (1024, 16),
    (1024, 128),
    (1024, 256),
)


def synthetic_batch(num_hits, num_targets, device):
    hits = torch.zeros(1, num_hits, 15, device=device)
    element_type = torch.where(torch.arange(num_hits, device=device) % 2 == 0, 1.0, 2.0)
    eta = torch.randn(num_hits, device=device).clamp(-3, 3)
    phi = torch.rand(num_hits, device=device) * (2 * torch.pi) - torch.pi
    energy = torch.rand(num_hits, device=device) * 5 + 0.1
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

    targets = torch.zeros(1, num_targets, 14, device=device)
    target_eta = torch.randn(num_targets, device=device).clamp(-3, 3)
    target_phi = torch.rand(num_targets, device=device) * (2 * torch.pi) - torch.pi
    target_pt = torch.rand(num_targets, device=device) * 20 + 0.5
    targets[0, :, 0] = torch.arange(num_targets, device=device) % 5 + 1
    targets[0, :, 2] = torch.log(target_pt)
    targets[0, :, 3] = target_eta
    targets[0, :, 4] = torch.sin(target_phi)
    targets[0, :, 5] = torch.cos(target_phi)
    targets[0, :, 6] = torch.log(target_pt * torch.cosh(target_eta))
    return PFBatch(X=hits, ytarget_set=targets)


def measure_step(model, batch, matcher_weights):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw_predictions = model(batch.X, batch.mask)
    torch.cuda.synchronize()
    forward_end = time.perf_counter()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        predictions = unpack_predictions(raw_predictions)
        auxiliary = [unpack_predictions(output) for output in raw_predictions.auxiliary_predictions]
        loss, _, _ = set_mlpf_loss(
            unpack_target(batch.ytarget_set, model),
            predictions,
            batch,
            {feature: 1.0 for feature in REGRESSION_FEATURES},
            matcher_weights=matcher_weights,
            no_object_weight=model.config.set_decoder.no_object_weight,
            cardinality_loss_weight=model.config.set_decoder.cardinality_loss_weight,
            auxiliary_predictions=auxiliary,
            auxiliary_loss_weight=model.config.set_decoder.auxiliary_loss_weight,
        )
    torch.cuda.synchronize()
    loss_end = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_end = time.perf_counter()
    return {
        "forward_ms": (forward_end - started) * 1000,
        "matching_loss_ms": (loss_end - forward_end) * 1000,
        "backward_ms": (backward_end - loss_end) * 1000,
        "total_ms": (backward_end - started) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("experiments/cld_set_hits_object_formation_comparison/scaling_gpu_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    device = torch.device("cuda:0")
    scenario = load_training_scenario(SCENARIO_PATH)
    platform = load_platform_profile(PLATFORM_PATH)
    rows = []
    for variant in VARIANTS:
        job = resolve_scenario_job(scenario, platform, variant, scenario.seeds[0], spec_file="particleflow_spec.yaml")
        config = job.resolved_config
        config.model.attention.use_flash_attn_varlen = False
        torch.manual_seed(12345)
        model = MLPF(config).to(device).train()
        matcher_weights = SetMatcherWeights(**config.model.set_decoder.matcher.model_dump())
        for num_hits, num_targets in CASES:
            torch.manual_seed(1000 + num_hits + num_targets)
            batch = synthetic_batch(num_hits, num_targets, device)
            measure_step(model, batch, matcher_weights)
            torch.cuda.reset_peak_memory_stats(device)
            timings = [measure_step(model, batch, matcher_weights) for _ in range(args.repeats)]
            row = {
                "variant": variant,
                "hits": num_hits,
                "targets": num_targets,
                "slots": config.model.set_decoder.num_slots,
                **{key: round(statistics.median(timing[key] for timing in timings), 2) for key in timings[0]},
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del batch
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "repeats": args.repeats,
        "batch_size": 1,
        "dtype": "bfloat16 autocast",
        "benchmark": "full model forward + Hungarian set loss + backward; synchronized wall-clock, no optimizer or DDP",
        "configuration_change": "flash_attn_varlen disabled because the local NVIDIA environment lacks the LUMI flash-attn extension",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
