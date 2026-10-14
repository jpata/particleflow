#!/usr/bin/env python3
"""Measure HEPTv2 checkpointing with real hits, matching, grouping loss and LAMB.

Run each configuration in a fresh process so peak reserved memory is comparable.
Uses the production model_step loss path, but a fixed, recorded sequence of local
events to isolate checkpointing. Does not replace a multi-rank platform smoke test.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.training import UNIT_REGRESSION_WEIGHTS, model_step
from mlpf.model.utils import get_lr_schedule
from mlpf.optimizers import get_optimizer
from mlpf.training_scenarios import load_platform_profile, load_training_scenario, resolve_scenario_jobs


def parity_snapshot(model, batch, weights):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, _, output, _, _ = model_step(batch, model, None, weights)
    loss.backward()
    tensors = [*output, *(t for aux in output.auxiliary_predictions for t in aux)]
    if output.hit_grouping is not None:
        tensors.extend(output.hit_grouping)
    return {
        "loss": loss.item(),
        "outputs": {str(i): t.detach().float().cpu() for i, t in enumerate(tensors)},
        "gradients": {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None},
    }


def compare_snapshots(reference, other):
    result = {"loss_abs_difference": abs(reference["loss"] - other["loss"])}
    for kind in ("outputs", "gradients"):
        assert reference[kind].keys() == other[kind].keys()
        differences = [reference[kind][k] - other[kind][k] for k in reference[kind]]
        norm = sum(t.double().square().sum().item() for t in reference[kind].values()) ** 0.5
        result[kind] = {
            "count": len(differences),
            "max_abs_difference": max(d.abs().max().item() for d in differences),
            "relative_l2_difference": sum(d.double().square().sum().item() for d in differences) ** 0.5 / max(norm, 1e-12),
        }
    return result


def compare_replay(model, batch, weights, seed):
    initial = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    snapshots = []
    # Same model, weights, events and RNG state; repeat each mode to expose GPU
    # nondeterminism, without allowing optimizer updates to amplify roundoff.
    for enabled in (False, False, True, True):
        model.load_state_dict(initial)
        model.config.heptv2.activation_checkpointing = enabled
        torch.manual_seed(seed)
        snapshots.append(parity_snapshot(model, batch, weights))
    return {
        "losses": [s["loss"] for s in snapshots],
        "baseline_repeat": compare_snapshots(snapshots[0], snapshots[1]),
        "checkpointed": compare_snapshots(snapshots[1], snapshots[2]),
        "checkpointed_repeat": compare_snapshots(snapshots[2], snapshots[3]),
    }


def measured_step(model, batch, optimizer, scheduler, weights, snapshot):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, components, _, output, _, _ = model_step(batch, model, None, weights)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    loss.backward()
    torch.cuda.synchronize()
    backward_time = time.perf_counter() - start
    result = {
        "loss": loss.item(),
        "components": {k: float(v) for k, v in components.items()},
        "padded_shape": list(batch.X.shape),
        "valid_hits": batch.mask.sum(-1).tolist(),
        "forward_backward_ms": backward_time * 1000,
        "forward_backward_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "forward_backward_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    # Host transfers and parity snapshots are outside the timed region.
    if snapshot is not None:
        tensors = [*output, *(t for aux in output.auxiliary_predictions for t in aux)]
        if output.hit_grouping is not None:
            tensors.extend(output.hit_grouping)
        torch.save(
            {
                "outputs": [t.detach().cpu() for t in tensors],
                "gradients": {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None},
            },
            snapshot,
        )
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
    torch.cuda.synchronize()
    optimizer_start = time.perf_counter()
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    result["step_ms"] = (backward_time + time.perf_counter() - optimizer_start) * 1000
    result["step_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    result["step_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/training/scenarios/cld_pf_heptv2_batch512_50k.yaml")
    parser.add_argument("--platform", default="configs/training/platforms/local.yaml")
    parser.add_argument("--data-dir", default="/mnt/work/mlpf/tensorflow_datasets/cld")
    parser.add_argument("--dataset", default="cld_edm_ttbar_hits/1:3.2.1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--tracker-momentum", action="store_true")
    parser.add_argument("--tracker-doublets", action="store_true")
    parser.add_argument("--features-only", action="store_true", help="Time input feature construction on a fixed real batch")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--parity-only", action="store_true", help="Compare repeated forward/backward passes without optimizer updates")
    parser.add_argument("--deterministic", action="store_true", help="Require deterministic PyTorch algorithms for parity diagnosis")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps <= args.warmup_steps or args.warmup_steps < 1:
        parser.error("Require positive batch size and steps > warmup-steps >= 1")
    if not torch.cuda.is_available():
        parser.error("A CUDA/ROCm GPU is required")

    scenario = load_training_scenario(args.scenario)
    platform = load_platform_profile(args.platform)
    jobs = resolve_scenario_jobs(scenario, platform, spec_file="particleflow_spec.yaml")
    config = next(job.resolved_config for job in jobs if job.variant_name == "hit_heptv2_grouped_tracker_features")
    config.model.heptv2.activation_checkpointing = args.checkpointing
    config.model.hit_feature_engineering.tracker_momentum = args.tracker_momentum
    config.model.hit_feature_engineering.tracker_doublets = args.tracker_doublets
    torch.use_deterministic_algorithms(args.deterministic)
    torch.manual_seed(config.seed)
    model = MLPF(config)
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    model = model.cuda().train()
    optimizer = get_optimizer(model, config)
    scheduler = get_lr_schedule(config, optimizer, config.num_steps)
    weights = UNIT_REGRESSION_WEIGHTS
    collate = Collater(["X", "ytarget", "ytarget_set"], [])

    def dataset(split):
        return PFDataset(
            args.data_dir,
            args.dataset,
            split,
            pad_to_multiple=config.pad_to_multiple_elements,
            feature_dim=config.input_dim,
            max_open_readers=4,
            build_target_set=True,
        ).ds

    training = dataset("train")
    validation = dataset("valid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(),
        "seed": config.seed,
        "model": config.model.model_dump(mode="json"),
        "training_recipe": {
            "optimizer": config.optimizer.value,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "lr_schedule": config.lr_schedule.value,
            "schedule_steps": config.num_steps,
            "dtype": "bfloat16",
            "world_size": 1,
            "batch_size": args.batch_size,
        },
        "training_event_indices": list(range(args.steps * args.batch_size)),
        "validation_event_indices": list(range(args.batch_size)),
        "steps": [],
    }
    if args.features_only:
        batch = collate([training[i] for i in range(args.batch_size)]).to("cuda")
        times, allocated, reserved = [], [], []
        base_memory = torch.cuda.memory_allocated()
        for iteration in range(13):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model._engineer_input_features(batch.X, batch.mask)
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - start)
            if iteration >= 3:
                times.append(elapsed)
                allocated.append(torch.cuda.max_memory_allocated() / 2**20)
                reserved.append(torch.cuda.max_memory_reserved() / 2**20)
            assert torch.isfinite(features).all()
            del features
        result["features"] = {
            "mean_ms": sum(times) / len(times),
            "times_ms": times,
            "peak_allocated_mib": max(allocated),
            "peak_reserved_mib": max(reserved),
            "incremental_peak_mib": max(allocated) - base_memory / 2**20,
            "padded_shape": list(batch.X.shape),
            "engineered_dim": model.input_dim,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["features"], indent=2), flush=True)
        return
    if args.parity_only:
        batch = collate([training[i] for i in range(args.batch_size)]).to("cuda")
        result["parity"] = compare_replay(model, batch, weights, config.seed)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["parity"], indent=2), flush=True)
        return

    result["status"] = "running"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for step in range(args.steps):
        records = [training[step * args.batch_size + i] for i in range(args.batch_size)]
        batch = collate(records).to("cuda")
        snapshot = args.output.with_suffix(".pt") if args.snapshot and step == 0 else None
        try:
            row = measured_step(model, batch, optimizer, scheduler, weights, snapshot)
        except torch.cuda.OutOfMemoryError as error:
            result.update(status="out_of_memory", failed_step=step, error=str(error))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            raise
        row["warmup"] = step < args.warmup_steps
        result["steps"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"step": step, **{k: v for k, v in row.items() if k != "components"}}), flush=True)
        del batch

    model.eval()
    optimizer.zero_grad(set_to_none=True)
    batch = collate([validation[i] for i in range(args.batch_size)]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, components, _, _, _, _ = model_step(batch, model, None, weights)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite validation loss")
    result["validation"] = {"loss": loss.item(), "components": {k: float(v) for k, v in components.items()}}
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"validation": result["validation"]}), flush=True)


if __name__ == "__main__":
    main()
