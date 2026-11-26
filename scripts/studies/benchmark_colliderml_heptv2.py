#!/usr/bin/env python3
"""Replay the ColliderML PU0 release recipes or capacity-matched backbones."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch
from torch import nn

from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.model.benchmarking import telemetry_backend
from mlpf.model.heptv2 import HEPTv2Layer
from mlpf.model.losses import make_task_loss_weighter, mlpf_loss
from mlpf.model.mlpf import MLPF, configure_model_trainable
from mlpf.model.training import UNIT_REGRESSION_WEIGHTS, model_step
from mlpf.model.utils import get_lr_schedule
from mlpf.optimizers import get_optimizer
from mlpf.training_scenarios import load_platform_profile, load_training_scenario, resolve_scenario_job

SAMPLE = "colliderml_ttbar_nopu_pf/1:1.5.0"
BATCH_SIZES = [1, 8, 16, 32, 48, 64, 96, 128]
UNPADDED_LENGTHS = [128, 256, 512, 768, 1024, 1280, 2048, 4096, 8192, 12288, 16384, 19968]


def open_dataset(config, data_dir, split, pad_to_multiple):
    return PFDataset(
        data_dir,
        SAMPLE,
        split,
        pad_to_multiple=pad_to_multiple,
        feature_dim=config.input_dim,
        max_open_readers=4,
    ).ds


def inference_batches(config, args, collate):
    data = open_dataset(config, args.data_dir, "test", config.pad_to_multiple_elements)
    events = [data[index] for index in range(max(args.batches))]
    for size in args.batches:
        yield f"batch/{size}", [collate(events[:size])]
    if args.lengths:
        data = open_dataset(config, args.data_dir, "test", None)
        first = data[0]
        elements = []
        total_length = 0
        longest_requested = max(args.lengths)
        for index in range(len(data)):
            record = first if index == 0 else data[index]
            valid = record["X"][:, 0] != 0
            event = torch.as_tensor(record["X"][valid])
            elements.append(event)
            total_length += len(event)
            if total_length >= longest_requested:
                break
        longest_event = max(elements, key=len)
        composite = torch.cat(elements)
        for length in args.lengths:
            inputs = longest_event if length <= len(longest_event) else composite
            if len(inputs) < length:
                raise ValueError(f"Test split has fewer than {length} elements")
            record = {
                "X": inputs[:length],
                "source_id": first["source_id"],
                "input_type_id": first["input_type_id"],
            }
            yield f"unpadded/{length}", [collate([record])]


def training_batches(config, args, collate):
    data = open_dataset(config, args.data_dir, "train", config.pad_to_multiple_elements)
    for size in args.batches:
        yield f"batch/{size}", [
            collate([data[index] for index in range(step * size, (step + 1) * size)]) for step in range(args.warmup + args.repeats)
        ]


def trainable_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def widen_attention_ffn(model, target_params):
    # Keep the release's input/output embedding and heads. Widen only the
    # two-linear attention FFN and round to a multiple suitable for tensor cores.
    layers = model.backbone
    input_width = layers[0].seq[0].in_features
    other_params = trainable_parameters(layers) - sum(trainable_parameters(layer.seq) for layer in layers)
    params_per_hidden_unit = len(layers) * (2 * input_width + 1)
    width = round((target_params - other_params - len(layers) * input_width) / params_per_hidden_unit / 8) * 8
    if width <= 0:
        raise ValueError(f"Invalid matched attention FFN width {width}")
    for layer in layers:
        layer.seq[0] = nn.Linear(input_width, width)
        layer.seq[2] = nn.Linear(width, input_width)
    return width


def make_model(config, weights, match_backbone_params=None):
    torch.manual_seed(config.seed)
    model = MLPF(config)
    if match_backbone_params is not None:
        widen_attention_ffn(model, match_backbone_params)
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    if weights:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    configure_model_trainable(model, config.model.trainable, True)
    return model.cuda()


def device_sample(backend):
    if backend is None:
        return {}
    try:
        return backend.sample()["metrics"]
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return {}


def sample_device(stop, samples, backend):
    while not stop.wait(0.05):
        samples.append(device_sample(backend))


def train_step(model, batch, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, *_ = model_step(batch, model, mlpf_loss, UNIT_REGRESSION_WEIGHTS)
    retained_mib = torch.cuda.memory_allocated() / 2**20
    forward_end = torch.cuda.Event(enable_timing=True)
    forward_end.record()
    loss.backward()
    backward_end = torch.cuda.Event(enable_timing=True)
    backward_end.record()
    optimizer.step()
    scheduler.step()
    return loss.detach(), forward_end, backward_end, retained_mib


def measure(model, host_batches, args, optimizer, scheduler):
    device = telemetry_backend(0)
    training = args.phase == "train"
    model.train(training)
    samples = []
    steps = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_device, args=(stop, samples, device))
    torch.cuda.empty_cache()
    try:
        for iteration in range(args.warmup + args.repeats):
            host_batch = host_batches[iteration if training else 0]
            batch = host_batch.to("cuda")
            torch.cuda.synchronize()
            measured = iteration >= args.warmup
            if iteration == args.warmup:
                samples.append(device_sample(device))
                monitor.start()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            wall_start = time.perf_counter()
            if training:
                loss, forward_end, backward_end, retained_mib = train_step(model, batch, optimizer, scheduler)
            else:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.predict_particles(batch.X, batch.mask, source_id=batch.source_id)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            if training and not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss")
            if not measured:
                continue
            lengths = host_batch.mask.sum(-1).tolist()
            step = {
                "wall_ms": wall_ms,
                "cuda_ms": start.elapsed_time(end),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "valid_lengths": lengths,
                "padded_width": host_batch.mask.shape[1],
            }
            if training:
                step["allocated_after_forward_loss_mib"] = retained_mib
                step["forward_loss_cuda_ms"] = start.elapsed_time(forward_end)
                step["backward_cuda_ms"] = forward_end.elapsed_time(backward_end)
                step["optimizer_cuda_ms"] = backward_end.elapsed_time(end)
            steps.append(step)
    finally:
        stop.set()
        if monitor.ident is not None:
            monitor.join()

    samples.append(device_sample(device))
    sizes = [sample["memory_used_bytes"] / 2**20 for sample in samples if sample.get("memory_used_bytes") is not None]
    times = [step["wall_ms"] for step in steps]
    events = sum(len(step["valid_lengths"]) for step in steps)
    tokens = sum(sum(step["valid_lengths"]) for step in steps)
    slots = sum(len(step["valid_lengths"]) * step["padded_width"] for step in steps)
    total_seconds = sum(times) / 1000
    return {
        "status": "ok",
        "median_ms": statistics.median(times),
        "p95_ms": statistics.quantiles(times, n=20, method="inclusive")[-1] if len(times) > 1 else times[0],
        "events_per_second": events / total_seconds,
        "valid_elements_per_second": tokens / total_seconds,
        "padding_efficiency": tokens / slots,
        "peak_allocated_mib": max(step["peak_allocated_mib"] for step in steps),
        "peak_reserved_mib": max(step["peak_reserved_mib"] for step in steps),
        "sampled_device_vram_mib": max(sizes, default=None),
        "device_sample_count": len(sizes),
        "steps": steps,
    }


def is_cuda_oom(error):
    while error is not None:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
        error = error.__cause__
    return False


def backbone_modules(model):
    for layer in model.backbone:
        yield "backbone", layer
        if isinstance(layer, HEPTv2Layer):
            yield "position_embedding", layer.pe_func
            yield "feedforward", layer.ff
            for norm in (layer.norm1, layer.norm2, layer.q_norm, layer.k_norm):
                yield "normalization", norm
            yield "attention", layer.attn
        else:
            yield "feedforward", layer.seq
            for norm in (layer.norm0, layer.norm1):
                yield "normalization", norm
            yield "attention", layer.mha


@contextmanager
def observe_backbone(model, scope):
    with ExitStack() as cleanup:
        for name, module in backbone_modules(model):
            if module is None:
                continue
            active = []

            def enter(_module, _inputs, name=name, active=active):
                context = scope(name)
                context.__enter__()
                active.append(context)

            def leave(_module, _inputs, _output, active=active):
                if active:
                    active.pop().__exit__(None, None, None)

            cleanup.callback(module.register_forward_pre_hook(enter).remove)
            cleanup.callback(module.register_forward_hook(leave, always_call=True).remove)
        yield


def diagnose_training(model, batch):
    # Count unique saved CUDA storage once, attributing shared views to the
    # first saving scope. Live parameters, buffers and input tensors are excluded.
    batch = batch.to("cuda")
    static = (*model.parameters(), *model.buffers(), batch.X, batch.mask, batch.ytarget, batch.source_id, batch.input_type_id)
    seen = {tensor.untyped_storage().data_ptr() for tensor in static if tensor is not None and tensor.is_cuda}
    groups = Counter()
    scopes = ["encoder_heads_loss"]

    @contextmanager
    def record_scope(name):
        scopes.append(name)
        try:
            yield
        finally:
            scopes.pop()

    def save(tensor):
        if tensor.is_cuda:
            with torch.no_grad():
                values = tensor.values() if tensor.is_nested else tensor
            storage = values.untyped_storage()
            pointer = storage.data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                groups[scopes[-1]] += storage.nbytes()
        return tensor.detach()

    model.zero_grad(set_to_none=True)
    with observe_backbone(model, record_scope), torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, *_ = model_step(batch, model, mlpf_loss, UNIT_REGRESSION_WEIGHTS)
    del loss
    return {"saved_storage_mib": {name: size / 2**20 for name, size in sorted(groups.items())}}


def profile_inference(model, batch):
    from torch.profiler import ProfilerActivity, profile

    batch = batch.to("cuda")
    samples = []
    with torch.inference_mode():
        for _ in range(3):
            with profile(activities=[ProfilerActivity.CUDA]) as profiler:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.predict_particles(batch.X, batch.mask, source_id=batch.source_id)
                torch.cuda.synchronize()
            kernels = [
                event
                for event in profiler.events()
                if event.device_type == torch.autograd.DeviceType.CUDA and not event.name.startswith(("Memcpy", "Memset"))
            ]
            samples.append({"kernel_count": len(kernels), "kernel_time_ms_sum": sum(event.time_range.elapsed_us() for event in kernels) / 1000})
    return {
        "kernel_count": statistics.median(sample["kernel_count"] for sample in samples),
        "kernel_time_ms_sum": statistics.median(sample["kernel_time_ms_sum"] for sample in samples),
        "samples": samples,
    }


def parity_snapshot(model, train_batch, test_batches, seed):
    result = {}
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for label, test_batch in test_batches:
            test_batch = test_batch.to("cuda")
            outputs = model.predict_particles(test_batch.X, test_batch.mask, source_id=test_batch.source_id)
            result[label] = {name: value[test_batch.mask].float().cpu() for name, value in outputs.items()}

    model.train()
    train_batch = train_batch.to("cuda")
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, _, outputs, _, _ = model_step(train_batch, model, mlpf_loss, UNIT_REGRESSION_WEIGHTS)
    loss.backward()
    result["train_prediction"] = {str(i): value.detach()[train_batch.mask].float().cpu() for i, value in enumerate(outputs)}
    result["train_loss"] = {"loss": loss.detach().float().cpu()}
    result["train_gradient"] = {name: p.grad.detach().float().cpu() for name, p in model.named_parameters() if p.grad is not None}
    return result


def compare_parity(reference, candidate):
    assert reference.keys() == candidate.keys()
    result = {}
    for group in reference:
        expected = reference[group]
        actual = candidate[group]
        assert expected.keys() == actual.keys()
        differences = [(actual[key].double() - value.double()).abs() for key, value in expected.items()]
        reference_norm = sum(value.double().square().sum().item() for value in expected.values()) ** 0.5
        relative_l2 = sum(difference.square().sum().item() for difference in differences) ** 0.5 / max(reference_norm, 1e-12)
        max_abs = max(difference.max().item() for difference in differences if difference.numel())
        result[group] = {
            "tensors": len(differences),
            "exact_tensors": sum(torch.equal(value, actual[key]) for key, value in expected.items()),
            "relative_l2_difference": relative_l2,
            "max_abs_difference": max_abs,
        }
    return result


def measure_parity(config, args):
    torch.use_deterministic_algorithms(True)
    train = open_dataset(config, args.data_dir, "train", config.pad_to_multiple_elements)
    collate = Collater(["X", "ytarget"], ["source_id", "input_type_id"])
    result = {}
    for size in args.batches:
        batch = collate([train[index] for index in range(size)])
        inference_collate = Collater(["X"], ["source_id", "input_type_id"])
        test = inference_batches(config, args, inference_collate)
        model = make_model(config, args.weights)
        result[f"train_batch/{size}"] = parity_snapshot(model, batch, ((label, batches[0]) for label, batches in test), config.seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "infer", "parity"), required=True)
    parser.add_argument("--variant", choices=("attention", "heptv2"), required=True)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pack-padding", action="store_true")
    parser.add_argument("--matched-capacity", action="store_true", help="Use seeded models and match Attention's backbone parameter count to HEPTv2")
    parser.add_argument("--diagnose-training-batch", type=int, help="Inspect saved forward storage at this training batch size")
    parser.add_argument("--profile-length", type=int, action="append", default=[], help="Profile this unpadded sequence after timing")
    parser.add_argument("--reference-snapshot", type=Path, help="Compare a parity replay with this snapshot")
    parser.add_argument("--platform", default="configs/training/platforms/tallinn_l40.yaml")
    parser.add_argument("--data-dir", default="/scratch/persistent/joosep/colliderml/tfds/")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--lengths", type=int, nargs="*", default=UNPADDED_LENGTHS)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batches = sorted(set(args.batches or ([8] if args.phase == "parity" else BATCH_SIZES)))
    if min(args.batches) < 1 or min(args.lengths, default=1) < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("Batch sizes, lengths, warmup and repeats must be positive")
    if args.profile_length and (args.phase != "infer" or not set(args.profile_length).issubset(args.lengths)):
        parser.error("--profile-length requires an inference length included in --lengths")
    if args.diagnose_training_batch is not None and (args.phase != "train" or args.diagnose_training_batch not in args.batches):
        parser.error("--diagnose-training-batch requires a training batch included in --batches")
    if args.matched_capacity and (args.weights or args.phase == "parity"):
        parser.error("--matched-capacity uses seeded models for training and inference")
    if not torch.cuda.is_available():
        parser.error("A CUDA or ROCm GPU is required")
    torch.set_num_threads(2)
    scenario = load_training_scenario("configs/training/scenarios/release_checkpoints.yaml")
    platform = load_platform_profile(args.platform)
    job = resolve_scenario_job(
        scenario,
        platform,
        f"colliderml_pf_{args.variant}",
        scenario.seeds[0],
        extra_overrides={"data_dir": args.data_dir},
    )
    config = job.resolved_config
    target_params = None
    if args.matched_capacity:
        config.lr = 0.0005
        if args.variant == "attention":
            heptv2_job = resolve_scenario_job(
                scenario,
                platform,
                "colliderml_pf_heptv2",
                scenario.seeds[0],
                extra_overrides={"data_dir": args.data_dir},
            )
            target_params = trainable_parameters(MLPF(heptv2_job.resolved_config).backbone)
        else:
            config.model.heptv2.activation_checkpointing = False
    if args.variant == "heptv2":
        if args.checkpointing is not None:
            config.model.heptv2.activation_checkpointing = args.checkpointing
        if args.pack_padding:
            config.model.heptv2.pack_padding = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "parity":
        snapshot = measure_parity(config, args)
        if args.reference_snapshot is None:
            torch.save(snapshot, args.output)
        else:
            reference = torch.load(args.reference_snapshot, map_location="cpu", weights_only=True)
            comparison = {key: compare_parity(reference[key], snapshot[key]) for key in snapshot}
            args.output.write_text(json.dumps(comparison, indent=2) + "\n")
            print(json.dumps(comparison, indent=2), flush=True)
        return

    result = {
        "arguments": vars(args),
        "sample": SAMPLE,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "device_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "hostname": os.uname().nodename,
        "heptv2_hash_limit": os.getenv("HEPTV2_ENCODER_NUM_HASHES"),
        "device_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "recipe": config.model_dump(mode="json"),
        "results": {},
    }
    training = args.phase == "train"
    collate = Collater(["X", "ytarget"] if training else ["X"], ["source_id", "input_type_id"])
    batches = training_batches(config, args, collate) if training else inference_batches(config, args, collate)
    for label, host_batches in batches:
        if args.phase == "infer" and args.variant == "heptv2":
            if host_batches[0].mask.shape[1] % config.model.heptv2.block_size:
                parser.error("HEPTv2 requires widths divisible by its block size")
        model = make_model(config, args.weights, target_params)
        if "parameters" not in result:
            first_layer = model.backbone[0]
            result["parameters"] = {
                "model_trainable": trainable_parameters(model),
                "backbone_trainable": trainable_parameters(model.backbone),
                "ffn_width": first_layer.ff.gate_proj.out_features if args.variant == "heptv2" else first_layer.seq[0].out_features,
            }
        optimizer = get_optimizer(model, config) if training else None
        scheduler = get_lr_schedule(config, optimizer, config.num_steps) if training else None
        try:
            point = measure(model, host_batches, args, optimizer, scheduler)
        except RuntimeError as error:
            if not is_cuda_oom(error):
                raise
            point = {"status": "oom", "error": str(error)}
        if point["status"] == "ok":
            if label in {f"unpadded/{length}" for length in args.profile_length}:
                point["profile"] = profile_inference(model, host_batches[0])
            if label == f"batch/{args.diagnose_training_batch}":
                point["diagnostics"] = diagnose_training(model, host_batches[args.warmup])
        result["results"][label] = point
        reference = result["results"].get("batch/1", {})
        if label.startswith("batch/") and point["status"] == reference.get("status") == "ok":
            speedup = point["events_per_second"] / reference["events_per_second"]
            point["batching_speedup"] = speedup
            point["batching_efficiency"] = speedup / host_batches[0].X.shape[0]
        print(f"{label} {json.dumps({k: v for k, v in point.items() if k != 'steps'})}", flush=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        del model, optimizer, scheduler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
