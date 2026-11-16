"""Single-device, input-only Panda-inspired pretraining: python -m mlpf.pretrain."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import ConcatDataset

from mlpf.conf import MLPFConfig
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFDataset
from mlpf.model.ssl import DistillationConfig, StudentTeacher, collate_inputs, encode_ssl_embeddings
from mlpf.model.ssl_probe import ProbeConfig, run_probe
from mlpf.model.ssl_snapshots import SnapshotConfig, save_snapshot
from mlpf.model.ssl_io import fetch_records, training_loader
from mlpf.model.ssl_timing import StepTimings


class PretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec_file: str = "particleflow_spec.yaml"
    model_name: str = "pyg-cld-v1"
    production: str = "cld"
    model_overrides: dict = Field(default_factory=dict)
    data_dir: str
    datasets: list[str] = Field(min_length=1)
    train_split: str = "train[:95%]"
    valid_split: str = "train[95%:]"
    train_events_per_dataset: int = Field(default=4096, ge=1)
    valid_events_per_dataset: int = Field(default=32, ge=1)
    batch_size: int = Field(default=4, ge=1)
    num_workers: int = Field(default=0, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    timing_frequency: int = Field(default=10, ge=1)
    steps: int = Field(default=2000, ge=1)
    validation_frequency: int = Field(default=100, ge=1)
    checkpoint_frequency: int = Field(default=100, ge=1)
    lr: float = Field(default=0.0001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    seed: int = Field(default=12345, ge=0)
    dtype: str = "bfloat16"
    ssl: DistillationConfig = Field(default_factory=DistillationConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def build_model_config(config):
    resolved = MLPFConfig.from_spec(config.spec_file, config.model_name, config.production).model_dump(mode="json")
    resolved["data_dir"] = config.data_dir
    return MLPFConfig.model_validate(merge(resolved, config.model_overrides))


def load_inputs(config, model_config, split, count):
    return ConcatDataset(
        [
            PFDataset(
                config.data_dir,
                name,
                split,
                num_samples=count,
                pad_to_multiple=None,
                feature_dim=model_config.input_dim,
                max_open_readers=4,
                input_only=True,
            ).ds
            for name in config.datasets
        ]
    )


def check_splits(config):
    """Resolve TFDS slices so differently spelled overlapping splits also fail."""
    if len(set(config.datasets)) != len(config.datasets):
        raise ValueError("Duplicate pretraining datasets")
    for name in config.datasets:
        info = tfds.builder(name, data_dir=config.data_dir).info
        train = tfds.core.ReadInstruction.from_spec(config.train_split).to_absolute(info.splits)
        valid = tfds.core.ReadInstruction.from_spec(config.valid_split).to_absolute(info.splits)
        for a in train:
            for b in valid:
                if a.splitname != b.splitname:
                    continue
                size = info.splits[a.splitname].num_examples
                if max(a.from_ or 0, b.from_ or 0) < min(size if a.to is None else a.to, size if b.to is None else b.to):
                    raise ValueError(f"Overlapping SSL train/validation slices in {name}")


def batch_at(dataset, indices, device):
    return {k: v.to(device) for k, v in collate_inputs(fetch_records(dataset, indices)).items()}


@torch.no_grad()
def validate(model, dataset, config, device, dtype):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    totals = {}
    for start in range(0, len(dataset), config.batch_size):
        stop = min(start + config.batch_size, len(dataset))
        batch = batch_at(dataset, range(start, stop), device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            _, metrics = model(**batch, generator=generator)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + float(value) * (stop - start)
    model.train()
    return {key: value / len(dataset) for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, step, config, model_config):
    checkpoint = {
        "ssl_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "pretrain_config": config.model_dump(mode="json"),
        "model_config": model_config.model_dump(mode="json"),
        "ssl_implementation": "detector-regions-v1" if config.ssl.mode == "regions" else "vectorized-v2",
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    # Standard training loader can consume this with --load_weights_only true.
    export = {
        "model_state_dict": model.student.state_dict(),
        "extra_state": {"step": step},
        "model_config": model_config.model_dump(mode="json"),
        "pretraining": "masked-detector-regions" if config.ssl.mode == "regions" else "observable-element-distillation",
    }
    temporary = path.parent / "student.tmp"
    torch.save(export, temporary)
    temporary.replace(path.parent / "student.pth")


def run(config, output, device, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(f"Output already exists: {output}; use a new directory or --resume")
    if config.dtype not in ("float32", "bfloat16"):
        raise ValueError("SSL supports float32 or bfloat16 (no unscaled FP16)")
    check_splits(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(device)
    dtype = getattr(torch, config.dtype)
    model_config = build_model_config(config)
    train_data = load_inputs(config, model_config, config.train_split, config.train_events_per_dataset)
    valid_data = load_inputs(config, model_config, config.valid_split, config.valid_events_per_dataset)
    student = MLPF(model_config).to(device)
    student.task_loss_weighter = make_task_loss_weighter(model_config.task_loss_weights.model_dump()).to(device)
    first = batch_at(train_data, [0], device)
    student.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        dimension = encode_ssl_embeddings(student, first["X"], first["X"][..., 0] != 0, first["source_id"]).shape[-1]
    model = StudentTeacher(student, dimension, config.ssl).to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay, fused=device.type == "cuda"
    )
    start = 1
    if resume:
        state = torch.load(output / "latest.pth", map_location="cpu", weights_only=False)
        saved = PretrainConfig.model_validate(state["pretrain_config"]).model_dump(mode="json")
        current = config.model_dump(mode="json")
        # Probes are isolated diagnostics and may be enabled on an existing run.
        for key in (
            "steps",
            "validation_frequency",
            "checkpoint_frequency",
            "probe",
            "snapshots",
            "num_workers",
            "prefetch_factor",
            "timing_frequency",
        ):
            saved.pop(key)
            current.pop(key)
        if saved != current:
            raise ValueError("Resume configuration differs from checkpoint")
        if state.get("model_config", model_config.model_dump(mode="json")) != model_config.model_dump(mode="json"):
            raise ValueError("Resolved model configuration differs from checkpoint")
        model.load_state_dict(state["ssl_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["fused"] = device.type == "cuda"
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["step"] + 1
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output / "pretrain-config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")))
    (output / "train-config.yaml").write_text(yaml.safe_dump(model_config.model_dump(mode="json")))
    if config.ssl.mode == "regions" and start == 1:
        # Preview is isolated from training RNG and never used as a model input.
        generator = torch.Generator(device=device).manual_seed(config.seed + 27182)
        preview = model.region_sampler(first["X"], first["X"][..., 0] != 0, generator)
        np.savez_compressed(output / "region_view.npz", X=first["X"].cpu().numpy(),
                            **{k: v.cpu().numpy() for k, v in preview.items()})
    with (output / "metrics.jsonl").open("a") as log:
        if config.snapshots.enabled and (start == 1 or not (output / "embeddings/cohort.json").exists()):
            snapshot = save_snapshot(model, valid_data, config, output, start - 1, device, dtype)
            log.write(json.dumps({"step": start - 1, "embeddings": snapshot}) + "\n")
            log.flush()
        if start == 1 and config.probe.enabled and config.probe.at_start:
            metrics = run_probe(model.student, config, model_config, output, 0, device, dtype)
            log.write(json.dumps({"step": 0, "probe": metrics}) + "\n")
            log.flush()
        loader = training_loader(train_data, config, start, device)
        batches = iter(loader)
        for step in range(start, config.steps + 1):
            timestamp = time.time()
            tick = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            timings = StepTimings(device, step == start or step % config.timing_frequency == 0)
            with timings.phase("data_wait"):
                cpu_batch = next(batches)
            with timings.phase("transfer"):
                batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with timings.phase("forward"), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss, metrics = model(**batch, timings=timings)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite SSL loss at step {step}")
            with timings.phase("backward"):
                loss.backward()
            with timings.phase("optimizer"):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=True)
                optimizer.step()
            with timings.phase("ema"):
                model.update_teacher()
            record = {"step": step, "train": {k: float(v) for k, v in metrics.items()}, "gradient_norm": float(norm)}
            training_seconds = time.perf_counter() - tick
            record["performance"] = {
                "start_unix_time": timestamp,
                "training_seconds": training_seconds,
                "events_per_second": config.batch_size / training_seconds,
            }
            if device.type == "cuda":
                record["performance"].update(
                    training_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    training_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                )
            if step % config.validation_frequency == 0 or step == config.steps:
                with timings.phase("validation"):
                    record["valid"] = validate(model, valid_data, config, device, dtype)
            if config.probe.enabled and (step % config.probe.frequency == 0 or step == config.steps):
                with timings.phase("probe"):
                    record["probe"] = run_probe(model.student, config, model_config, output, step, device, dtype)
            if config.snapshots.enabled and (step % config.snapshots.frequency == 0 or step == config.steps):
                with timings.phase("snapshot"):
                    record["embeddings"] = save_snapshot(model, valid_data, config, output, step, device, dtype)
            if step % config.checkpoint_frequency == 0 or step == config.steps:
                with timings.phase("checkpoint"):
                    save_checkpoint(output / "latest.pth", model, optimizer, step, config, model_config)
            if timings.enabled:
                record["timings"] = timings.results()
            record["performance"]["step_seconds"] = time.perf_counter() - tick
            if device.type == "cuda":
                record["performance"]["step_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or "valid" in record:
                print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, help="Override pretraining batch size (must match when resuming)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.steps is not None:
        settings["steps"] = args.steps
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    run(PretrainConfig.model_validate(settings), args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
