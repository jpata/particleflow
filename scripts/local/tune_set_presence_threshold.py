#!/usr/bin/env python3
"""Sweep the set-decoder presence threshold on a saved checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, SequentialSampler

from mlpf.conf import MLPFConfig
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import Collater, PFDataset
from mlpf.model.utils import unpack_predictions, unpack_target
from mlpf.model.validation_metrics import compute_validation_particle_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", default="cld_edm_ttbar_hits")
    parser.add_argument("--split-config", default="10")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--num-events", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold-start", type=float, default=0.20)
    parser.add_argument("--threshold-stop", type=float, default=0.65)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def physical_set_predictions(raw_predictions):
    presence, pid, momentum, pileup = raw_predictions
    presence = presence.float()
    pid = pid.float()
    momentum = momentum.float().clone()
    momentum[..., 0] = torch.exp(momentum[..., 0].clamp(-20.0, 20.0))
    momentum[..., 4] = torch.exp(momentum[..., 4].clamp(-20.0, 20.0))
    predictions = unpack_predictions((presence, pid, momentum, pileup.float()))
    predictions["cls_id"] = torch.argmax(pid[..., 1:], dim=-1) + 1
    return predictions, torch.softmax(presence, dim=-1)[..., 1]


def add_metrics(accumulator, batch_metrics):
    for name, (total, count) in batch_metrics.items():
        values = accumulator.setdefault(name, [0.0, 0.0])
        values[0] += total
        values[1] += count


def main():
    args = parse_args()
    config = MLPFConfig.model_validate(yaml.safe_load(args.config.read_text()))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    model = MLPF(config)
    model.task_loss_weighter = make_task_loss_weighter(config.task_loss_weights.model_dump())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    version = config.test_dataset[args.sample].version
    dataset = PFDataset(
        str(args.data_dir),
        f"{args.sample}/{args.split_config}:{version}",
        args.dataset_split,
        num_samples=args.num_events,
        pad_to_multiple=config.pad_to_multiple_elements,
        feature_dim=config.input_dim,
        max_open_readers=config.max_open_readers,
        build_target_set=True,
    ).ds
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=Collater(["X", "ytarget_set"], []),
        num_workers=0,
    )

    count = int(round((args.threshold_stop - args.threshold_start) / args.threshold_step))
    thresholds = [round(args.threshold_start + index * args.threshold_step, 6) for index in range(count + 1)]
    accumulators = {threshold: {} for threshold in thresholds}

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_predictions = model(batch.X, batch.mask).predictions
            predictions, presence_probability = physical_set_predictions(raw_predictions)
            targets = unpack_target(batch.ytarget_set.float(), None)
            targets["pt"] = torch.exp(targets["pt"].clamp(-20.0, 20.0))
            targets["energy"] = torch.exp(targets["energy"].clamp(-20.0, 20.0))
            for threshold in thresholds:
                metrics = compute_validation_particle_metrics(
                    targets,
                    batch.target_mask,
                    predictions,
                    presence_probability >= threshold,
                    num_classes=config.num_classes,
                )
                add_metrics(accumulators[threshold], metrics)
            if batch_index % 25 == 0 or batch_index == len(loader):
                print(f"processed {batch_index}/{len(loader)} batches", flush=True)

    output = {
        "checkpoint": str(args.checkpoint),
        "sample": args.sample,
        "split_config": args.split_config,
        "dataset_split": args.dataset_split,
        "num_events": len(dataset),
        "thresholds": {},
    }
    for threshold, metrics in accumulators.items():
        output["thresholds"][str(threshold)] = {name: total / count if count else None for name, (total, count) in metrics.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
