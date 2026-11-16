"""Measure real CLD SSL steps on repeated high-occupancy events, including Adam state."""

import argparse
import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from mlpf.model.mlpf import MLPF
from mlpf.model.losses import make_task_loss_weighter
from mlpf.model.ssl import StudentTeacher, collate_inputs, encode_ssl_embeddings
from mlpf.model.ssl_probe import fit_probe, load_probe_data
from mlpf.pretrain import PretrainConfig, build_model_config, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/cld_ssl.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--scan-per-dataset", type=int, default=128)
    parser.add_argument("--probe-batches", nargs="*", type=int, default=[])
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--probe-chunk-sizes", nargs="+", type=int)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    config = PretrainConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
    resolved = build_model_config(config)
    data = load_inputs(config, resolved, config.train_split, args.scan_per_dataset)
    counts = [int((data[i]["X"][:, 0] != 0).sum()) for i in range(len(data))]
    largest_index = int(np.argmax(counts))
    largest = data[largest_index]
    print(json.dumps({"scan_events": len(data), "elements_quantiles": np.quantile(counts, [0, 0.5, 0.95, 1]).tolist()}), flush=True)
    student = MLPF(resolved).cuda().eval()
    student.task_loss_weighter = make_task_loss_weighter(resolved.task_loss_weights.model_dump()).cuda()
    first = {k: v.cuda() for k, v in collate_inputs([largest]).items()}
    dtype = getattr(torch, config.dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        dim = encode_ssl_embeddings(student, first["X"], first["X"][..., 0] != 0, first["source_id"]).shape[-1]
    model = StudentTeacher(student, dim, config.ssl).cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay, fused=True)
    free, total = torch.cuda.mem_get_info()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "total_bytes": total,
        "free_before_bytes": free,
        "config": config.model_dump(mode="json"),
        "scan_events": len(data),
        "elements_quantiles": np.quantile(counts, [0, 0.5, 0.95, 1]).tolist(),
        "stress_event_index": largest_index,
        "stress_event_elements": int((largest["X"][:, 0] != 0).sum()),
        "results": [],
        "probe_results": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for size in args.batches:
        batch = None
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = {k: v.cuda() for k, v in collate_inputs([largest] * size).items()}
            torch.cuda.synchronize()
            durations = []
            for iteration in range(args.iterations + 1):
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    loss, _ = model(**batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                model.update_teacher()
                torch.cuda.synchronize()
                if iteration:  # Exclude warmup/Adam-state allocation from throughput.
                    durations.append(time.perf_counter() - start)
            result = {
                "batch_size": size,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "seconds_per_step": float(np.median(durations)),
                "events_per_second": size / float(np.median(durations)),
            }
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": size, "oom": True}
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        save()
        del batch
        if result.get("oom") or result["peak_reserved_bytes"] > total * 0.8:
            break

    if args.probe_batches:
        optimizer.zero_grad(set_to_none=True)
        labeled = load_probe_data(config, resolved, config.train_split, args.scan_per_dataset)[largest_index]
        chunks = args.probe_chunk_sizes or [resolved.model.set_decoder.hit_grouping_chunk_size if resolved.model.set_decoder else None]
        for chunk, size in itertools.product(chunks, args.probe_batches):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            probe = config.probe.model_copy(deep=True)
            probe.batch_size, probe.steps = size, args.probe_steps
            probe.hit_pairs.enabled = False
            if chunk is not None:
                model.student.config.set_decoder.hit_grouping_chunk_size = chunk
            start = time.perf_counter()
            timings = {}
            try:
                _, fitted = fit_probe(model.student, [labeled] * size, [labeled], probe, torch.device("cuda"), dtype, timings=timings)
                torch.cuda.synchronize()
                result = {
                    "batch_size": size,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "seconds_total": time.perf_counter() - start,
                    "updates": args.probe_steps,
                    "hit_grouping_chunk_size": chunk,
                    "timings": timings,
                }
                del fitted
            except torch.cuda.OutOfMemoryError:
                result = {"batch_size": size, "oom": True}
            report["probe_results"].append(result)
            print(json.dumps({"probe": result}), flush=True)
            save()
            if result.get("oom") or result["peak_reserved_bytes"] > total * 0.8:
                break


if __name__ == "__main__":
    main()
