"""Measure real CLD SSL steps on repeated high-occupancy events, including Adam state."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from mlpf.model.mlpf import MLPF
from mlpf.model.ssl import StudentTeacher, collate_inputs
from mlpf.pretrain import PretrainConfig, build_model_config, load_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/cld_ssl.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument("--scan-per-dataset", type=int, default=128)
    args = parser.parse_args()
    config = PretrainConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
    resolved = build_model_config(config)
    data = load_inputs(config, resolved, config.train_split, args.scan_per_dataset)
    largest = max((data[i] for i in range(len(data))), key=lambda r: int((r["X"][:, 0] != 0).sum()))
    student = MLPF(resolved).cuda().eval()
    first = {k: v.cuda() for k, v in collate_inputs([largest]).items()}
    dtype = getattr(torch, config.dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        dim = student.encode_backbone(first["X"], first["X"][..., 0] != 0, first["source_id"]).shape[-1]
    model = StudentTeacher(student, dim, config.ssl).cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    free, total = torch.cuda.mem_get_info()
    report = {
        "gpu": torch.cuda.get_device_name(),
        "total_bytes": total,
        "free_before_bytes": free,
        "config": config.model_dump(mode="json"),
        "scan_events": len(data),
        "stress_event_elements": int((largest["X"][:, 0] != 0).sum()),
        "results": [],
    }
    for size in args.batches:
        batch = None
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = {k: v.cuda() for k, v in collate_inputs([largest] * size).items()}
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    loss, _ = model(**batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                model.update_teacher()
            torch.cuda.synchronize()
            result = {
                "batch_size": size,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "seconds_per_step": (time.perf_counter() - start) / 3,
            }
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": size, "oom": True}
        report["results"].append(result)
        print(json.dumps(result), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del batch
        if result.get("oom") or result["peak_reserved_bytes"] > total * 0.8:
            break


if __name__ == "__main__":
    main()
