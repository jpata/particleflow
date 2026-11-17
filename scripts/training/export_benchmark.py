"""Export compact, portable results from labeled benchmark sweeps (no model/data dumps)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def write_csv(path, rows, fields=()):
    keys = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text())


def collect_sweep(label, root):
    """Export explicit data/stack provenance, but no home paths, device UUIDs or logs."""
    root = Path(root)
    state = read_json(root / "sweep-state.json")
    cases, steps, validation = [], [], []
    for path in sorted((root / "runs").glob("*/*/benchmark/*/summary.json")):
        summary = read_json(path)
        metadata = read_json(path.parent / "metadata.json")
        config = metadata["config"]
        manifest = read_json(path.parents[2] / "scenario-manifest.json")
        job = manifest["job"]
        variant, batch, seed = job["variant_name"], job["global_batch_size"], job["seed"]
        matching = [case for case in state["cases"] if (case["variant"], case["global_batch_size"], case["seed"]) == (variant, batch, seed)]
        if len(matching) != 1:
            raise ValueError(f"Ambiguous case identity in {root}: {variant}, {batch}, {seed}")
        case = matching[0]
        key = {
            "platform": label,
            "variant": variant,
            "batch": batch,
            "seed": seed,
            "run_id": path.parents[2].name,
            "session_id": path.parent.name,
            "rank": summary["rank"],
        }
        train = summary.get("device_phase_metrics", {}).get("training_loop", {})
        record = {
            **key,
            "status": case["status"],
            "worker_status": summary["status"],
            "source_sweep": root.name,
            "world_size": metadata["world_size"],
            "device": metadata["device"],
            "architecture": metadata["architecture"],
            "python": metadata["python"],
            "torch": metadata["torch"],
            "cuda_runtime": metadata.get("cuda_runtime"),
            "hip_runtime": metadata.get("hip_runtime"),
            "packages": metadata.get("packages", {}),
            "git": {k: metadata.get("git", {}).get(k) for k in ("revision", "tracked_diff_sha256")},
            "dirty": bool(metadata.get("git", {}).get("status")),
            "train_dataset": config["train_dataset"],
            "valid_dataset": config["valid_dataset"],
            "test_dataset": config["test_dataset"],
            "model": config["model"],
            "runtime": {k: config.get(k) for k in ("dtype", "compile", "num_workers", "prefetch_factor", "max_open_readers")},
            "training": {
                k: config.get(k) for k in ("num_steps", "optimizer", "lr", "lr_schedule", "weight_decay", "task_loss_weights", "nvalid", "ntest")
            },
            "steps": summary["steps"],
            "warmup_steps": summary["warmup_steps_excluded"],
            "events": summary["events"],
            "hits": summary["hits"],
            "started_at_unix_time": summary["started_at_unix_time"],
            "finished_at_unix_time": summary["finished_at_unix_time"],
            "worker_seconds": summary["worker_elapsed_seconds"],
            "case_seconds": case["finished_at"] - case["started_at"],
            "events_per_second": summary["steady_events_per_training_second"],
            "hits_per_second": summary["steady_hits_per_training_second"],
            "step_p50_seconds": (summary["steady_training_step_seconds"] or {}).get("p50"),
            "step_p99_seconds": (summary["steady_training_step_seconds"] or {}).get("p99"),
            "framework_memory_bytes": summary.get("framework_memory_bytes"),
            "device_memory_peak_bytes": summary.get("device_metric_peaks", {}).get("memory_used_bytes"),
            "training_device_metrics": train,
            "phase_exclusive_seconds": summary["phase_exclusive_seconds"],
            "data_verification": state.get("data_verification", {"status": "not_recorded"}),
            "scenario_sha256": state.get("scenario_sha256"),
            "spec_sha256": state.get("spec_sha256"),
        }
        record["device"] = {k: v for k, v in (record["device"] or {}).items() if k != "uuid"}
        record["source_summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["source_metadata_sha256"] = hashlib.sha256((path.parent / "metadata.json").read_bytes()).hexdigest()
        cases.append(record)
        raw_steps = []
        for line in (path.parent / "steps.jsonl").read_text().splitlines():
            step = json.loads(line)
            if "duration_seconds" not in step:
                continue  # CUDA timing records are separate from host-step measurements.
            raw_steps.append(step)
            steps.append(
                {
                    **key,
                    **{k: step[k] for k in ("step", "events", "hits", "padded_hits", "duration_seconds", "data_wait_seconds")},
                    "total_unweighted": step.get("losses", {}).get("Total_unweighted"),
                }
            )
        record["sequence_lengths_sha256"] = (
            hashlib.sha256(json.dumps([step["sequence_lengths"] for step in raw_steps], separators=(",", ":")).encode()).hexdigest()
            if raw_steps and all("sequence_lengths" in step for step in raw_steps)
            else None
        )
        if case["status"] == "completed":
            if len(raw_steps) != summary["steps"] or sum(step["events"] for step in raw_steps) != summary["events"]:
                raise ValueError(f"Raw step/event counts disagree with the summary for {label}/{variant}/batch{batch}")
            steady = raw_steps[summary["warmup_steps_excluded"] :]
            duration = sum(step["duration_seconds"] for step in steady)
            for field, metric in (("events", "steady_events_per_training_second"), ("hits", "steady_hits_per_training_second")):
                if not duration or not math.isclose(sum(step[field] for step in steady) / duration, summary[metric], rel_tol=1e-10):
                    raise ValueError(f"Raw throughput disagrees with {metric} for {label}/{variant}/batch{batch}")
        for history_path in sorted((path.parents[2] / "history").glob("step_*.json")):
            history = read_json(history_path)
            validation.append(
                {
                    **key,
                    "step": int(history_path.stem.removeprefix("step_")),
                    "train_total_unweighted": history.get("train", {}).get("Total_unweighted"),
                    "valid_total_unweighted": history.get("valid", {}).get("Total_unweighted"),
                }
            )
    if not cases:
        raise ValueError(f"No worker summaries found in {root}")
    return cases, steps, validation


def export(sweeps, output, release):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing releases")
    if len({label for label, _ in sweeps}) != len(sweeps):
        raise ValueError("Platform labels must be unique")
    cases, steps, validation = [], [], []
    for label, root in sweeps:
        c, s, v = collect_sweep(label, root)
        cases.extend(c)
        steps.extend(s)
        validation.extend(v)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps({"schema_version": 1, "release": release, "cases": cases}, indent=2) + "\n")
    flat = []
    for case in cases:
        row = {k: v for k, v in case.items() if not isinstance(v, (dict, list))}
        row.update(
            device_name=case["device"].get("name"),
            device_memory_bytes=case["device"].get("total_memory_bytes"),
            allocated_bytes=(case["framework_memory_bytes"] or {}).get("process_peak_allocated_bytes"),
            reserved_bytes=(case["framework_memory_bytes"] or {}).get("process_peak_reserved_bytes"),
            git_revision=case["git"].get("revision"),
        )
        flat.append(row)
    write_csv(output / "cases.csv", flat)
    write_csv(output / "steps.csv", steps)
    write_csv(
        output / "validation.csv",
        validation,
        fields=("platform", "variant", "batch", "seed", "run_id", "session_id", "rank", "step", "train_total_unweighted", "valid_total_unweighted"),
    )
    shutil.copy2(Path(__file__).with_name("plot_benchmark.py"), output / "render_plots.py")
    return cases


def write_checksums(output):
    output = Path(output)
    entries = [
        {"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(output.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    ]
    (output / "manifest.json").write_text(json.dumps({"files": entries}, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args(argv)
    sweeps = [value.split("=", 1) for value in args.sweep]
    if any(len(item) != 2 or not all(item) for item in sweeps):
        parser.error("Each sweep must be LABEL=PATH")
    cases = export(sweeps, args.output, args.release)
    write_checksums(args.output)
    print(f"Exported {len(cases)} worker sessions to {args.output}")


if __name__ == "__main__":
    main()
