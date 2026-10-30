"""Export portable per-worker benchmark results and allocation work totals.

Usage: python -m mlpf.benchmark_report experiments/<scenario-or-run>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_results(root):
    rows = []
    for path in sorted(Path(root).glob("**/benchmark/*/summary.json")):
        summary = json.loads(path.read_text())
        metadata = json.loads((path.parent / "metadata.json").read_text())
        manifest_path = path.parents[2] / "scenario-manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        job = manifest.get("job", {})
        environment = metadata.get("environment", {})
        training = summary.get("device_phase_metrics", {}).get("training_loop", {})
        rows.append(
            {
                "run_dir": str(path.parents[2]),
                "variant": job.get("variant_name"),
                "platform": job.get("platform_name"),
                "seed": job.get("seed"),
                "device_name": (metadata.get("device") or {}).get("name"),
                "device_total_memory_bytes": (metadata.get("device") or {}).get("total_memory_bytes"),
                "architecture": metadata.get("architecture"),
                "torch_version": metadata.get("torch"),
                "cuda_runtime": metadata.get("cuda_runtime"),
                "hip_runtime": metadata.get("hip_runtime"),
                "dtype": metadata.get("config", {}).get("dtype"),
                "num_workers": metadata.get("config", {}).get("num_workers"),
                "git_revision": metadata.get("git", {}).get("revision"),
                "git_dirty": bool(metadata.get("git", {}).get("status")),
                "train_dataset_json": json.dumps(metadata.get("config", {}).get("train_dataset"), sort_keys=True),
                "steps": summary.get("steps"),
                "warmup_steps_excluded": summary.get("warmup_steps_excluded"),
                "global_batch_size": job.get("global_batch_size"),
                "session_dir": str(path.parent),
                "job_id": environment.get("SLURM_JOB_ID"),
                "array_task": environment.get("SLURM_ARRAY_TASK_ID"),
                "hostname": metadata.get("hostname"),
                "rank": summary["rank"],
                "world_size": metadata.get("world_size", 1),
                "status": summary["status"],
                "events": summary["events"],
                "hits": summary["hits"],
                "worker_elapsed_seconds": summary["worker_elapsed_seconds"],
                "started_at_unix_time": summary["started_at_unix_time"],
                "finished_at_unix_time": summary["finished_at_unix_time"],
                "steady_events_per_training_second": summary["steady_events_per_training_second"],
                "steady_hits_per_training_second": summary["steady_hits_per_training_second"],
                "steady_step_p50_seconds": (summary["steady_training_step_seconds"] or {}).get("p50"),
                "steady_step_p99_seconds": (summary["steady_training_step_seconds"] or {}).get("p99"),
                "training_gpu_busy_percent": training.get("time_weighted_gpu_busy_percent"),
                "training_gpu_busy_covered_seconds": training.get("gpu_busy_covered_seconds"),
                "training_estimated_energy_joules": training.get("estimated_energy_joules"),
                "training_power_covered_seconds": training.get("power_covered_seconds"),
                "device_memory_peak_bytes": summary.get("device_metric_peaks", {}).get("memory_used_bytes"),
                "framework_process_peak_allocated_bytes": (summary.get("framework_memory_bytes") or {}).get("process_peak_allocated_bytes"),
                "telemetry_errors": summary["telemetry_error_count"],
            }
        )
    return rows


def allocation_totals(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run_dir"], row["job_id"], row["array_task"])].append(row)
    results = []
    for (run_dir, job_id, task), workers in groups.items():
        span = max(row["finished_at_unix_time"] for row in workers) - min(row["started_at_unix_time"] for row in workers)
        events = sum(row["events"] for row in workers)
        ranks = {(row["hostname"], row["rank"]) for row in workers}
        results.append(
            {
                "run_dir": run_dir,
                "job_id": job_id,
                "array_task": task,
                "worker_sessions": len(workers),
                "observed_workers": len(ranks),
                "expected_world_size": max(row["world_size"] for row in workers),
                "events_processed_including_retries": events,
                "hits_processed_including_retries": sum(row["hits"] for row in workers),
                "observed_worker_span_seconds": span,
                "events_per_observed_span_second": events / span if span > 0 else None,
                "recorded_gpu_worker_hours": sum(row["worker_elapsed_seconds"] for row in workers if row["rank"] != "cpu") / 3600,
                "all_sessions_completed": all(row["status"] == "completed" for row in workers),
            }
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    rows = load_results(args.directory)
    if not rows:
        parser.error("No completed benchmark session summaries found")
    report = {
        "schema_version": 1,
        "workers": rows,
        "allocations": allocation_totals(rows),
        "notes": [
            "Per-worker durations are not summed to estimate elapsed wall time; actual work is summed across ranks.",
            "Recorded GPU-worker hours exclude launcher/import time before recorder creation; use Slurm for allocation charges.",
            "Device telemetry is device/package scope, not per-process attribution; energy estimates cover only reported intervals.",
            "Retry sessions count actual compute, including reprocessed events. Observed spans can include gaps between sessions.",
        ],
    }
    destination = args.directory / "benchmark-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    with (args.directory / "benchmark-report.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['run_dir']} rank={row['rank']} status={row['status']} events={row['events']} events/s={row['steady_events_per_training_second']} GPU busy={row['training_gpu_busy_percent']}"
        )
    print("Wrote", destination)


if __name__ == "__main__":
    main()
