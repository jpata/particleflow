"""Report sampled training throughput, GPU memory and validation for a scenario."""

import argparse
import json
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize_run(manifest_path, warmup_steps):
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    train_dir = run_dir / "runs/train"
    accumulator = EventAccumulator(str(train_dir), size_guidance={"scalars": 0}) if train_dir.exists() else None
    if accumulator is not None:
        accumulator.Reload()
    tags = accumulator.Tags()["scalars"] if accumulator is not None else []

    def values(tag):
        return accumulator.Scalars(tag) if tag in tags else []

    times = values("step/time_train_total")
    measured = [event.value for event in times if event.step > warmup_steps]
    batch = manifest["job"]["global_batch_size"]
    mean_time = statistics.mean(measured) if measured else None
    peak = [event.value for tag in tags if "max_allocated_MB" in tag for event in values(tag)]
    histories = []
    for path in sorted((run_dir / "history").glob("step_*.json"), key=lambda path: int(path.stem.split("_")[-1])):
        try:
            histories.append((path, json.loads(path.read_text())))
        except json.JSONDecodeError:
            continue  # A running job may still be writing its latest history.
    last_history = histories[-1][1] if histories else {}
    return {
        "variant": manifest["job"]["variant_name"],
        "seed": manifest["job"]["seed"],
        "global_batch_size": batch,
        "last_logged_step": max((event.step for event in times), default=0),
        "warmup_steps_excluded": warmup_steps,
        "measured_step_samples": len(measured),
        "mean_training_step_seconds": mean_time,
        "training_events_per_second": batch / mean_time if mean_time else None,
        "peak_gpu_allocated_mib": max(peak) if peak else None,
        "last_validation_step": int(histories[-1][0].stem.split("_")[-1]) if histories else None,
        "validation_loss_unweighted": last_history.get("valid", {}).get("Total_unweighted"),
        "run_dir": str(run_dir),
    }


def display(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()
    runs = [summarize_run(path, args.warmup_steps) for path in sorted(args.scenario_dir.glob("*/scenario-manifest.json"))]
    output = args.scenario_dir / "benchmark-summary.json"
    output.write_text(json.dumps(runs, indent=2) + "\n")
    print("Training-step timings exclude validation/checkpoint work; first", args.warmup_steps, "steps excluded.")
    print("variant  step  events/s  step seconds  peak GPU MiB  unweighted validation loss")
    for run in runs:
        print(
            "  ".join(
                display(run[key])
                for key in (
                    "variant",
                    "last_logged_step",
                    "training_events_per_second",
                    "mean_training_step_seconds",
                    "peak_gpu_allocated_mib",
                    "validation_loss_unweighted",
                )
            )
        )
    print("Wrote", output)


if __name__ == "__main__":
    main()
