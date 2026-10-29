"""Extract allocation-request evidence from completed GH200 TensorBoard logs."""

import argparse
import csv
import json
import statistics
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def describe(values):
    if not values:
        return None
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=100)
    args = parser.parse_args()
    summaries, raw = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for manifest_path in sorted(args.scenario_dir.glob("*/scenario-manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        variant = manifest["job"]["variant_name"]
        acc = EventAccumulator(str(manifest_path.parent / "runs/train"), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        selected = [tag for tag in tags if tag.startswith(("system/gpu_utilization", "step/", "diagnostic/train_step/time/"))]
        series = {tag: acc.Scalars(tag) for tag in selected}
        for tag, events in series.items():
            raw.extend((variant, tag, event.step, event.wall_time, event.value) for event in events)
        val_freq = manifest["resolved_config"]["val_freq"]
        gpu_events = series.get("system/gpu_utilization_avg", [])
        # Windows immediately after validation also include validation, test and
        # plotting activity. Sample counts per window were not saved, so these
        # are means of window means, not exact time-weighted job averages.
        train_windows = []
        for previous, event in pairwise(gpu_events):
            crosses_validation = (previous.step - 1) // val_freq != (event.step - 1) // val_freq
            if previous.step >= args.warmup_steps and not crosses_validation:
                train_windows.append(event.value)
        measured = {tag: [event.value for event in events if event.step > args.warmup_steps] for tag, events in series.items()}
        step_times = measured.get("step/time_train_total", [])
        elems = measured.get("step/num_elems", [])
        summary = {
            "variant": variant,
            "gpu_busy_training_window_means_percent": describe(train_windows),
            "gpu_busy_all_postwarmup_window_means_percent": describe(measured.get("system/gpu_utilization_avg", [])),
            "largest_recorded_gpu_busy_sample_percent": max(measured.get("system/gpu_utilization_max", []), default=None),
            "sampled_training_step_seconds": describe(step_times),
            "sampled_data_load_seconds": describe(measured.get("step/time_data_load", [])),
            "sampled_hits_per_batch": describe(elems),
            "sampled_hits_per_second": statistics.mean(elems) / statistics.mean(step_times) if elems and step_times else None,
            "phase_mean_seconds": {tag: statistics.mean(values) for tag, values in measured.items() if tag.startswith("diagnostic/train_step/time/") and values},
            "run_dir": str(manifest_path.parent),
        }
        summaries.append(summary)
        for axis, tag in zip(axes, ("system/gpu_utilization_avg", "step/time_train_total")):
            events = [event for event in series.get(tag, []) if event.step > args.warmup_steps]
            axis.plot([event.step for event in events], [event.value for event in events], label=variant, linewidth=1.3)
    for axis, ylabel in zip(axes, ("GPU busy time (%) — window means", "Sampled training step (seconds)")):
        axis.set(xlabel="Training step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylim(0, 100)
    fig.suptitle("GH200, batch 8 — utilization windows can include validation/test work")
    fig.savefig(args.scenario_dir / "gpu-utilization.png", dpi=160)
    payload = {
        "notes": [
            "GPU busy time is not SM occupancy or percent of peak FLOPs.",
            "Window sample counts were not saved; window means cannot yield an exact time-weighted job average.",
            "Training-window statistics exclude warmup and windows spanning validation steps.",
            "Step timing uses logged steps and includes profiling/logging overhead; validation/checkpoints are excluded.",
        ],
        "runs": summaries,
    }
    (args.scenario_dir / "gpu-utilization.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.scenario_dir / "gpu-utilization-scalars.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "tag", "step", "wall_time", "value"))
        writer.writerows(raw)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
