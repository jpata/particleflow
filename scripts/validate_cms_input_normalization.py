"""Check CMS input normalization on TFDS train events and plot feature distributions.

Example:
    python scripts/validate_cms_input_normalization.py \
      --data-dir /local/joosep/mlpf/tensorflow_datasets/cms/2.8.0 \
      --output-dir plots/cms_input_normalization_2.8.0

Events are evenly spaced within every config. Plot and summary weights account for the
number of train events in each config; this is a diagnostic sample, not a replacement for
the full-split statistics in mlpf/model/input_normalization_stats/cms_2.8.0.yaml.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlpf.conf import SOURCE_IDS, X_FEATURES, load_stats, stats_path
from mlpf.model.input_normalization import InputNormalization
from compute_input_normalization_stats import open_builds

FEATURES = {
    "track": ("pt", "e", "eta", "px", "num_hits", "pterror"),
    "cluster": ("pt", "e", "px", "corr_energy", "pterror", "sigma_x"),
}
STAGES = ("raw", "transformed", "normalized")
COLORS = {"raw": "#687b91", "transformed": "#c77935", "normalized": "#277c9d"}


def reference_normalization(raw, stats):
    """Independent NumPy implementation of the recorded per-column transforms."""
    transformed = raw.astype(np.float64, copy=True)
    normalized = raw.astype(np.float64, copy=True)
    groups = {"track": raw[:, 0] == 1, "cluster": raw[:, 0] >= 2}
    for group, mask in groups.items():
        if not mask.any():
            continue
        for column in stats[group]:
            index = column["column"]
            values = raw[mask, index].astype(np.float64)
            if column["clip"] is not None:
                values = np.clip(values, *column["clip"])
            if column["log1p_scale"] is not None:
                values = np.log1p(values / column["log1p_scale"])
            transformed[mask, index] = values
            normalized[mask, index] = (values - column["mean"]) / column["std"] if column["std"] > 0 else 0.0
    return transformed, normalized, groups


def collect(builders, events_per_config, plot_rows_per_event):
    stats = load_stats("cms", "2.8.0")
    model = InputNormalization(len(X_FEATURES["cms"]), {"cms": "2.8.0"}).eval()
    source_id = torch.tensor([SOURCE_IDS["cms"]])
    rng = np.random.default_rng(280)
    columns = {group: {c["name"]: c["column"] for c in stats[group]} for group in FEATURES}
    plotted = {group: {feature: {stage: [] for stage in STAGES} | {"weight": []} for feature in features} for group, features in FEATURES.items()}
    moments = {group: {"count": 0.0, "sum": np.zeros(len(X_FEATURES["cms"])), "sum_sq": np.zeros(len(X_FEATURES["cms"]))} for group in FEATURES}
    summary = {
        "population_events": 0,
        "sampled_events": 0,
        "configs": 0,
        "rows": {"track": 0, "cluster": 0},
        "nonfinite_outputs": 0,
        "max_abs_error": 0.0,
        "mismatched_values": 0,
    }

    with torch.no_grad():
        for builder in builders:
            full_size = builder.info.splits["train"].num_examples
            selected_size = min(full_size, events_per_config)
            positions = np.linspace(0, full_size - 1, selected_size, dtype=np.int64)
            event_weight = full_size / selected_size
            source = builder.as_data_source(split="train")
            summary["population_events"] += full_size
            summary["sampled_events"] += selected_size
            summary["configs"] += 1
            for position in positions:
                raw = np.asarray(source[int(position)]["X"], dtype=np.float32)
                transformed, expected, groups = reference_normalization(raw, stats)
                actual = model(torch.from_numpy(raw).unsqueeze(0), source_id).squeeze(0).numpy()
                difference = np.abs(actual.astype(np.float64) - expected)
                summary["max_abs_error"] = max(summary["max_abs_error"], float(difference.max(initial=0)))
                summary["mismatched_values"] += int(np.count_nonzero(difference > 1e-4 + 1e-5 * np.abs(expected)))
                summary["nonfinite_outputs"] += int(np.count_nonzero(~np.isfinite(actual)))
                for group, mask in groups.items():
                    rows = np.flatnonzero(mask)
                    if not len(rows):
                        continue
                    summary["rows"][group] += len(rows)
                    values = actual[rows].astype(np.float64)
                    moments[group]["count"] += event_weight * len(rows)
                    moments[group]["sum"] += event_weight * values.sum(axis=0)
                    moments[group]["sum_sq"] += event_weight * np.square(values).sum(axis=0)
                    chosen = rng.choice(rows, size=min(plot_rows_per_event, len(rows)), replace=False)
                    for feature in FEATURES[group]:
                        index = columns[group][feature]
                        item = plotted[group][feature]
                        item["raw"].append(raw[chosen, index].astype(np.float64))
                        item["transformed"].append(transformed[chosen, index])
                        item["normalized"].append(actual[chosen, index].astype(np.float64))
                        item["weight"].append(np.full(len(chosen), event_weight * len(rows) / len(chosen)))
            print(f"{summary['configs']:2d}/{len(builders)} {builder.info.full_name}: {selected_size} train events", flush=True)

    for group, values in moments.items():
        mean = values["sum"] / values["count"]
        std = np.sqrt(np.maximum(values["sum_sq"] / values["count"] - mean**2, 0.0))
        summary[f"{group}_all_feature_sample_moments"] = {
            column["name"]: {"mean": float(mean[column["column"]]), "std": float(std[column["column"]])} for column in stats[group]
        }
        summary[f"{group}_narrow_features"] = [column["name"] for column in stats[group] if column["std"] > 0 and std[column["column"]] < 0.5]
        summary[f"{group}_sample_moments"] = {
            feature: {"mean": float(mean[columns[group][feature]]), "std": float(std[columns[group][feature]])} for feature in FEATURES[group]
        }
    return plotted, summary


def plot_group(group, plotted, summary, output_dir):
    fig, axes = plt.subplots(len(FEATURES[group]), len(STAGES), figsize=(15, 15))
    for row, feature in enumerate(FEATURES[group]):
        item = plotted[group][feature]
        weights = np.concatenate(item["weight"])
        for col, stage in enumerate(STAGES):
            ax = axes[row, col]
            values = np.concatenate(item[stage])
            low, high = np.quantile(values, [0.005, 0.995])
            if low == high:
                low, high = low - 0.5, high + 0.5
            visible = (values >= low) & (values <= high)
            ax.hist(values[visible], bins=60, weights=weights[visible], density=True, color=COLORS[stage], histtype="stepfilled", alpha=0.8)
            ax.set_title(f"{feature}: {stage}", fontsize=10)
            ax.set_ylabel("density")
            ax.grid(alpha=0.2)
            if stage == "normalized":
                ax.axvline(0, color="#242b34", linewidth=0.8)
                moments = summary[f"{group}_sample_moments"][feature]
                ax.set_xlabel(f"sample mean {moments['mean']:.2f}, std {moments['std']:.2f}")
            else:
                ax.set_xlabel("value (central 99% shown)")
    fig.suptitle(
        f"CMS 2.8.0 {group} inputs: raw → transformed → normalized\n"
        f"{summary['sampled_events']:,} events across {summary['configs']} configs; weighted by train-event count",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / f"{group}_distributions.svg"
    fig.savefig(path)
    plt.close(fig)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-per-config", type=int, default=20)
    parser.add_argument("--plot-rows-per-event", type=int, default=200)
    args = parser.parse_args()
    if args.events_per_config < 1 or args.plot_rows_per_event < 1:
        parser.error("sampling counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    builders = open_builds("cms", "2.8.0", args.data_dir)
    plotted, summary = collect(builders, args.events_per_config, args.plot_rows_per_event)
    summary["statistics_sha256"] = hashlib.sha256(stats_path("cms", "2.8.0").read_bytes()).hexdigest()
    summary["sampling"] = {
        "method": "evenly_spaced_train_indices_per_config",
        "events_per_config": args.events_per_config,
        "max_plot_rows_per_event_and_group": args.plot_rows_per_event,
        "plot_percentile_range": [0.5, 99.5],
    }
    path = args.output_dir / "validation.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for group in FEATURES:
        plot_group(group, plotted, summary, args.output_dir)
    print(path)
    if summary["mismatched_values"] or summary["nonfinite_outputs"]:
        raise SystemExit("CMS input normalization validation failed")


if __name__ == "__main__":
    main()
