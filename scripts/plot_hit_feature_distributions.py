#!/usr/bin/env python3
"""Plot raw and forward-pass engineered hit-feature distributions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow_datasets as tfds
import torch

from mlpf.conf import EDM4HEP
from mlpf.model.mlpf import HitFeatureEngineering


COLORS = {"tracker": "#0072B2", "calorimeter": "#D55E00"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="TFDS data root")
    parser.add_argument("--dataset", default="cld_edm_ttbar_hits/10:3.2.1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-events", type=int, default=40)
    parser.add_argument("--max-tracker-per-event", type=int, default=2000)
    parser.add_argument("--max-calo-per-event", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sample_rows(values, mask, maximum, rng):
    indices = np.flatnonzero(mask)
    if len(indices) > maximum:
        indices = rng.choice(indices, size=maximum, replace=False)
    return values[indices]


def collect_distributions(args):
    builder = tfds.builder(args.dataset, data_dir=str(args.data_dir))
    source = builder.as_data_source(split=args.split)
    if len(source) == 0:
        raise ValueError(f"Dataset {args.dataset} split {args.split} is empty")

    rng = np.random.default_rng(args.seed)
    num_events = min(args.num_events, len(source))
    event_indices = np.sort(rng.choice(len(source), size=num_events, replace=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = HitFeatureEngineering().eval().to(device)

    raw_names = EDM4HEP.HitFeatures.get_names()
    geometry_names = list(layer.GEOMETRY_FEATURE_NAMES)
    tracker_names = list(layer.tracker_neighborhood.OUTPUT_FEATURE_NAMES)
    calo_names = list(layer.calorimeter_neighborhood.OUTPUT_FEATURE_NAMES)
    offsets = np.cumsum(
        [0, len(raw_names), len(geometry_names), len(tracker_names), len(calo_names)]
    )

    chunks = {
        "raw_tracker": [],
        "raw_calorimeter": [],
        "geometry_tracker": [],
        "geometry_calorimeter": [],
        "tracker": [],
        "calorimeter": [],
    }
    total_hits = {"tracker": 0, "calorimeter": 0}
    sampled_hits = {"tracker": 0, "calorimeter": 0}

    with torch.no_grad():
        for event_index in event_indices:
            raw = np.asarray(source[int(event_index)]["X"], dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != len(raw_names):
                raise ValueError(
                    f"Event {event_index} has unexpected X shape {raw.shape}"
                )
            if not np.isfinite(raw).all():
                raise ValueError(
                    f"Event {event_index} contains non-finite raw features"
                )

            tensor = torch.from_numpy(raw).unsqueeze(0).to(device)
            mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=device)
            engineered = layer(tensor, mask)[0].cpu().numpy()
            if not np.isfinite(engineered).all():
                raise ValueError(
                    f"Event {event_index} contains non-finite engineered features"
                )

            tracker_mask = raw[:, 0] == 1
            calo_mask = raw[:, 0] == 2
            total_hits["tracker"] += int(tracker_mask.sum())
            total_hits["calorimeter"] += int(calo_mask.sum())

            tracker_rows = sample_rows(
                engineered, tracker_mask, args.max_tracker_per_event, rng
            )
            calo_rows = sample_rows(engineered, calo_mask, args.max_calo_per_event, rng)
            sampled_hits["tracker"] += len(tracker_rows)
            sampled_hits["calorimeter"] += len(calo_rows)

            chunks["raw_tracker"].append(tracker_rows[:, offsets[0] : offsets[1]])
            chunks["raw_calorimeter"].append(calo_rows[:, offsets[0] : offsets[1]])
            chunks["geometry_tracker"].append(tracker_rows[:, offsets[1] : offsets[2]])
            chunks["geometry_calorimeter"].append(calo_rows[:, offsets[1] : offsets[2]])
            chunks["tracker"].append(tracker_rows[:, offsets[2] : offsets[3]])
            chunks["calorimeter"].append(calo_rows[:, offsets[3] : offsets[4]])

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    groups = [
        (
            "raw",
            raw_names,
            {
                "tracker": arrays["raw_tracker"],
                "calorimeter": arrays["raw_calorimeter"],
            },
        ),
        (
            "geometry",
            geometry_names,
            {
                "tracker": arrays["geometry_tracker"],
                "calorimeter": arrays["geometry_calorimeter"],
            },
        ),
        ("tracker", tracker_names, {"tracker": arrays["tracker"]}),
        ("calorimeter", calo_names, {"calorimeter": arrays["calorimeter"]}),
    ]
    summary = {
        "dataset": builder.info.full_name,
        "split": args.split,
        "event_indices": event_indices.tolist(),
        "device": str(device),
        "total_hits": total_hits,
        "sampled_hits": sampled_hits,
        "num_features": int(offsets[-1]),
    }
    return groups, summary


def robust_edges(populations, bins=60):
    finite = [values[np.isfinite(values)] for values in populations if len(values)]
    combined = np.concatenate(finite) if finite else np.zeros(1)
    unique = np.unique(combined)
    if len(unique) <= 20 and np.allclose(unique, np.round(unique), atol=1.0e-6):
        lower = math.floor(float(unique.min())) - 0.5
        upper = math.ceil(float(unique.max())) + 0.5
        return np.arange(lower, upper + 1.0, 1.0)
    lower, upper = np.quantile(combined, [0.001, 0.999])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        center = float(combined[0]) if len(combined) else 0.0
        width = max(abs(center) * 0.05, 0.5)
        lower, upper = center - width, center + width
    return np.linspace(lower, upper, bins + 1)


def feature_stats(values):
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, [0.001, 0.01, 0.5, 0.99, 0.999])
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q0.1%": float(quantiles[0]),
        "q1%": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q99%": float(quantiles[3]),
        "q99.9%": float(quantiles[4]),
        "max": float(np.max(values)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def plot_feature(ax, name, populations, feature_index):
    values_by_population = {
        label: values[:, feature_index] for label, values in populations.items()
    }
    edges = robust_edges(list(values_by_population.values()))
    for label, values in values_by_population.items():
        clipped = values[(values >= edges[0]) & (values <= edges[-1])]
        weights = np.full(len(clipped), 1.0 / max(len(values), 1))
        ax.hist(
            clipped,
            bins=edges,
            weights=weights,
            histtype="step",
            linewidth=1.4,
            color=COLORS[label],
            label=label,
        )
    ax.set_title(name, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("fraction / bin", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.2)
    if len(populations) > 1:
        ax.legend(fontsize=7)


def save_group_grids(output_dir, group_name, feature_names, populations, chunk_size=25):
    outputs = []
    for chunk_index, start in enumerate(
        range(0, len(feature_names), chunk_size), start=1
    ):
        stop = min(start + chunk_size, len(feature_names))
        nplots = stop - start
        ncols = 5
        nrows = math.ceil(nplots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.0 * nrows), squeeze=False)
        for axis, feature_index in zip(axes.flat, range(start, stop)):
            plot_feature(axis, feature_names[feature_index], populations, feature_index)
        for axis in axes.flat[nplots:]:
            axis.set_visible(False)
        fig.suptitle(
            f"{group_name.capitalize()} hit features ({start + 1}–{stop} of {len(feature_names)})",
            fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        suffix = f"_{chunk_index}" if len(feature_names) > chunk_size else ""
        output = output_dir / f"{group_name}_features{suffix}.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        outputs.append(output)
    return outputs


def save_complete_pdf(output_dir, groups):
    output = output_dir / "all_hit_feature_distributions.pdf"
    with PdfPages(output) as pdf:
        for group_name, feature_names, populations in groups:
            for start in range(0, len(feature_names), 4):
                stop = min(start + 4, len(feature_names))
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), squeeze=False)
                for axis, feature_index in zip(axes.flat, range(start, stop)):
                    plot_feature(
                        axis, feature_names[feature_index], populations, feature_index
                    )
                for axis in axes.flat[stop - start :]:
                    axis.set_visible(False)
                fig.suptitle(
                    f"{group_name.capitalize()} features ({start + 1}–{stop} of {len(feature_names)})"
                )
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(fig)
                plt.close(fig)
    return output


def save_statistics(output_dir, groups):
    output = output_dir / "feature_statistics.csv"
    fieldnames = [
        "group",
        "feature",
        "population",
        "count",
        "mean",
        "std",
        "min",
        "q0.1%",
        "q1%",
        "median",
        "q99%",
        "q99.9%",
        "max",
        "zero_fraction",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_name, feature_names, populations in groups:
            for population, values in populations.items():
                for feature_index, feature_name in enumerate(feature_names):
                    writer.writerow(
                        {
                            "group": group_name,
                            "feature": feature_name,
                            "population": population,
                            **feature_stats(values[:, feature_index]),
                        }
                    )
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups, summary = collect_distributions(args)
    outputs = []
    for group_name, feature_names, populations in groups:
        outputs.extend(
            save_group_grids(args.output_dir, group_name, feature_names, populations)
        )
    outputs.append(save_complete_pdf(args.output_dir, groups))
    outputs.append(save_statistics(args.output_dir, groups))

    print(f"Dataset: {summary['dataset']} ({summary['split']})")
    print(
        f"Device: {summary['device']}; events: {len(summary['event_indices'])}; features: {summary['num_features']}"
    )
    print(
        f"Total hits: {summary['total_hits']}; sampled hits: {summary['sampled_hits']}"
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
