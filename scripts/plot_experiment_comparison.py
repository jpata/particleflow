#!/usr/bin/env python3
"""Compare training histories from hit and track/cluster MLPF experiments."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"step_(\d+)\.json$")
LOSS_COMPONENTS = (
    "Classification_binary",
    "Classification",
    "Regression_pt",
    "Regression_eta",
    "Regression_sin_phi",
    "Regression_cos_phi",
    "Regression_energy",
    "Total",
)
SAMPLES = ("ttbar", "ww_fullhad", "qq")
JET_METRICS = (
    ("med", "Median response"),
    ("iqr", "Response IQR"),
    ("match_frac", "Match fraction"),
)


def load_history(path: Path) -> list[tuple[int, dict]]:
    history = []
    for filename in path.glob("step_*.json"):
        match = STEP_RE.fullmatch(filename.name)
        if match:
            with filename.open() as handle:
                history.append((int(match.group(1)), json.load(handle)))
    if not history:
        raise FileNotFoundError(f"No step_*.json files found in {path}")
    return sorted(history)


def jet_value(record: dict, sample: str, input_suffix: str, metric: str) -> float:
    suffix = f"_{sample}_{input_suffix}/jet_ratio/jet_ratio_target_to_pred_pt/{metric}"
    values = [value for key, value in record.items() if key.startswith("step/") and key.endswith(suffix)]
    return float(values[0]) if len(values) == 1 else np.nan


def plot_losses(histories: dict[str, list[tuple[int, dict]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for component, axis in zip(LOSS_COMPONENTS, axes.flat):
        for label, history in histories.items():
            steps = [step for step, _ in history]
            train = [record["train"].get(component, np.nan) for _, record in history]
            valid = [record["valid"].get(component, np.nan) for _, record in history]
            line = axis.plot(steps, train, marker="o", markersize=3, label=f"{label}, train")[0]
            axis.plot(steps, valid, marker="s", markersize=3, linestyle="--", color=line.get_color(), label=f"{label}, validation")
        axis.set_title(component.replace("_", " "))
        axis.set_xlabel("Training step")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="x", style="plain")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Hit versus track/cluster training losses")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"loss_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_jet_metrics(histories: dict[str, list[tuple[int, dict]]], input_suffixes: dict[str, str], output_dir: Path) -> None:
    fig, axes = plt.subplots(len(SAMPLES), len(JET_METRICS), figsize=(15, 11), sharex=True)
    for row, sample in enumerate(SAMPLES):
        for column, (metric, metric_label) in enumerate(JET_METRICS):
            axis = axes[row, column]
            for label, history in histories.items():
                steps = [step for step, _ in history]
                values = [jet_value(record, sample, input_suffixes[label], metric) for _, record in history]
                axis.plot(steps, values, marker="o", markersize=4, label=label)
            axis.set_title(f"{sample}: {metric_label}")
            axis.set_xlabel("Training step")
            axis.set_ylabel(metric_label)
            axis.grid(alpha=0.25)
            axis.ticklabel_format(axis="x", style="plain")
            if metric == "med":
                axis.axhline(1.0, color="black", linewidth=1, linestyle=":")
            if metric == "match_frac":
                axis.set_ylim(0.0, 1.05)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Hit versus track/cluster jet validation metrics")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"jet_metric_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hit-history", required=True, type=Path)
    parser.add_argument("--pf-history", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = {
        "Hit inputs": load_history(args.hit_history),
        "Track/cluster inputs": load_history(args.pf_history),
    }
    plot_losses(histories, args.output_dir)
    plot_jet_metrics(histories, {"Hit inputs": "hits", "Track/cluster inputs": "pf"}, args.output_dir)


if __name__ == "__main__":
    main()
