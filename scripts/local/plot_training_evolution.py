#!/usr/bin/env python3
"""Plot key training and jet-validation metrics versus training step."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_FILE_RE = re.compile(r"step_(\d+)\.json$")
JET_METRIC_SUFFIX = "/jet_ratio/jet_ratio_target_to_pred_pt/"


def load_history(history_dir: Path, sample: str) -> dict[str, np.ndarray]:
    """Load losses and jet metrics from step-indexed training history files."""
    rows = []
    for path in history_dir.glob("step_*.json"):
        match = STEP_FILE_RE.fullmatch(path.name)
        if match is None:
            continue

        with path.open() as handle:
            data = json.load(handle)

        metric_prefix = f"step/{sample}{JET_METRIC_SUFFIX}"
        rows.append(
            {
                "step": int(match.group(1)),
                "train_loss": data.get("train", {}).get("Total", np.nan),
                "valid_loss": data.get("valid", {}).get("Total", np.nan),
                "match_frac": data.get(f"{metric_prefix}match_frac", np.nan),
                "response_iqr": data.get(f"{metric_prefix}iqr", np.nan),
                "response_median": data.get(f"{metric_prefix}med", np.nan),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No step_*.json history files found in {history_dir}")

    rows.sort(key=lambda row: row["step"])
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]
    }


def plot_training_evolution(history: dict[str, np.ndarray], output_dir: Path) -> None:
    """Create a four-panel summary and save both PNG and PDF versions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = history["step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(steps, history["train_loss"], marker="o", label="Train")
    axes[0, 0].plot(steps, history["valid_loss"], marker="s", label="Validation")
    axes[0, 0].set_ylabel("Total loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(steps, history["match_frac"], marker="o", color="tab:green")
    axes[0, 1].set_ylabel("Jet matching fraction")
    axes[0, 1].set_ylim(0.0, 1.05)

    axes[1, 0].plot(steps, history["response_iqr"], marker="o", color="tab:red")
    axes[1, 0].set_ylabel("Jet response IQR")
    axes[1, 0].set_xlabel("Training step")

    axes[1, 1].plot(steps, history["response_median"], marker="o", color="tab:purple")
    axes[1, 1].axhline(
        1.0, color="black", linestyle="--", linewidth=1, label="Ideal response"
    )
    axes[1, 1].set_ylabel("Median jet response")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    fig.suptitle("Training evolution")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"training_evolution.{extension}", dpi=150, bbox_inches="tight"
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sample", required=True, help="Test sample used in the history metric keys"
    )
    args = parser.parse_args()

    history = load_history(args.history_dir, args.sample)
    plot_training_evolution(history, args.output_dir)
    print(
        f"Training evolution plot written to {args.output_dir / 'training_evolution.png'}"
    )


if __name__ == "__main__":
    main()
