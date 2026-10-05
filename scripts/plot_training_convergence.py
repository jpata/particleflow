#!/usr/bin/env python3
"""Plot sampled training-batch losses against held-out full-split validation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSSES = ("Total", "Classification_binary", "Classification", "Regression_pt",
          "Regression_energy", "Regression_eta", "Regression_sin_phi", "Regression_cos_phi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir
    accumulator = EventAccumulator(str(root / "runs/train"), size_guidance={"scalars": 0}).Reload()
    histories = sorted((root / "history").glob("step_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    validation = [(int(p.stem.split("_")[-1]), json.loads(p.read_text())["valid"]) for p in histories]
    if not validation:
        raise ValueError("No validation histories found")
    config = json.loads((root / "hyperparameters.json").read_text())
    calibration = config["task_loss_weights"]["calibration_steps"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for name, ax in zip(LOSSES, axes.flat):
        points = accumulator.Scalars("step/loss_" + name)
        x = np.array([point.step for point in points])
        y = np.array([point.value for point in points])
        assert np.isfinite(y).all(), name
        # Weighted total is not comparable across task-weight calibration.
        selected = x > calibration if name == "Total" else np.ones(len(x), dtype=bool)
        x, y = x[selected], y[selected]
        ax.plot(x, y, color="C0", alpha=.25, linewidth=1, label="Training batches")
        if len(y) >= 5:
            ax.plot(x[4:], np.convolve(y, np.ones(5) / 5, mode="valid"), color="C0", label="Train: 5-point mean")
        vx = [step for step, _ in validation]
        vy = [losses[name] for _, losses in validation]
        assert np.isfinite(vy).all(), name
        ax.plot(vx, vy, "o-", color="C1", label="Held-out: 100 events")
        ax.set(title=name.replace("_", " "), xlabel="Optimizer step", ylabel="Loss")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("ColliderML ttbar: 900 train / 100 held-out events\nTotal loss shown only after task-weight calibration", fontsize=14)
    fig.savefig(root / "convergence.png", dpi=150)
    plt.close(fig)
    first, last = validation[0][1], validation[-1][1]
    summary = {"experiment_dir": str(root.resolve()), "task_weight_calibration_step": calibration,
               "validation": [{"step": step, "losses": {key: float(losses[key]) for key in LOSSES}}
                              for step, losses in validation],
               "validation_total_relative_change": float(last["Total"] / first["Total"] - 1),
               "caveat": "Training curves are sampled batches, validation is the full fixed held-out split. This tiny sample cannot establish physics convergence."}
    (root / "convergence.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
