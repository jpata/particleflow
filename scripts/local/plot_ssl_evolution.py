"""Plot saved SSL embeddings and supervised probes without GPU or TFDS access."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare_embeddings(reference, current):
    """Align rotations for display; CKA measures similarity without that alignment."""
    ref = reference.astype(np.float64) - reference.mean(0)
    cur = current.astype(np.float64) - current.mean(0)
    cross = cur.T @ ref
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    denom = np.linalg.norm(ref.T @ ref) * np.linalg.norm(cur.T @ cur)
    cka = float(np.square(cross).sum() / max(denom, 1e-30))
    cosine = np.sum(cur * ref, axis=1) / np.maximum(np.linalg.norm(cur, axis=1) * np.linalg.norm(ref, axis=1), 1e-12)
    return cur @ (u @ vt), {"linear_cka_to_first": cka, "centered_cosine_to_first": float(cosine.mean()), "feature_std": float(current.std(0).mean())}


def plot_embeddings(run, output, max_panels=6):
    paths = sorted((run / "embeddings").glob("step-*.npz"), key=lambda p: int(p.stem.split("-")[-1]))
    if not paths:
        return
    with np.load(paths[0], allow_pickle=False) as data:
        reference = data["student"].copy()
        cohort = str(data["cohort_sha256"])
        types = data["X"][:, 0].copy()
    centered = reference - reference.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, -2:][:, ::-1]
    selected = set(np.linspace(0, len(paths) - 1, min(max_panels, len(paths)), dtype=int).tolist())
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), squeeze=False, sharex=True, sharey=True)
    history = []
    panel = 0
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            if str(data["cohort_sha256"]) != cohort:
                raise ValueError(f"Different cohort in {path}")
            aligned, metrics = compare_embeddings(reference, data["student"])
            metrics["step"] = int(data["step"])
            history.append(metrics)
            if index in selected:
                projected = aligned @ basis
                ax = axes[0, panel]
                for kind in np.unique(types):
                    mask = types == kind
                    ax.scatter(projected[mask, 0], projected[mask, 1], s=3, alpha=0.4, label=f"type {int(kind)}")
                ax.set_title(f"Step {metrics['step']}\nCKA={metrics['linear_cka_to_first']:.3f}")
                ax.set_xlabel("Reference PC1")
                panel += 1
    axes[0, 0].set_ylabel("Reference PC2")
    axes[0, 0].legend(markerscale=3)
    fig.suptitle("Fixed input rows; centered and rotation-aligned to the first snapshot")
    fig.tight_layout()
    fig.savefig(output / "embeddings.png", dpi=160)
    plt.close(fig)
    (output / "embedding_history.json").write_text(json.dumps(history, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(
        axes,
        ("linear_cka_to_first", "centered_cosine_to_first", "feature_std"),
        ("Linear CKA to first", "Centered cosine to first", "Mean feature standard deviation"),
    ):
        ax.plot([r["step"] for r in history], [r[key] for r in history], "o-")
        ax.set(xlabel="SSL step", ylabel=label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "embedding_drift.png", dpi=160)
    plt.close(fig)


def plot_probes(run, output):
    paths = sorted((run / "probes").glob("step-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    if not paths:
        return
    rows = [json.loads(p.read_text()) for p in paths]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(
        axes,
        ("unweighted_task_loss", "presence_f1", "pid_accuracy_on_targets"),
        ("Unweighted PF task loss", "Input-aligned presence F1", "PID accuracy on targets"),
    ):
        for stage, name in (("initial", "Before probe training"), ("valid", "Trained probe")):
            values = [r[stage].get(key, np.nan) for r in rows]
            ax.plot([r["ssl_step"] for r in rows], values, "o-", label=name)
        ax.set(xlabel="SSL step", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "probes.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "evolution"
    output.mkdir(parents=True, exist_ok=True)
    plot_embeddings(args.run, output)
    plot_probes(args.run, output)
    print(output)


if __name__ == "__main__":
    main()
