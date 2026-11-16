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
    cosine = np.sum(cur * ref, axis=1) / np.maximum(
        np.linalg.norm(cur, axis=1) * np.linalg.norm(ref, axis=1), 1e-12
    )
    return cur @ (u @ vt), {
        "linear_cka_to_first": cka,
        "centered_cosine_to_first": float(cosine.mean()),
        "feature_std": float(current.std(0).mean()),
    }


def plot_embeddings(run, output, max_panels=6, max_step=None):
    paths = snapshot_paths(run, max_step)
    if not paths:
        return
    with np.load(paths[0], allow_pickle=False) as data:
        reference = data["student"].copy()
        cohort = str(data["cohort_sha256"])
        types = data["X"][:, 0].copy()
    centered = reference - reference.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, -2:][:, ::-1]
    selected = set(
        np.linspace(0, len(paths) - 1, min(max_panels, len(paths)), dtype=int).tolist()
    )
    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(4 * len(selected), 4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
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
                    ax.scatter(
                        projected[mask, 0],
                        projected[mask, 1],
                        s=3,
                        alpha=0.4,
                        label=f"type {int(kind)}",
                    )
                ax.set_title(
                    f"Step {metrics['step']}\nCKA={metrics['linear_cka_to_first']:.3f}"
                )
                ax.set_xlabel("Reference PC1")
                panel += 1
    axes[0, 0].set_ylabel("Reference PC2")
    axes[0, 0].legend(markerscale=3)
    fig.suptitle(
        "Fixed input rows; centered and rotation-aligned to the first snapshot"
    )
    fig.tight_layout()
    fig.savefig(output / "embeddings.png", dpi=160)
    plt.close(fig)
    (output / "embedding_history.json").write_text(json.dumps(history, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, label in zip(
        axes,
        ("linear_cka_to_first", "centered_cosine_to_first", "feature_std"),
        (
            "Linear CKA to first",
            "Centered cosine to first",
            "Mean feature standard deviation",
        ),
    ):
        ax.plot([r["step"] for r in history], [r[key] for r in history], "o-")
        ax.set(xlabel="SSL step", ylabel=label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "embedding_drift.png", dpi=160)
    plt.close(fig)


def probe_plot_fields(result):
    if "particle_f1" in result:
        return (
            ("unweighted_task_loss", "Unweighted PF task loss"),
            ("particle_f1", "Matched particle F1"),
            ("matched_pid_accuracy", "PID accuracy on matched particles"),
        )
    return (
        ("unweighted_task_loss", "Unweighted PF task loss"),
        ("presence_f1", "Input-aligned presence F1"),
        ("pid_accuracy_on_targets", "PID accuracy on targets"),
    )


def plot_hit_pair_probes(rows, output):
    rows = [r for r in rows if "hit_pairs" in r]
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, detector in enumerate(("tracker", "calorimeter")):
        for j, metric in enumerate(("average_precision", "roc_auc")):
            ax = axes[i, j]
            for representation in ("raw", "embedding"):
                values = [r["hit_pairs"]["detectors"][detector].get(representation, {}).get("mean", {}).get(metric, np.nan) for r in rows]
                ax.plot([r["ssl_step"] for r in rows], values, "o-", label=representation)
            if metric == "average_precision":
                prevalence = [r["hit_pairs"]["detectors"][detector].get("raw", {}).get("mean", {}).get("positive_fraction", np.nan) for r in rows]
                ax.plot([r["ssl_step"] for r in rows], prevalence, "--", label="positive-pair fraction")
            ax.set(xlabel="SSL step", ylabel=metric.replace("_", " "), title=detector)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("Frozen hit-pair probes (oracle foreground; held-out events)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"hit_pair_probes.{extension}", dpi=160)
    plt.close(fig)


def plot_probes(run, output, max_step=None):
    paths = sorted(
        (run / "probes").glob("step-*.json"), key=lambda p: int(p.stem.split("-")[-1])
    )
    if not paths:
        return
    paths = [
        p for p in paths if max_step is None or int(p.stem.split("-")[-1]) <= max_step
    ]
    if not paths:
        return
    rows = [json.loads(p.read_text()) for p in paths]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    plot_hit_pair_probes(rows, output)
    for ax, (key, label) in zip(axes, probe_plot_fields(rows[-1]["valid"])):
        plotted = False
        for stage, name in (
            ("initial", "Before probe training"),
            ("valid", "Trained probe"),
        ):
            values = [r[stage].get(key, np.nan) for r in rows]
            if not any(v is not None and np.isfinite(v) for v in values):
                continue
            ax.plot([r["ssl_step"] for r in rows], values, "o-", label=name)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "No defined values yet", transform=ax.transAxes, ha="center")
        ax.set(xlabel="SSL step", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "probes.png", dpi=160)
    plt.close(fig)


def snapshot_paths(run, max_step=None):
    paths = sorted(
        (run / "embeddings").glob("step-*.npz"),
        key=lambda p: int(p.stem.split("-")[-1]),
    )
    return [
        p for p in paths if max_step is None or int(p.stem.split("-")[-1]) <= max_step
    ]


def load_cohort(paths):
    snapshots = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            snapshot = {key: data[key].copy() for key in data.files}
        if snapshots:
            for key in (
                "cohort_sha256",
                "X",
                "dataset_index",
                "event_index",
                "element_index",
            ):
                if not np.array_equal(snapshot[key], snapshots[0][key]):
                    raise ValueError(f"Changed cohort field {key} in {path}")
        if not np.isfinite(snapshot["student"]).all():
            raise ValueError(f"Nonfinite embeddings in {path}")
        snapshots.append(snapshot)
    return snapshots


def nearest_neighbors(embedding, k, metric):
    """Exact neighbors with stable tie breaking and explicit self exclusion."""
    x = embedding.astype(np.float64)
    if metric == "cosine":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        distance = 1 - x @ x.T
    else:
        norms = np.square(x).sum(1)
        distance = norms[:, None] + norms[None, :] - 2 * x @ x.T
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1, kind="stable")[:, : min(k, len(x) - 1)]


def neighbor_overlap(a, b):
    if not a.shape[1]:
        return np.nan
    return float((a[:, :, None] == b[:, None, :]).any(axis=2).mean())


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_neighborhoods(snapshots, output, neighbors=15, max_elements=2000):
    if not snapshots:
        return
    if len(snapshots[0]["X"]) < 2:
        print("Skipping nearest-neighbor diagnostics: fewer than two cohort elements")
        return
    # One fixed uniform subsample bounds quadratic distance memory for hit inputs.
    rng = np.random.default_rng(27182)
    indices = np.sort(
        rng.choice(
            len(snapshots[0]["X"]),
            min(max_elements, len(snapshots[0]["X"])),
            replace=False,
        )
    )
    types = snapshots[0]["X"][indices, 0]
    groups = [("All elements", np.ones(len(indices), dtype=bool))]
    groups += [
        (f"Type {int(kind)}", types == kind)
        for kind in np.unique(types)
        if (types == kind).sum() >= 2
    ]
    steps = [int(s["step"]) for s in snapshots]
    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(4 * len(groups), 8),
        squeeze=False,
        layout="constrained",
    )
    results = {
        "steps": steps,
        "selected_rows": indices.tolist(),
        "requested_k": neighbors,
        "groups": {},
    }
    for col, (label, mask) in enumerate(groups):
        k = min(neighbors, int(mask.sum()) - 1)
        results["groups"][label] = {"elements": int(mask.sum()), "k": k}
        for row, metric in enumerate(("cosine", "euclidean")):
            knn = [
                nearest_neighbors(s["student"][indices[mask]], k, metric)
                for s in snapshots
            ]
            overlap = np.asarray([[neighbor_overlap(a, b) for b in knn] for a in knn])
            results["groups"][label][metric] = overlap.tolist()
            ax = axes[row, col]
            im = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis", origin="lower")
            ax.set(
                title=f"{label}: {metric}, k={k}",
                xlabel="SSL step",
                ylabel="SSL step",
                xticks=range(len(steps)),
                xticklabels=steps,
                yticks=range(len(steps)),
                yticklabels=steps,
            )
            ax.tick_params(axis="x", labelrotation=60)
    fig.colorbar(im, ax=axes, label="Fraction of shared nearest neighbors", shrink=0.75)
    fig.suptitle(
        "Local neighborhood stability across checkpoints\nNeighbors drawn across the fixed cohort; type panels restrict both queries and candidates"
    )
    save_figure(fig, output, "neighborhood_stability")
    (output / "neighborhood_history.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def plot_physics(snapshots, output):
    """Explicitly opt-in: columns follow the CLD track/cluster schema."""
    if not snapshots:
        return
    reference = snapshots[0]["student"]
    centered = reference.astype(np.float64) - reference.mean(0)
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
    basis = vectors[:, -2:][:, ::-1]
    X = snapshots[0]["X"]
    colors = [
        (np.log10(np.maximum(X[:, 1], 1e-6)), "log10(input pT / ET [GeV])", "viridis"),
        (X[:, 2], "Input eta", "coolwarm"),
        (np.arctan2(X[:, 3], X[:, 4]), "Input phi [rad]", "twilight"),
    ]
    chosen = np.unique(
        np.linspace(0, len(snapshots) - 1, min(3, len(snapshots)), dtype=int)
    )
    fig, axes = plt.subplots(
        3,
        len(chosen),
        figsize=(4.5 * len(chosen), 10),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row, (values, label, cmap) in enumerate(colors):
        low, high = (-np.pi, np.pi) if row == 2 else np.percentile(values, [1, 99])
        for col, index in enumerate(chosen):
            aligned, _ = compare_embeddings(reference, snapshots[index]["student"])
            projected = aligned @ basis
            ax = axes[row, col]
            im = ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=values,
                cmap=cmap,
                vmin=low,
                vmax=high,
                s=6,
                alpha=0.7,
                rasterized=True,
            )
            ax.set(
                title=f"Step {int(snapshots[index]['step'])}",
                xlabel="Reference PC1",
                ylabel="Reference PC2",
            )
        fig.colorbar(
            im,
            ax=axes[row, :],
            label=label,
            extend="neither" if row == 2 else "both",
            shrink=0.8,
        )
    explained = eigenvalues[-2:].sum() / max(eigenvalues.sum(), 1e-30)
    fig.suptitle(
        f"Physics structure in CLD embeddings | reference PCs explain {explained:.1%} of initial variance\n"
        "Fixed rows and color scales; rotation aligned; pT/ET and eta colors clipped at cohort 1st/99th percentiles"
    )
    save_figure(fig, output, "embedding_physics")


def precision_recall(target, score):
    """Threshold sweep grouped at tied scores; AP uses a stepwise PR integral."""
    order = np.argsort(-score, kind="stable")
    labels, scores = target[order], score[order]
    end = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tp = np.cumsum(labels)[end]
    precision = tp / (end + 1)
    recall = tp / max(int(target.sum()), 1)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision)) if target.any() else None
    return np.r_[0, recall], np.r_[1, precision], ap


def probe_group_metrics(data, mask):
    target = data["targets"][:, 0].astype(int)
    present = target != 0
    predicted = data["presence_logits"].argmax(-1) != 0
    tp = int((mask & present & predicted).sum())
    fp = int((mask & ~present & predicted).sum())
    fn = int((mask & present & ~predicted).sum())
    positive = mask & present
    correct = int(((data["pid_logits"].argmax(-1) == target) & positive).sum())
    return {
        "elements": int(mask.sum()),
        "targets": int(positive.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "presence_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "pid_accuracy": correct / int(positive.sum()) if positive.any() else None,
    }


def quantile_groups(values, label):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
    if len(edges) == 1:
        return [(f"{label}={edges[0]:.2g}", np.ones(len(values), dtype=bool))]
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return [
        (f"{edges[i]:.2g}–{edges[i + 1]:.2g}", bins == i) for i in range(len(edges) - 1)
    ]


def plot_probe_details(run, output, max_step=None, cld=False):
    paths = sorted(
        (run / "probes").glob("step-*-predictions.npz"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    paths = [
        p for p in paths if max_step is None or int(p.name.split("-")[1]) <= max_step
    ]
    data, steps = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            row = {key: archive[key].copy() for key in archive.files}
        if str(row["output_mode"]) != "elementwise":
            print("Skipping input-aligned probe details for set outputs")
            return
        if not (
            np.array_equal(row["input_offsets"], row["prediction_offsets"])
            and np.array_equal(row["input_offsets"], row["target_offsets"])
        ):
            raise ValueError(f"Unaligned prediction axes in {path}")
        if data:
            for key in (
                "X",
                "targets",
                "dataset_index",
                "event_index",
                "input_offsets",
            ):
                if not np.array_equal(row[key], data[0][key]):
                    raise ValueError(f"Changed probe cohort field {key} in {path}")
        data.append(row)
        steps.append(int(path.name.split("-")[1]))
    if not data:
        return
    X = data[0]["X"]
    types = [(f"Type {int(kind)}", X[:, 0] == kind) for kind in np.unique(X[:, 0])]
    groupings = [("Input type", types)]
    if cld:
        groupings += [
            ("Input pT / ET [GeV]", quantile_groups(X[:, 1], "pT/ET")),
            ("Input eta", quantile_groups(X[:, 2], "eta")),
        ]
    fig, axes = plt.subplots(
        2,
        len(groupings),
        figsize=(5 * len(groupings), 8),
        squeeze=False,
        layout="constrained",
    )
    summary = {"steps": steps, "events": len(data[0]["dataset_index"]), "groups": {}}
    for col, (name, groups) in enumerate(groupings):
        metrics = [[probe_group_metrics(d, mask) for _, mask in groups] for d in data]
        summary["groups"][name] = {
            "labels": [label for label, _ in groups],
            "metrics_by_step": metrics,
        }
        for row, (key, title) in enumerate(
            (
                ("presence_f1", "Presence F1"),
                ("pid_accuracy", "PID accuracy on targets"),
            )
        ):
            values = np.array(
                [
                    [m[key] if m[key] is not None else np.nan for m in ms]
                    for ms in metrics
                ]
            )
            ax = axes[row, col]
            im = ax.imshow(
                values, vmin=0, vmax=1, aspect="auto", cmap="viridis", origin="lower"
            )
            ax.set(
                title=f"{title}\n{name}",
                ylabel="SSL step",
                xticks=range(len(groups)),
                xticklabels=[f"{label}\nn={mask.sum()}" for label, mask in groups],
                yticks=range(len(steps)),
                yticklabels=steps,
            )
            for i in range(len(steps)):
                for j in range(len(groups)):
                    if np.isfinite(values[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{values[i, j]:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="black" if values[i, j] > 0.55 else "white",
                        )
    fig.colorbar(im, ax=axes, shrink=0.7, label="Metric value")
    fig.suptitle(
        f"Trained probe subgroups | fixed export cohort: {summary['events']} events, {len(X)} elements\n"
        "Input bins; n counts input elements. PID excludes no-target rows. Descriptive estimates without uncertainty intervals."
    )
    save_figure(fig, output, "probe_subgroups")
    groups = [("All elements", np.ones(len(X), dtype=bool))] + types
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(5 * len(groups), 4.5),
        squeeze=False,
        layout="constrained",
    )
    summary["precision_recall"] = {}
    # Keep curves readable on long runs; subgroup JSON and heatmaps retain every probe.
    selected = np.unique(np.linspace(0, len(data) - 1, min(6, len(data)), dtype=int))
    for ax, (name, mask) in zip(axes[0], groups):
        summary["precision_recall"][name] = []
        for i, d in enumerate(data):
            logits = d["presence_logits"][mask].astype(np.float64)
            exp = np.exp(logits - logits.max(1, keepdims=True))
            score = exp[:, 1] / exp.sum(1)
            target = d["targets"][mask, 0] != 0
            recall, precision, ap = precision_recall(target, score)
            summary["precision_recall"][name].append(
                {"step": steps[i], "average_precision": ap}
            )
            if i in selected:
                (line,) = ax.step(
                    recall,
                    precision,
                    where="pre",
                    label=f"{steps[i]}: AP={ap:.3f}"
                    if ap is not None
                    else f"{steps[i]}: no positives",
                )
                m = probe_group_metrics(d, mask)
                ax.plot(
                    m["tp"] / max(m["tp"] + m["fn"], 1),
                    m["tp"] / max(m["tp"] + m["fp"], 1),
                    "o",
                    color=line.get_color(),
                    markersize=4,
                )
        ax.axhline(
            float((data[0]["targets"][mask, 0] != 0).mean()),
            color="gray",
            ls=":",
            label="Prevalence",
        )
        ax.set(
            title=name, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02)
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Presence precision–recall on the fixed exported cohort | dots: probability threshold 0.5"
    )
    save_figure(fig, output, "probe_precision_recall")
    (output / "probe_subgroups.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot_training_and_probes(run, output, max_step=None):
    """Refresh the combined dashboard from complete records in the live log."""
    path = run / "metrics.jsonl"
    if not path.exists():
        return
    records = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # The training process may be writing the final record.
            raise
        if max_step is None or record["step"] <= max_step:
            records.append(record)
    train = [r for r in records if "train" in r]
    valid = [r for r in records if "valid" in r]
    probes = [r for r in records if "probe" in r]
    if not train:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), layout="constrained")
    region_mode = "occupancy_loss" in train[-1]["train"]
    keys = ("loss", "occupancy_loss", "energy_loss") if region_mode else ("loss", "teacher_entropy", "effective_prototypes")
    titles = ("Masked-region total objective", "Region occupancy BCE", "Region log-energy Huber loss") if region_mode else (
        "Distillation cross-entropy", "Mean teacher entropy (nats)", "Effective prototype count")
    for ax, key, title in zip(axes[0], keys, titles):
        for rows, stage, style, label in (
            (train, "train", "-", "Training"),
            (valid, "valid", "o-", "Validation"),
        ):
            ax.plot(
                [r["step"] for r in rows],
                [r[stage][key] for r in rows],
                style,
                label=label,
                linewidth=1,
            )
        ax.set(xlabel="SSL step", title=title)
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    for ax, (key, title) in zip(axes[1], probe_plot_fields(probes[-1]["probe"]["valid"] if probes else {})):
        available = [r for r in probes if r["probe"]["valid"].get(key) is not None and np.isfinite(r["probe"]["valid"][key])]
        if available:
            ax.plot(
                [r["step"] for r in available],
                [r["probe"]["valid"][key] for r in available],
                "o-",
                color="#2a8c65",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No compatible probe results yet",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set(xlabel="SSL step", title=title)
        ax.grid(alpha=0.2)
    detail = f"Training through step {train[-1]['step']}"
    if probes:
        last = probes[-1]["probe"]
        detail += f" | probe through {probes[-1]['step']}: {last['steps']} updates, {last['valid']['events']} validation events"
    fig.suptitle(f"SSL and frozen-backbone probes\n{detail}")
    save_figure(fig, output, "training_and_probes")
    summary = {
        "latest_training_step": train[-1]["step"],
        "validation_steps": [r["step"] for r in valid],
        "probe_steps": [r["step"] for r in probes],
    }
    (output / "training_and_probes.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-step", type=int, help="Analyze a fixed prefix of a running job"
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=12,
        help="Bound pairwise neighborhood comparisons",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=2000,
        help="Fixed subsample for quadratic neighbor search",
    )
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument(
        "--input-schema",
        choices=["cld-track-cluster"],
        help="Enable schema-specific physics maps and probe bins",
    )
    args = parser.parse_args()
    if args.max_snapshots < 2 or args.max_elements < 2 or args.neighbors < 1:
        parser.error(
            "max-snapshots and max-elements must be >= 2; neighbors must be >= 1"
        )
    output = args.output or args.run / "evolution"
    output.mkdir(parents=True, exist_ok=True)
    paths = snapshot_paths(args.run, args.max_step)
    selected = np.unique(
        np.linspace(0, len(paths) - 1, min(args.max_snapshots, len(paths)), dtype=int)
    )
    snapshots = load_cohort([paths[i] for i in selected])
    plot_embeddings(args.run, output, max_step=args.max_step)
    plot_probes(args.run, output, max_step=args.max_step)
    plot_training_and_probes(args.run, output, max_step=args.max_step)
    plot_neighborhoods(snapshots, output, args.neighbors, args.max_elements)
    if args.input_schema == "cld-track-cluster":
        plot_physics(snapshots, output)
    plot_probe_details(
        args.run,
        output,
        max_step=args.max_step,
        cld=args.input_schema == "cld-track-cluster",
    )
    print(output)


if __name__ == "__main__":
    main()
