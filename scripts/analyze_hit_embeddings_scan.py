#!/usr/bin/env python3
"""Quantitatively scan hit-embedding structure across backbone layers and training checkpoints.

For a hit-based MLPF experiment, this loads every checkpoint found in
`<exp-dir>/checkpoints/`, runs `MLPF.encode_backbone_layers` on a handful of
events to obtain the per-hit embedding at each backbone stage (input
encoding -> detector-specific tracker/calo attention legs -> shared "common"
attention layers), and for each (checkpoint, layer) computes how well those
embeddings reflect the true generator-particle origin of each hit
(`ytarget[..., particle_number]`):

  - NN same-origin rate: fraction of hits whose single nearest neighbor in
    embedding space truly comes from the same particle (vs. a random
    baseline set by the particle-size distribution of the event).
  - Adjusted Rand Index (ARI) between a KMeans(k=n_true_particles) clustering
    of the embeddings and the true particle labels (chance level = 0).
  - A linear (logistic regression) probe accuracy for the true particle
    class (charged/neutral hadron, photon, electron, muon), cross-validated
    with GroupKFold grouped by particle so hits of one particle never leak
    across train/test (chance level = majority-class fraction).

It then plots each metric vs. training step (one line per layer) and vs.
layer depth (one line per selected checkpoint), and dumps the raw per-event
and pooled-probe numbers to JSON for further inspection.

Example:
    uv run python3 scripts/analyze_hit_embeddings_scan.py \\
        --exp-dir experiments/cld_hits_output_comparison/elementwise_seed12345_20260906_012757_908052 \\
        --num-events 8 --output-dir tmp/hit_embedding_scan
"""

from __future__ import annotations

import argparse
import json
import pickle as pkl
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlpf.conf import CLASS_NAMES_CAPITALIZED
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFDataset
from mlpf.model.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment dir with model_kwargs.pkl and checkpoints/")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*", default=None, help="Subset of checkpoint steps to use (default: all found)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tfds_validation_cld_hits"))
    parser.add_argument("--dataset", default="cld_edm_ttbar_hits/10:3.2.1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-events", type=int, default=8)
    parser.add_argument("--start-event", type=int, default=0)
    parser.add_argument("--min-hits-per-particle", type=int, default=5, help="Min true hits for a particle to enter the metrics")
    parser.add_argument("--max-hits-per-event", type=int, default=4000, help="Subsample kept hits per event to this many, for compute")
    parser.add_argument(
        "--max-probe-samples", type=int, default=20000, help="Subsample the pooled probe set (per checkpoint/layer) to this many rows"
    )
    parser.add_argument("--probe-cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/hit_embedding_scan"))
    return parser.parse_args()


def list_checkpoints(exp_dir: Path, steps: list[int] | None) -> list[Path]:
    ckpt_dir = exp_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("checkpoint-*.pth"), key=lambda p: int(p.stem.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    if steps is not None:
        wanted = set(steps)
        ckpts = [p for p in ckpts if int(p.stem.split("-")[1]) in wanted]
    return ckpts


def build_model(exp_dir: Path, device: str):
    with open(exp_dir / "model_kwargs.pkl", "rb") as f:
        config = pkl.load(f)
    model = MLPF(config)
    model.eval().to(device)
    return model, config


def get_layer_embeddings(model, X: np.ndarray, device: str) -> list[np.ndarray]:
    """Return [N, D] per-hit embeddings at every backbone stage, from input encoding to final layer."""
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = (X_t[..., 0] != 0).float()
    autocast_enabled = device == "cuda"
    with torch.no_grad(), torch.autocast(device_type="cuda" if autocast_enabled else "cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        layers = model.encode_backbone_layers(X_t, mask_t)
    return [layer[0].float().cpu().numpy() for layer in layers]


def build_layer_labels(num_layers: int, use_detector_backbone: bool) -> list[str]:
    labels = ["input_encoding"]
    remaining = num_layers - 1
    if use_detector_backbone and remaining > 0:
        labels.append("detector_legs")
        remaining -= 1
    labels.extend(f"common_{i + 1}" for i in range(remaining))
    return labels


def particle_class_per_hit(ytarget: np.ndarray) -> np.ndarray:
    """Broadcast the true particle class from each particle's representative hit to all its hits. -1 = unknown."""
    particle_number = ytarget[:, 13]
    cls_id = ytarget[:, 0]
    cls_by_particle = {int(pn): int(c) for pn, c in zip(particle_number, cls_id) if pn > 0 and c != 0}
    return np.array([cls_by_particle.get(int(pn), -1) for pn in particle_number])


def prepare_event(X: np.ndarray, ytarget: np.ndarray, min_hits: int, max_hits: int, rng: np.random.Generator) -> dict | None:
    particle_number = ytarget[:, 13]
    valid = particle_number > 0
    counts = Counter(particle_number[valid].astype(int).tolist())
    keep_particles = {p for p, c in counts.items() if c >= min_hits}
    keep_mask = valid & np.isin(particle_number.astype(int), list(keep_particles))
    idx = np.flatnonzero(keep_mask)
    if len(np.unique(particle_number[idx])) < 2:
        return None
    if len(idx) > max_hits:
        idx = np.sort(rng.choice(idx, size=max_hits, replace=False))

    pn = particle_number[idx].astype(int)
    selected_counts = Counter(pn.tolist())
    sizes = np.array([selected_counts[p] for p in pn])
    nn_random_baseline = float(np.mean((sizes - 1) / (len(pn) - 1)))

    return {
        "X": X,
        "idx": idx,
        "pn": pn,
        "n_particles": int(len(np.unique(pn))),
        "cls": particle_class_per_hit(ytarget)[idx],
        "nn_random_baseline": nn_random_baseline,
    }


def compute_layer_metrics(emb_k: np.ndarray, pn: np.ndarray, n_particles: int, seed: int) -> dict:
    nn = NearestNeighbors(n_neighbors=2).fit(emb_k)
    nearest = nn.kneighbors(emb_k, return_distance=False)[:, 1]
    nn_rate = float(np.mean(pn[nearest] == pn))

    km_labels = KMeans(n_clusters=n_particles, n_init=3, random_state=seed).fit_predict(emb_k)
    ari = float(adjusted_rand_score(pn, km_labels))

    return {"nn_same_origin_rate": nn_rate, "ari": ari}


def run_probe(embeddings: np.ndarray, y: np.ndarray, groups: np.ndarray, cv_folds: int, seed: int, max_samples: int, rng: np.random.Generator):
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < cv_folds:
        return None
    if len(y) > max_samples:
        sel = rng.choice(len(y), size=max_samples, replace=False)
        embeddings, y, groups = embeddings[sel], y[sel], groups[sel]

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=seed))
    scores = cross_val_score(pipe, embeddings, y, groups=groups, cv=GroupKFold(n_splits=cv_folds), n_jobs=cv_folds)
    majority_baseline = float(np.max(np.bincount(y)) / len(y))
    return {"probe_acc_mean": float(scores.mean()), "probe_acc_std": float(scores.std()), "probe_baseline": majority_baseline}


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def select_steps_for_layer_plot(steps: list[int], n: int = 4) -> list[int]:
    if len(steps) <= n:
        return steps
    idx = np.linspace(0, len(steps) - 1, n).round().astype(int)
    return sorted({steps[i] for i in idx})


def plot_vs_checkpoint(per_layer_event, probe_records, layer_labels, nn_baseline, out_path):
    steps = sorted({r["step"] for r in per_layer_event})
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for layer_idx, label in enumerate(layer_labels):
        nn_means, nn_sems, ari_means, ari_sems = [], [], [], []
        for step in steps:
            rows = [r for r in per_layer_event if r["step"] == step and r["layer"] == layer_idx]
            m, s = mean_sem([r["nn_same_origin_rate"] for r in rows])
            nn_means.append(m)
            nn_sems.append(s)
            m, s = mean_sem([r["ari"] for r in rows])
            ari_means.append(m)
            ari_sems.append(s)
        axes[0].errorbar(steps, nn_means, yerr=nn_sems, marker="o", label=label)
        axes[1].errorbar(steps, ari_means, yerr=ari_sems, marker="o", label=label)

        probe_steps, probe_means, probe_stds = [], [], []
        for step in steps:
            recs = [r for r in probe_records if r["step"] == step and r["layer"] == layer_idx]
            if recs:
                probe_steps.append(step)
                probe_means.append(recs[0]["probe_acc_mean"])
                probe_stds.append(recs[0]["probe_acc_std"])
        if probe_means:
            axes[2].errorbar(probe_steps, probe_means, yerr=probe_stds, marker="o", color=f"C{layer_idx}")
    for layer_idx, label in enumerate(layer_labels):
        axes[2].plot([], [], marker="o", color=f"C{layer_idx}", label=label)

    axes[0].axhline(nn_baseline, color="gray", ls="--", label="random baseline")
    axes[1].axhline(0.0, color="gray", ls="--", label="chance (ARI=0)")
    baselines = [r["probe_baseline"] for r in probe_records]
    if baselines:
        axes[2].axhline(np.mean(baselines), color="gray", ls="--", label="majority-class baseline")

    axes[0].set_title("NN same-origin rate")
    axes[1].set_title("ARI (KMeans vs. true origin)")
    axes[2].set_title("Linear probe accuracy for true particle class")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vs_layer(per_layer_event, probe_records, layer_labels, nn_baseline, selected_steps, out_path):
    x = np.arange(len(layer_labels))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for step in selected_steps:
        nn_means, nn_sems, ari_means, ari_sems = [], [], [], []
        for layer_idx in range(len(layer_labels)):
            rows = [r for r in per_layer_event if r["step"] == step and r["layer"] == layer_idx]
            m, s = mean_sem([r["nn_same_origin_rate"] for r in rows])
            nn_means.append(m)
            nn_sems.append(s)
            m, s = mean_sem([r["ari"] for r in rows])
            ari_means.append(m)
            ari_sems.append(s)
        axes[0].errorbar(x, nn_means, yerr=nn_sems, marker="o", label=f"step {step}")
        axes[1].errorbar(x, ari_means, yerr=ari_sems, marker="o", label=f"step {step}")

        probe_means, probe_stds = [], []
        for layer_idx in range(len(layer_labels)):
            recs = [r for r in probe_records if r["step"] == step and r["layer"] == layer_idx]
            probe_means.append(recs[0]["probe_acc_mean"] if recs else np.nan)
            probe_stds.append(recs[0]["probe_acc_std"] if recs else 0.0)
        axes[2].errorbar(x, probe_means, yerr=probe_stds, marker="o", label=f"step {step}")

    axes[0].axhline(nn_baseline, color="gray", ls="--", label="random baseline")
    axes[1].axhline(0.0, color="gray", ls="--", label="chance (ARI=0)")
    baselines = [r["probe_baseline"] for r in probe_records]
    if baselines:
        axes[2].axhline(np.mean(baselines), color="gray", ls="--", label="majority-class baseline")

    axes[0].set_title("NN same-origin rate")
    axes[1].set_title("ARI (KMeans vs. true origin)")
    axes[2].set_title("Linear probe accuracy for true particle class")
    for ax in axes:
        ax.set_xticks(x, layer_labels, rotation=30, ha="right")
        ax.set_xlabel("backbone layer")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    checkpoints = list_checkpoints(args.exp_dir, args.checkpoint_steps)
    print(f"Found {len(checkpoints)} checkpoints: {[int(p.stem.split('-')[1]) for p in checkpoints]}")

    model, config = build_model(args.exp_dir, args.device)
    class_names = CLASS_NAMES_CAPITALIZED[config.dataset.value]

    ds = PFDataset(
        data_dir=str(args.data_dir),
        name=args.dataset,
        split=args.split,
        pad_to_multiple=None,
        num_samples=args.start_event + args.num_events,
    )
    print(f"Loaded dataset {args.dataset} split={args.split}, using events [{args.start_event}, {args.start_event + args.num_events})")

    events = []
    for i in range(args.start_event, args.start_event + args.num_events):
        item = ds.ds[i]
        event_rng = np.random.default_rng(args.seed * 100000 + i)
        prepared = prepare_event(item["X"], item["ytarget"], args.min_hits_per_particle, args.max_hits_per_event, event_rng)
        if prepared is None:
            print(f"  event {i}: skipping, fewer than 2 particles with >= {args.min_hits_per_particle} hits")
            continue
        prepared["event_idx"] = i
        events.append(prepared)
        print(f"  event {i}: {len(prepared['idx'])} hits kept, {prepared['n_particles']} particles")

    nn_baselines = [e["nn_random_baseline"] for e in events]
    nn_baseline = float(np.mean(nn_baselines))

    layer_labels = None
    per_layer_event = []
    probe_records = []

    for checkpoint_path in checkpoints:
        step = int(checkpoint_path.stem.split("-")[1])
        print(f"\nCheckpoint step {step} ({checkpoint_path.name})")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        load_checkpoint(checkpoint, model, None, strict=False)
        model.eval()

        pooled = defaultdict(lambda: {"emb": [], "y": [], "groups": []})

        for event in events:
            layers = get_layer_embeddings(model, event["X"], args.device)
            if layer_labels is None:
                layer_labels = build_layer_labels(len(layers), model.use_detector_backbone)
                print(f"Backbone stages: {layer_labels}")

            for layer_idx, emb_full in enumerate(layers):
                emb_k = emb_full[event["idx"]]
                metrics = compute_layer_metrics(emb_k, event["pn"], event["n_particles"], args.seed)
                per_layer_event.append({"step": step, "layer": layer_idx, "event": event["event_idx"], **metrics})

                known = event["cls"] != -1
                if known.any():
                    pooled[layer_idx]["emb"].append(emb_k[known])
                    pooled[layer_idx]["y"].append(event["cls"][known])
                    pooled[layer_idx]["groups"].append(event["event_idx"] * 100000 + event["pn"][known])

        for layer_idx, label in enumerate(layer_labels):
            emb = np.concatenate(pooled[layer_idx]["emb"])
            y = np.concatenate(pooled[layer_idx]["y"])
            groups = np.concatenate(pooled[layer_idx]["groups"])
            probe = run_probe(emb, y, groups, args.probe_cv_folds, args.seed, args.max_probe_samples, rng)
            if probe is not None:
                probe_records.append({"step": step, "layer": layer_idx, **probe})
                print(
                    f"  layer {layer_idx} ({label}): probe acc = {probe['probe_acc_mean']:.3f} "
                    f"+/- {probe['probe_acc_std']:.3f} (baseline {probe['probe_baseline']:.3f})"
                )

        del checkpoint
        if args.device == "cuda":
            torch.cuda.empty_cache()

    selected_steps = select_steps_for_layer_plot(sorted({r["step"] for r in per_layer_event}))

    plot_vs_checkpoint(per_layer_event, probe_records, layer_labels, nn_baseline, args.output_dir / "metrics_vs_checkpoint.png")
    plot_vs_layer(per_layer_event, probe_records, layer_labels, nn_baseline, selected_steps, args.output_dir / "metrics_vs_layer.png")
    print(f"\nWrote {args.output_dir / 'metrics_vs_checkpoint.png'}")
    print(f"Wrote {args.output_dir / 'metrics_vs_layer.png'}")

    with open(args.output_dir / "scan_results.json", "w") as f:
        json.dump(
            {
                "exp_dir": str(args.exp_dir),
                "dataset": args.dataset,
                "split": args.split,
                "layer_labels": layer_labels,
                "class_names": class_names,
                "min_hits_per_particle": args.min_hits_per_particle,
                "nn_random_baseline": nn_baseline,
                "per_layer_event": per_layer_event,
                "probe": probe_records,
            },
            f,
            indent=2,
        )
    print(f"Wrote {args.output_dir / 'scan_results.json'}")


if __name__ == "__main__":
    main()
