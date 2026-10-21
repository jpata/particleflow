#!/usr/bin/env python3
"""Analyze the last-layer hit embeddings of a trained hit-based MLPF model.

Loads a trained checkpoint from a hit-based MLPF experiment directory, runs
`MLPF.encode_backbone` on a few events to obtain the per-hit learned
representation of the final backbone layer, and checks whether hits that
truly originate from the same generator-level particle (ground truth
`particle_number` stored in `ytarget`) end up nearby in embedding space.

For each event this produces:
  - a 2D projection (UMAP if available, else PCA) of the hit embeddings,
    colored by true particle origin, by true particle class, and by
    detector/hit type, saved as a PNG.
  - quantitative clustering scores (silhouette score and nearest-neighbor
    same-origin rate) for the true particle-origin grouping, each compared
    against a label-shuffled baseline and (for the NN rate) a baseline
    expected purely from cluster-size statistics.

Example:
    uv run python3 scripts/analyze_hit_embeddings.py \\
        --exp-dir experiments/cld_hits_output_comparison/elementwise_seed12345_20260906_012757_908052 \\
        --num-events 3 --output-dir tmp/hit_embedding_analysis
"""

from __future__ import annotations

import argparse
import json
import pickle as pkl
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from mlpf.conf import CLASS_NAMES_CAPITALIZED
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFDataset
from mlpf.model.utils import load_checkpoint

try:
    import umap

    _HAVE_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA

    _HAVE_UMAP = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment dir with model_kwargs.pkl and checkpoints/")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename in <exp-dir>/checkpoints/ (default: latest step)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tfds_validation_cld_hits"), help="TFDS data root")
    parser.add_argument("--dataset", default="cld_edm_ttbar_hits/10:3.2.1", help="TFDS dataset name/config:version")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-events", type=int, default=3)
    parser.add_argument("--start-event", type=int, default=0)
    parser.add_argument("--min-hits-per-particle", type=int, default=5, help="Min true hits for a particle to enter the clustering stats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/hit_embedding_analysis"))
    return parser.parse_args()


def pick_checkpoint(exp_dir: Path, name: str | None) -> Path:
    ckpt_dir = exp_dir / "checkpoints"
    if name is not None:
        return ckpt_dir / name
    ckpts = sorted(ckpt_dir.glob("checkpoint-*.pth"), key=lambda p: int(p.stem.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return ckpts[-1]


def load_model(exp_dir: Path, checkpoint_path: Path, device: str):
    with open(exp_dir / "model_kwargs.pkl", "rb") as f:
        config = pkl.load(f)
    model = MLPF(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_checkpoint(checkpoint, model, None, strict=False)
    model.eval().to(device)
    return model, config


def get_hit_embeddings(model, X: np.ndarray, device: str) -> np.ndarray:
    """Run the model backbone and return the last-layer per-hit embedding, shape [N, D]."""
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = (X_t[..., 0] != 0).float()
    autocast_enabled = device == "cuda"
    with torch.no_grad(), torch.autocast(device_type="cuda" if autocast_enabled else "cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        embeddings = model.encode_backbone(X_t, mask_t)
    return embeddings[0].float().cpu().numpy()


def particle_class_per_hit(ytarget: np.ndarray) -> np.ndarray:
    """Broadcast the true particle class from each particle's representative hit to all its hits."""
    particle_number = ytarget[:, 13]
    cls_id = ytarget[:, 0]
    cls_by_particle = {int(pn): int(c) for pn, c in zip(particle_number, cls_id) if pn > 0 and c != 0}
    return np.array([cls_by_particle.get(int(pn), -1) for pn in particle_number])


def project_2d(embeddings: np.ndarray, seed: int) -> np.ndarray:
    if _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    else:
        reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(embeddings)


def compute_cluster_metrics(embeddings: np.ndarray, particle_number: np.ndarray, min_hits: int, rng: np.random.Generator) -> dict | None:
    valid = particle_number > 0
    pn, emb = particle_number[valid], embeddings[valid]

    counts = {p: int((pn == p).sum()) for p in np.unique(pn)}
    keep = np.isin(pn, [p for p, c in counts.items() if c >= min_hits])
    pn_k, emb_k = pn[keep], emb[keep]
    if len(np.unique(pn_k)) < 2:
        return None

    sil_true = float(silhouette_score(emb_k, pn_k))
    sil_shuffled = float(silhouette_score(emb_k, rng.permutation(pn_k)))

    nn = NearestNeighbors(n_neighbors=2).fit(emb_k)
    nearest = nn.kneighbors(emb_k, return_distance=False)[:, 1]
    same_origin_rate = float(np.mean(pn_k[nearest] == pn_k))

    sizes = np.array([counts[p] for p in pn_k])
    random_baseline_rate = float(np.mean((sizes - 1) / (len(pn_k) - 1)))

    return {
        "n_hits_used": int(len(pn_k)),
        "n_particles_used": int(len(np.unique(pn_k))),
        "silhouette_true_origin": sil_true,
        "silhouette_shuffled_origin": sil_shuffled,
        "nn_same_origin_rate": same_origin_rate,
        "nn_same_origin_rate_random_baseline": random_baseline_rate,
    }


def plot_event(embeddings: np.ndarray, X: np.ndarray, ytarget: np.ndarray, event_idx: int, seed: int, out_dir: Path):
    particle_number = ytarget[:, 13]
    elemtype = X[:, 0]
    particle_class = particle_class_per_hit(ytarget)

    proj = project_2d(embeddings, seed)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # panel 1: colored by true particle origin (categorical, one color per particle)
    ax = axes[0]
    valid = particle_number > 0
    ax.scatter(proj[~valid, 0], proj[~valid, 1], c="lightgray", s=6, label="no true origin", alpha=0.5)
    pn_valid = particle_number[valid]
    unique_pn = np.unique(pn_valid)
    color_idx = {p: i % 20 for i, p in enumerate(unique_pn)}
    colors = np.array([color_idx[p] for p in pn_valid])
    ax.scatter(proj[valid, 0], proj[valid, 1], c=colors, cmap="tab20", s=8)
    ax.set_title(f"colored by true particle origin\n({len(unique_pn)} particles)")

    # panel 2: colored by true particle class
    ax = axes[1]
    class_names = ["none"] + CLASS_NAMES_CAPITALIZED["cld_hits"][1:]
    for cid in sorted(np.unique(particle_class)):
        m = particle_class == cid
        label = "no true origin" if cid == -1 else class_names[cid] if cid < len(class_names) else str(cid)
        ax.scatter(proj[m, 0], proj[m, 1], s=8, label=label, alpha=0.7)
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title("colored by true particle class")

    # panel 3: colored by hit/detector type
    ax = axes[2]
    for et, label in [(1, "tracker hit"), (2, "calorimeter hit")]:
        m = elemtype == et
        ax.scatter(proj[m, 0], proj[m, 1], s=8, label=label, alpha=0.7)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title("colored by detector hit type")

    for ax in axes:
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    method = "UMAP" if _HAVE_UMAP else "PCA"
    fig.suptitle(f"Event {event_idx}: last-layer hit embeddings ({method} projection, {embeddings.shape[0]} hits)")
    fig.tight_layout()
    out_path = out_dir / f"event_{event_idx}_embeddings.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    checkpoint_path = pick_checkpoint(args.exp_dir, args.checkpoint)
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(args.exp_dir, checkpoint_path, args.device)
    print(f"Model: dataset={config.dataset}, learned_representation_mode={config.model.learned_representation_mode}")

    ds = PFDataset(
        data_dir=str(args.data_dir),
        name=args.dataset,
        split=args.split,
        pad_to_multiple=None,
        num_samples=args.start_event + args.num_events,
    )
    print(f"Loaded dataset {args.dataset} split={args.split}, using events [{args.start_event}, {args.start_event + args.num_events})")

    all_metrics = []
    for i in range(args.start_event, args.start_event + args.num_events):
        item = ds.ds[i]
        X, ytarget = item["X"], item["ytarget"]
        n_hits = X.shape[0]
        n_particles = len(np.unique(ytarget[:, 13][ytarget[:, 13] > 0]))
        print(f"\nEvent {i}: {n_hits} hits, {n_particles} true particles with associated hits")

        embeddings = get_hit_embeddings(model, X, args.device)

        metrics = compute_cluster_metrics(embeddings, ytarget[:, 13], args.min_hits_per_particle, rng)
        if metrics is None:
            print(f"  Skipping metrics: fewer than 2 particles have >= {args.min_hits_per_particle} hits")
        else:
            metrics["event_idx"] = i
            all_metrics.append(metrics)
            print(f"  particles with >= {args.min_hits_per_particle} hits: {metrics['n_particles_used']} ({metrics['n_hits_used']} hits)")
            print(f"  silhouette score, true origin labels:     {metrics['silhouette_true_origin']:+.4f}")
            print(f"  silhouette score, shuffled origin labels: {metrics['silhouette_shuffled_origin']:+.4f}")
            print(f"  nearest-neighbor same-origin rate:        {metrics['nn_same_origin_rate']:.4f}")
            print(f"  same-origin rate random baseline:         {metrics['nn_same_origin_rate_random_baseline']:.4f}")

        out_path = plot_event(embeddings, X, ytarget, i, args.seed, args.output_dir)
        print(f"  wrote {out_path}")

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset,
                "split": args.split,
                "min_hits_per_particle": args.min_hits_per_particle,
                "per_event": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nWrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
