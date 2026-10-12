#!/usr/bin/env python3
"""Frozen linear pair-affinity probes for event-local hit instances."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from mlpf.model.instance_probe import candidate_edges, cluster


def instance_metrics(truth, pred):
    """Permutation-invariant metrics, including singleton instances."""
    _, t = np.unique(truth, return_inverse=True)
    _, p = np.unique(pred, return_inverse=True)
    counts = coo_matrix((np.ones(len(t)), (t, p))).tocsr()
    nt = np.asarray(counts.sum(1)).ravel()
    npred = np.asarray(counts.sum(0)).ravel()
    c = counts.tocoo()
    precision = float(np.sum(c.data**2 / npred[c.col]) / len(t))
    recall = float(np.sum(c.data**2 / nt[c.row]) / len(t))
    iou = c.data / (nt[c.row] + npred[c.col] - c.data)
    # Strict IoU > 0.5 implies one-to-one matches for disjoint partitions.
    matches = int((iou > 0.5).sum())
    return dict(
        ari=adjusted_rand_score(t, p),
        bcubed_precision=precision,
        bcubed_recall=recall,
        bcubed_f1=2 * precision * recall / max(precision + recall, 1e-12),
        instance_f1_iou0p5=2 * matches / (len(nt) + len(npred)),
        truth_instances=len(nt),
        predicted_instances=len(npred),
        hits=len(t),
    )


def pair_features(event, representation, edges):
    z = event[representation]
    return np.abs(z[edges[:, 0]] - z[edges[:, 1]])


def prepare(args):
    import torch
    import yaml
    from mlpf.conf import MLPFConfig, Y_FEATURES
    from mlpf.model.mlpf import MLPF
    from mlpf.model.losses import make_task_loss_weighter
    from mlpf.model.PFDataset import PFDataset

    torch.manual_seed(args.seed)
    config = MLPFConfig.model_validate(
        yaml.safe_load((args.run / "train-config.yaml").read_text())
    )
    config.model.attention.use_flash_attn_varlen = False
    model = MLPF(config)
    model.task_loss_weighter = make_task_loss_weighter(
        config.task_loss_weights.model_dump()
    )
    checkpoint = args.run / "checkpoints" / f"checkpoint-{args.step}.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    del state
    model.requires_grad_(False).eval().to(args.device)
    dataset = PFDataset(
        str(args.data_dir),
        args.dataset,
        "test",
        pad_to_multiple=config.pad_to_multiple_elements,
        feature_dim=config.input_dim,
        max_open_readers=4,
    ).ds
    total = args.train_events + args.valid_events + args.test_events
    if args.offset + total > len(dataset):
        raise ValueError("Requested event range exceeds dataset")
    # Disjoint probe events from TFDS test: excluded from backbone gradient updates,
    # but possibly used for backbone validation/model selection.
    indices = np.random.default_rng(args.seed).permutation(
        np.arange(args.offset, args.offset + total)
    )
    manifest = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        checkpoint=str(checkpoint.resolve()),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=hashlib.file_digest(
            checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        event_indices=indices.tolist(),
        backbone_split="test",
        foreground_only=True,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for j, idx in enumerate(indices):
        split = (
            "train"
            if j < args.train_events
            else "valid"
            if j < args.train_events + args.valid_events
            else "test"
        )
        record = dataset[int(idx)]
        x = np.asarray(record["X"], dtype=np.float32)
        labels = record["ytarget"][:, Y_FEATURES.index("particle_number")].astype(
            np.int64
        )
        tensor = torch.as_tensor(x, device=args.device)[None]
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=args.device.split(":")[0],
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ),
        ):
            embedding = (
                model.encode_backbone(tensor, tensor[..., 0] != 0)[0]
                .float()
                .cpu()
                .numpy()
            )
        if not np.isfinite(embedding).all() or not np.isfinite(x).all():
            raise ValueError(f"Nonfinite features in event {idx}")
        for detector, kind in [("tracker", 1), ("calorimeter", 2)]:
            mask = (x[:, 0] == kind) & (labels > 0)
            if mask.sum() < 2:
                continue
            path = args.output / "cache" / detector / split
            path.mkdir(parents=True, exist_ok=True)
            raw = x[mask]
            np.savez_compressed(
                path / f"{idx}.npz",
                raw=raw,
                embedding=embedding[mask],
                labels=labels[mask],
                edges=candidate_edges(raw, args.neighbors),
            )
        print(f"extracted {j + 1}/{total}: event {idx} ({split})", flush=True)


def read(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def evaluate(paths, representation, scaler, model, threshold):
    rows = []
    for path in paths:
        event = read(path)
        edges, labels = event["edges"], event["labels"]
        target = labels[edges[:, 0]] == labels[edges[:, 1]]
        scores = model.predict_proba(
            scaler.transform(pair_features(event, representation, edges))
        )[:, 1]
        row = instance_metrics(labels, cluster(len(labels), edges, scores >= threshold))
        row["oracle_graph_ari"] = instance_metrics(
            labels, cluster(len(labels), edges, target)
        )["ari"]
        row["edge_positive_fraction"] = float(target.mean())
        if len(np.unique(target)) == 2:
            row["edge_ap"] = average_precision_score(target, scores)
            row["edge_auc"] = roc_auc_score(target, scores)
        row["event"] = path.stem
        rows.append(row)
    keys = sorted(set().union(*(r.keys() for r in rows)) - {"event"})
    return dict(
        mean={k: float(np.mean([r[k] for r in rows if k in r])) for k in keys},
        events=rows,
    )


def train(args):
    results = {}
    for detector in ("tracker", "calorimeter"):
        paths = {
            s: sorted((args.output / "cache" / detector / s).glob("*.npz"))
            for s in ("train", "valid", "test")
        }
        if any(not p for p in paths.values()):
            raise ValueError(f"Missing events for {detector}")
        for representation in ("raw", "embedding"):
            # Identical sampled edges and event order for the two representations.
            rng = np.random.default_rng(args.seed)
            scaler = StandardScaler()
            samples = []
            for path in paths["train"]:
                event = read(path)
                edges = event["edges"]
                edges = edges[
                    rng.choice(
                        len(edges), min(args.pairs_per_event, len(edges)), replace=False
                    )
                ]
                samples.append((path, edges))
                scaler.partial_fit(pair_features(event, representation, edges))
            model = SGDClassifier(
                loss="log_loss",
                alpha=args.alpha,
                learning_rate="adaptive",
                eta0=0.01,
                random_state=args.seed,
                average=True,
            )
            training_pairs = []
            for path, edges in samples:
                event = read(path)
                y = (
                    event["labels"][edges[:, 0]] == event["labels"][edges[:, 1]]
                ).astype(int)
                z = scaler.transform(pair_features(event, representation, edges))
                freq = np.bincount(y, minlength=2)
                weights = len(y) / (2 * np.maximum(freq[y], 1))
                training_pairs.append((z, y, weights))
            for epoch in range(args.epochs):
                for index in rng.permutation(len(samples)):
                    z, y, weights = training_pairs[index]
                    model.partial_fit(z, y, classes=[0, 1], sample_weight=weights)
            del training_pairs
            # Tune segmentation, not edge accuracy, using validation events only.
            sweep = {}
            for threshold in (0.3, 0.5, 0.7, 0.85, 0.95, 0.98, 0.995):
                sweep[threshold] = evaluate(
                    paths["valid"], representation, scaler, model, threshold
                )["mean"]["ari"]
            threshold = max(sweep, key=sweep.get)
            key = f"{detector}_{representation}"
            result = dict(
                threshold=threshold,
                validation_ari_sweep=sweep,
                test=evaluate(paths["test"], representation, scaler, model, threshold),
                feature_dimension=len(model.coef_[0]),
                train_events=len(paths["train"]),
            )
            results[key] = result
            joblib.dump(
                dict(scaler=scaler, model=model, threshold=threshold),
                args.output / f"{key}.joblib",
            )
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(key, json.dumps(result["test"]["mean"]), flush=True)
    return results


def plot_results(results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("edge_ap", "ari"),
        ("Same-particle edge average precision", "Instance adjusted Rand index"),
    ):
        for j, representation in enumerate(("raw", "embedding")):
            values = [
                results[f"{detector}_{representation}"]["test"]["mean"][metric]
                for detector in ("tracker", "calorimeter")
            ]
            bars = ax.bar(
                np.arange(2) + (j - 0.5) * 0.34,
                values,
                width=0.34,
                label=representation,
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=9)
        if metric == "ari":
            for j, detector in enumerate(("tracker", "calorimeter")):
                oracle = results[f"{detector}_raw"]["test"]["mean"]["oracle_graph_ari"]
                ax.plot(
                    [j - 0.34, j + 0.34],
                    [oracle, oracle],
                    "k--",
                    label="Oracle candidate graph" if j == 0 else None,
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Tracker", "Calorimeter"],
            ylim=(0, 1.08),
            title=title,
        )
        ax.legend(fontsize=8)
    fig.suptitle("Frozen grouped HEPTv2 at 50k: foreground hit-instance probes")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        type=Path,
        default=Path(
            "experiments/cld_pf_heptv2_hit_grouping_50k/hit_heptv2_grouped_seed12345_20261009_164038_591219"
        ),
    )
    p.add_argument("--step", type=int, default=50000)
    p.add_argument(
        "--data-dir", type=Path, default=Path("/mnt/work/mlpf/tensorflow_datasets/cld")
    )
    p.add_argument("--dataset", default="cld_edm_ttbar_hits/1:3.2.1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-events", type=int, default=64)
    p.add_argument("--valid-events", type=int, default=16)
    p.add_argument("--test-events", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--neighbors", type=int, default=16)
    p.add_argument("--pairs-per-event", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    if args.output.exists():
        p.error(
            "Output already exists; use a fresh directory to prevent mixing event caches"
        )
    args.output.mkdir(parents=True)
    if (
        min(
            args.train_events,
            args.valid_events,
            args.test_events,
            args.neighbors,
            args.pairs_per_event,
            args.epochs,
        )
        <= 0
        or args.offset < 0
    ):
        p.error(
            "Event counts, neighbors, pair count and epochs must be positive; offset must be nonnegative"
        )
    prepare(args)
    plot_results(train(args), args.output)


if __name__ == "__main__":
    main()
