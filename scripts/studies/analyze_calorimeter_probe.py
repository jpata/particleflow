"""Audit calorimeter edge prevalence, geometric baselines and difficult edges."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from scripts.studies.probe_hit_instances import pair_features, read


def graph_edges(coords, k=16):
    n = len(coords)
    nn = (
        NearestNeighbors(n_neighbors=min(k + 1, n))
        .fit(coords)
        .kneighbors(coords, return_distance=False)
    )
    edges = np.sort(np.stack([np.repeat(np.arange(n), nn.shape[1]), nn.ravel()], 1), 1)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def ranking(y, score):
    if len(np.unique(y)) < 2:
        return {}
    weights = np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    return dict(
        ap=float(average_precision_score(y, score)),
        auc=float(roc_auc_score(y, score)),
        balanced_ap=float(average_precision_score(y, score, sample_weight=weights)),
        different_particle_ap=float(average_precision_score(~y, -score)),
    )


def analyze(root):
    fits = {
        r: joblib.load(root / f"calorimeter_{r}.joblib") for r in ("raw", "embedding")
    }
    rows = []
    for path in sorted((root / "cache/calorimeter/test").glob("*.npz")):
        d = read(path)
        x, labels, edges = d["raw"], d["labels"], d["edges"]
        a, b = edges.T
        y = labels[a] == labels[b]
        row = dict(event=path.stem, edges=len(y), positive_fraction=float(y.mean()))
        counts = np.unique(labels, return_counts=True)[1]
        row["all_pairs_positive_fraction"] = float(
            np.sum(counts * (counts - 1)) / (len(labels) * (len(labels) - 1))
        )
        spatial = graph_edges(x[:, 6:9])
        direction = graph_edges(x[:, [2, 3, 4]])
        spatial_keys = spatial[:, 0] * len(labels) + spatial[:, 1]
        direction_keys = direction[:, 0] * len(labels) + direction[:, 1]
        keys = a * len(labels) + b
        sm = np.isin(keys, spatial_keys)
        dm = np.isin(keys, direction_keys)
        for name, mask in [
            ("xyz", sm),
            ("direction", dm),
            ("both", sm & dm),
            ("xyz_only", sm & ~dm),
            ("direction_only", dm & ~sm),
        ]:
            row[f"{name}/edge_fraction"] = float(mask.mean())
            row[f"{name}/positive_fraction"] = (
                float(y[mask].mean()) if mask.any() else 0
            )
        boundary = np.zeros(len(labels), dtype=bool)
        boundary[edges[~y].ravel()] = True
        hard = boundary[a] & boundary[b]
        row["boundary/hit_fraction"] = float(boundary.mean())
        row["boundary/edge_fraction"] = float(hard.mean())
        row["boundary/positive_fraction"] = float(y[hard].mean())
        scores = {
            "xyz_distance": -np.linalg.norm(x[a, 6:9] - x[b, 6:9], axis=1),
            "direction_distance": -np.linalg.norm(
                x[a][:, [2, 3, 4]] - x[b][:, [2, 3, 4]], axis=1
            ),
        }
        for rep, fit in fits.items():
            scores[rep] = fit["model"].predict_proba(
                fit["scaler"].transform(pair_features(d, rep, edges))
            )[:, 1]
            keep = scores[rep] >= fit["threshold"]
            row[f"{rep}/selected_precision"] = float(y[keep].mean())
            row[f"{rep}/selected_recall"] = float(keep[y].mean())
            row[f"{rep}/false_positive_rate"] = float(keep[~y].mean())
            row[f"{rep}/false_edges"] = int((keep & ~y).sum())
            row[f"{rep}/joined_truth_pairs"] = len(
                np.unique(
                    np.sort(
                        np.stack([labels[a[keep & ~y]], labels[b[keep & ~y]]], 1), 1
                    ),
                    axis=0,
                )
            )
        for name, score in scores.items():
            for scope, mask in [("all", np.ones(len(y), bool)), ("boundary", hard)]:
                for key, val in ranking(y[mask], score[mask]).items():
                    row[f"{name}/{scope}/{key}"] = val
        rows.append(row)
        print("analyzed event", path.stem, flush=True)
    keys = set(rows[0]) - {"event"}
    return dict(
        definition="Equal-event means over existing probe-test events; boundary means both endpoints have at least one different-particle candidate neighbor. Balanced AP reweights each class to 50%.",
        mean={k: float(np.mean([r[k] for r in rows])) for k in sorted(keys)},
        events=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        type=Path,
        default=Path("experiments/hit_instance_probe_50k_initial"),
    )
    args = p.parse_args()
    result = analyze(args.experiment)
    (args.experiment / "calorimeter_edge_audit.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result["mean"], indent=2))
