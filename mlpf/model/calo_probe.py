"""Shared cross-layer calorimeter probe geometry and linear ranking metrics."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def particle_weights(labels):
    """Equal total weight per represented particle, then per anchor."""
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv] / len(counts) if len(labels) else np.empty(0)


def build_candidates(x, labels, neighbors=16, layer_gap=1, tolerance=1.25, seed=12345):
    """Forward layers within (system, side); geometry-only union, truth-only triplet sampling."""
    rng = np.random.default_rng(seed)
    surfaces = x[:, 12:15].astype(np.int64)
    groups = {tuple(key): np.flatnonzero(np.all(surfaces == key, axis=1)) for key in np.unique(surfaces, axis=0)}
    candidates, triplets = {}, []
    support = dict(
        hits=len(x),
        target_layer_exists=0,
        same_particle_in_target_layer=0,
        positive_in_candidates=0,
        competing_candidates=0,
        distance_matched_triplets=0,
    )
    for (system, side, layer), anchors in groups.items():
        target = groups.get((system, side, layer + layer_gap))
        if target is None:
            continue
        support["target_layer_exists"] += len(anchors)
        support["same_particle_in_target_layer"] += int(np.isin(labels[anchors], labels[target]).sum())
        lists = []
        for columns in ([6, 7, 8], [2, 3, 4]):
            coords = x[:, columns]
            nn = NearestNeighbors(n_neighbors=min(neighbors, len(target))).fit(coords[target])
            lists.append(target[nn.kneighbors(coords[anchors], return_distance=False)])
        for ai, a in enumerate(anchors):
            c = np.unique(np.concatenate([v[ai] for v in lists]))
            candidates[int(a)] = c
            pos = labels[c] == labels[a]
            support["positive_in_candidates"] += int(pos.any())
            if not pos.any() or pos.all():
                continue
            support["competing_candidates"] += 1
            dp = np.linalg.norm(x[c, 6:9] - x[a, 6:9], axis=1)
            da = np.linalg.norm(x[c][:, [2, 3, 4]] - x[a, [2, 3, 4]], axis=1)
            p, n = np.flatnonzero(pos), np.flatnonzero(~pos)
            # Match BOTH distance measures, symmetrically: neither member can be
            # more than tolerance times as far as the other. No preference for
            # a closer positive. Sample uniformly from all admissible pairs.
            ratio_xyz = np.maximum(dp[p, None], dp[n][None]) / np.maximum(np.minimum(dp[p, None], dp[n][None]), 1e-6)
            ratio_dir = np.maximum(da[p, None], da[n][None]) / np.maximum(np.minimum(da[p, None], da[n][None]), 1e-8)
            choices = np.argwhere((ratio_xyz <= tolerance) & (ratio_dir <= tolerance))
            if len(choices):
                pi, ni = choices[rng.integers(len(choices))]
                triplets.append((a, c[p[pi]], c[n[ni]]))
    support["distance_matched_triplets"] = len(triplets)
    return candidates, np.asarray(triplets, dtype=np.int64).reshape(-1, 3), support


def differences(z, triplets):
    a, p, n = triplets.T
    return np.abs(z[a] - z[p]), np.abs(z[a] - z[n])


def accuracy(margin):
    return (margin > 0).astype(float) + 0.5 * (margin == 0)


def score_edges(event, rep, fit, a, b):
    if len(a) == 0:
        return np.empty(0)
    if rep == "xyz_distance":
        return -np.linalg.norm(event["raw"][a, 6:9] - event["raw"][b, 6:9], axis=1)
    if rep == "direction_distance":
        return -np.linalg.norm(event["raw"][a][:, [2, 3, 4]] - event["raw"][b][:, [2, 3, 4]], axis=1)
    z = event[rep]
    parts = []
    for start in range(0, len(a), 8192):
        feat = np.abs(z[a[start : start + 8192]] - z[b[start : start + 8192]])
        parts.append(fit["model"].decision_function(fit["scaler"].transform(feat)))
    return np.concatenate(parts)


def ranking_metric(event, triplets, rep, fit):
    a, p, n = triplets.T
    margin = score_edges(event, rep, fit, a, p) - score_edges(event, rep, fit, a, n)
    return float(particle_weights(event["labels"][a]) @ accuracy(margin))
