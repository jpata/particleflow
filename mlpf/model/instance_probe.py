"""Shared instance graph and held-out ARI diagnostics for frozen hit features."""

import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


def candidate_edges(x, k):
    """Common label-free graph: union of spatial and direction kNN edges."""
    n = len(x)
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    blocks = []
    for coords in (x[:, 6:9], x[:, [2, 3, 4]]):
        indices = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords).kneighbors(coords, return_distance=False)
        blocks.append(np.stack((np.repeat(np.arange(n), indices.shape[1]), indices.ravel()), axis=1))
    edges = np.sort(np.concatenate(blocks), axis=1)
    return np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)


def cluster(n, edges, keep):
    selected = edges[keep]
    graph = coo_matrix((np.ones(len(selected)), (selected[:, 0], selected[:, 1])), shape=(n, n)).tocsr()
    return connected_components(graph, directed=False)[1]


THRESHOLDS = (0.3, 0.5, 0.7, 0.85, 0.95, 0.98, 0.995)


def empty_ari_metrics(detector):
    prefix = f"metrics/linear_probe/{detector}/"
    names = [
        "ari_fit_event_samples",
        "ari_calibration_event_samples",
        "ari_eval_event_samples",
        "ari_fit_success_fraction",
        "raw/ari",
        "embedding/ari",
        "embedding_minus_raw/ari",
        "oracle_graph/ari",
        "raw/ari_threshold",
        "embedding/ari_threshold",
    ]
    return {prefix + name: [0.0, 0.0] for name in names}


def pair_features(event, representation, edges):
    z = event[representation]
    return np.abs(z[edges[:, 0]] - z[edges[:, 1]])


def fit_affinity(events, rep):
    """Same deterministic edge samples for both representations, capped per event."""
    rng = np.random.default_rng(12345)
    samples = []
    scaler = StandardScaler()
    for event, edges in events:
        if not len(edges):
            continue
        edges = edges[rng.choice(len(edges), min(2048, len(edges)), replace=False)]
        features = pair_features(event, rep, edges)
        y = (event["labels"][edges[:, 0]] == event["labels"][edges[:, 1]]).astype(int)
        counts = np.bincount(y, minlength=2)
        weights = 1.0 / np.maximum(counts[y], 1)
        weights /= weights.sum()  # equal event weight, balanced present classes
        scaler.partial_fit(features, sample_weight=weights)
        samples.append((features, y, weights))
    if not samples:
        return None
    x = np.concatenate([scaler.transform(x) for x, _, _ in samples])
    y = np.concatenate([y for _, y, _ in samples])
    if len(np.unique(y)) < 2:
        return None
    weights = np.concatenate([w for _, _, w in samples])
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(C=0.1, max_iter=200, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y, sample_weight=weights)
    if model.n_iter_[0] >= model.max_iter:
        return None
    return scaler, model


def edge_probabilities(event, edges, rep, fit):
    scaler, model = fit
    return np.concatenate(
        [model.predict_proba(scaler.transform(pair_features(event, rep, edges[start : start + 8192])))[:, 1] for start in range(0, len(edges), 8192)]
    )


def select_threshold(calibration, rep, fit):
    scores = [(event, edges, edge_probabilities(event, edges, rep, fit)) for event, edges in calibration]
    values = [
        np.mean(
            [
                adjusted_rand_score(
                    event["labels"],
                    cluster(len(event["labels"]), edges, score >= threshold),
                )
                for event, edges, score in scores
            ]
        )
        for threshold in THRESHOLDS
    ]
    return THRESHOLDS[int(np.argmax(values))]


def compute_instance_ari(events, detector):
    """Per four collected events: two fit, one calibration, one final evaluation."""
    result = empty_ari_metrics(detector)
    prefix = f"metrics/linear_probe/{detector}/"

    def add(name, total, count=1):
        result[prefix + name][0] += float(total)
        result[prefix + name][1] += float(count)

    prepared = [(i, event, candidate_edges(event["raw"], 16)) for i, event in enumerate(events) if event is not None and len(event["labels"]) >= 2]
    training = [(e, edges) for i, e, edges in prepared if i % 4 in (0, 1)]
    calibration = [(e, edges) for i, e, edges in prepared if i % 4 == 2]
    testing = [(e, edges) for i, e, edges in prepared if i % 4 == 3]
    add("ari_fit_event_samples", len(training))
    add("ari_calibration_event_samples", len(calibration))
    add("ari_eval_event_samples", len(testing))
    fits = {}
    if len(training) >= 4 and len(calibration) >= 2 and len(testing) >= 2:
        fits = {rep: fit_affinity(training, rep) for rep in ("raw", "embedding")}
    success = len(fits) == 2 and all(f is not None for f in fits.values())
    add("ari_fit_success_fraction", success)
    if success:
        thresholds = {rep: select_threshold(calibration, rep, fits[rep]) for rep in fits}
        for rep, threshold in thresholds.items():
            add(rep + "/ari_threshold", threshold)
    for event, edges in testing:
        y = event["labels"]
        oracle = cluster(len(y), edges, y[edges[:, 0]] == y[edges[:, 1]])
        add("oracle_graph/ari", adjusted_rand_score(y, oracle))
        if not success:
            continue
        values = {}
        for rep, fit in fits.items():
            scores = edge_probabilities(event, edges, rep, fit)
            values[rep] = adjusted_rand_score(y, cluster(len(y), edges, scores >= thresholds[rep]))
            add(rep + "/ari", values[rep])
        add("embedding_minus_raw/ari", values["embedding"] - values["raw"])
    return result
