"""Frozen, event-local hit-pair probes; simulation labels are diagnostics only."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mlpf.conf import Y_FEATURES


class HitPairProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    neighbors: int = Field(default=16, ge=1)
    train_pairs_per_event: int = Field(default=2048, ge=1)
    valid_pairs_per_event: int = Field(default=8192, ge=1)
    epochs: int = Field(default=10, ge=1)
    alpha: float = Field(default=0.0001, gt=0)


def pair_events(cache, kind, config, seed, training):
    from mlpf.model.instance_probe import candidate_edges

    rng = np.random.default_rng(seed)
    limit = config.train_pairs_per_event if training else config.valid_pairs_per_event
    events = []
    for index, record in enumerate(cache):
        x = np.asarray(record["X"])
        labels = np.asarray(record["ytarget"])[:, Y_FEATURES.index("particle_number")]
        # This diagnostic is explicitly conditional on oracle foreground.
        keep = (x[:, 0] == kind) & (labels > 0)
        if keep.sum() < 2:
            continue
        raw, labels = x[keep], labels[keep]
        edges = candidate_edges(raw, config.neighbors)
        edges = edges[rng.choice(len(edges), min(limit, len(edges)), replace=False)]
        embedding = record["embedding"].float().numpy()[keep]
        events.append(
            {
                "event_index": index,
                "raw": np.abs(raw[edges[:, 0]] - raw[edges[:, 1]]),
                "embedding": np.abs(embedding[edges[:, 0]] - embedding[edges[:, 1]]),
                "target": (labels[edges[:, 0]] == labels[edges[:, 1]]).astype(np.int64),
            }
        )
    return events


def fit_hit_pair_probes(train_cache, valid_cache, config, seed, artifact_path=None):
    """Fit four linear logistic probes on identical geometry-selected pairs.

    Standardization uses training data only; each event has equal weight, with
    its present classes balanced. Validation retains natural pair prevalence.
    No thresholds are selected and validation labels never affect fitting.
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    result = {"foreground_only": True, "config": config.model_dump(mode="json"), "detectors": {}}
    artifacts = {}
    for detector, kind in (("tracker", 1), ("calorimeter", 2)):
        train = pair_events(train_cache, kind, config, seed, True)
        valid = pair_events(valid_cache, kind, config, seed + 1, False)
        results = result["detectors"][detector] = {}
        if not train or not valid or len(np.unique(np.concatenate([e["target"] for e in train]))) < 2:
            results["status"] = "insufficient labeled pairs or training classes"
            continue
        for representation in ("raw", "embedding"):
            scaler = StandardScaler()
            weights = []
            for event in train:
                y = event["target"]
                counts = np.bincount(y, minlength=2)
                w = 1.0 / counts[y]
                w /= w.sum()
                weights.append(w)
                scaler.partial_fit(event[representation], sample_weight=w)
            # Scale equal-event weights to mean one over the complete fit set.
            normalization = sum(len(e["target"]) for e in train) / len(train)
            features = [scaler.transform(e[representation]) for e in train]
            model = SGDClassifier(loss="log_loss", alpha=config.alpha, learning_rate="constant", eta0=0.01, average=True, random_state=seed)
            rng = np.random.default_rng(seed)
            for _ in range(config.epochs):
                for i in rng.permutation(len(train)):
                    model.partial_fit(features[i], train[i]["target"], classes=[0, 1], sample_weight=weights[i] * normalization)
            rows = []
            for event in valid:
                y = event["target"]
                scores = model.decision_function(scaler.transform(event[representation]))
                row = {"event_index": event["event_index"], "pairs": len(y), "positive_fraction": float(y.mean())}
                if len(np.unique(y)) == 2:
                    row.update(average_precision=float(average_precision_score(y, scores)), roc_auc=float(roc_auc_score(y, scores)))
                rows.append(row)
            means = {
                key: float(np.mean([r[key] for r in rows if key in r]))
                for key in ("positive_fraction", "average_precision", "roc_auc")
                if any(key in r for r in rows)
            }
            results[representation] = {
                "mean": means,
                "events": rows,
                "train_events": len(train),
                "valid_events_with_both_classes": sum("roc_auc" in r for r in rows),
                "feature_dimension": features[0].shape[1],
            }
            prefix = f"{detector}_{representation}"
            for key, value in (("mean", scaler.mean_), ("scale", scaler.scale_), ("coef", model.coef_), ("intercept", model.intercept_)):
                artifacts[f"{prefix}_{key}"] = value
    if artifact_path is not None:
        temporary = artifact_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **artifacts)
        temporary.replace(artifact_path)
    return result
