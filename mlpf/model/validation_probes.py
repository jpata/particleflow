"""Bounded rank-local refits of frozen calorimeter probes during validation."""

import time
import hashlib
import warnings
from contextlib import contextmanager

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mlpf.conf import Y_FEATURES
from mlpf.model.instance_probe import compute_instance_ari, empty_ari_metrics
from mlpf.model.calo_probe import (
    build_candidates,
    differences,
    particle_weights,
    ranking_metric,
    score_edges,
)

PREFIX = "metrics/linear_probe/calorimeter/"
METHODS = ("xyz_distance", "direction_distance", "raw", "embedding")
SUPPORT = (
    "hits",
    "target_layer_exists",
    "same_particle_in_target_layer",
    "positive_in_candidates",
    "competing_candidates",
    "distance_matched_triplets",
)


def empty_metrics():
    keys = [
        "fit_event_samples",
        "eval_event_samples",
        "fit_triplet_samples",
        "eval_triplet_samples",
        "collected_event_samples",
        "fit_success_fraction",
        "seconds_per_rank",
    ]
    keys += ["coverage/" + k for k in SUPPORT[1:]]
    keys += [
        f"{m}/{k}"
        for m in METHODS
        for k in ("triplet_accuracy", "top1_competing", "top1_all_candidate_anchors")
    ]
    keys += ["embedding_minus_raw/triplet_accuracy"]
    metrics = {PREFIX + k: [0.0, 0.0] for k in keys}
    metrics.update(empty_ari_metrics("calorimeter"))
    metrics.update(empty_ari_metrics("tracker"))
    return metrics


class ValidationCaloProbe:
    def __init__(self, max_events=16):
        self.max_events = max_events
        self.events = []
        self.seen_events = set()
        self.seconds = 0.0

    @contextmanager
    def capture(self, model, batch):
        """Capture decoder input on the existing forward, then always remove hook."""
        holder = []
        hook = model.set_decoder.register_forward_pre_hook(
            lambda module, inputs: holder.append(inputs[0].detach())
        )
        try:
            yield
        finally:
            hook.remove()
        if holder:
            self.add_batch(batch, holder[0])

    def add_batch(self, batch, embedding):
        start = time.perf_counter()
        pn = Y_FEATURES.index("particle_number")
        for i in range(len(batch.X)):
            if len(self.events) >= self.max_events:
                break
            mask = (
                batch.mask[i] & (batch.X[i, :, 0] == 2) & (batch.ytarget[i, :, pn] > 0)
            )
            if not mask.any():
                continue
            event = dict(
                raw=batch.X[i, mask].detach().float().cpu().numpy(),
                embedding=embedding[i, mask].detach().float().cpu().numpy(),
                labels=batch.ytarget[i, mask, pn].detach().long().cpu().numpy(),
            )
            if (
                not np.isfinite(event["raw"]).all()
                or not np.isfinite(event["embedding"]).all()
            ):
                continue
            # Guard against repeated validation records crossing the fit/eval split.
            digest = hashlib.blake2b(digest_size=16)
            digest.update(event["raw"].tobytes())
            digest.update(event["labels"].tobytes())
            identity = digest.digest()
            if identity in self.seen_events:
                continue
            self.seen_events.add(identity)
            tracker_mask = (
                batch.mask[i] & (batch.X[i, :, 0] == 1) & (batch.ytarget[i, :, pn] > 0)
            )
            event["tracker"] = dict(
                raw=batch.X[i, tracker_mask].detach().float().cpu().numpy(),
                embedding=embedding[i, tracker_mask].detach().float().cpu().numpy(),
                labels=batch.ytarget[i, tracker_mask, pn].detach().long().cpu().numpy(),
            )
            self.events.append(event)
        self.seconds += time.perf_counter() - start

    def finalize(self):
        metrics = empty_metrics()
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            self._compute(metrics)
            metrics.update(compute_instance_ari(self.events, "calorimeter"))
            metrics.update(
                compute_instance_ari([e.get("tracker") for e in self.events], "tracker")
            )
        metrics[PREFIX + "seconds_per_rank"] = [
            self.seconds + time.perf_counter() - start,
            1.0,
        ]
        return metrics

    def _compute(self, metrics):
        def add(name, value, count=1.0):
            slot = metrics[PREFIX + name]
            slot[0] += float(value)
            slot[1] += float(count)

        add("collected_event_samples", len(self.events))
        prepared = [
            (e, *build_candidates(e["raw"], e["labels"], seed=12345 + i))
            for i, e in enumerate(self.events)
        ]
        # Whole events: every fourth event is probe evaluation. Never fit scalers
        # or classifier weights on these events. Refit from scratch each cycle.
        training = [e for i, e in enumerate(prepared) if i % 4 != 3 and len(e[2])]
        testing = [e for i, e in enumerate(prepared) if i % 4 == 3]
        add("fit_event_samples", len(training))
        add("eval_event_samples", len(testing))
        add("fit_triplet_samples", sum(len(e[2]) for e in training))
        add("eval_triplet_samples", sum(len(e[2]) for e in testing))
        for _, _, _, support in testing:
            for k in SUPPORT[1:]:
                add("coverage/" + k, support[k], support["hits"])
        fits = {}
        if len(training) >= 4 and sum(bool(len(e[2])) for e in testing) >= 2:
            for rep in ("raw", "embedding"):
                fits[rep] = fit_ranker(training, rep)
        success = len(fits) == 2 and all(fit is not None for fit in fits.values())
        add("fit_success_fraction", success)
        for event, candidates, triples, _ in testing:
            accuracies = {}
            for method in METHODS:
                if method in ("raw", "embedding") and not success:
                    continue
                fit = fits.get(method)
                if len(triples):
                    value = ranking_metric(event, triples, method, fit)
                    add(f"{method}/triplet_accuracy", value)
                    accuracies[method] = value
                retrieval = retrieval_metrics(event, candidates, method, fit)
                for k, value in retrieval.items():
                    add(f"{method}/{k}", value)
            if success and len(triples):
                add(
                    "embedding_minus_raw/triplet_accuracy",
                    accuracies["embedding"] - accuracies["raw"],
                )


def fit_ranker(training, rep):
    pairs = []
    scaler = StandardScaler()
    for event, _, triples, _ in training:
        pos, neg = differences(event[rep], triples)
        weight = particle_weights(event["labels"][triples[:, 0]])
        pairs.append((pos, neg, weight))
        scaler.partial_fit(
            np.concatenate([pos, neg]), sample_weight=np.tile(weight / 2, 2)
        )
    design = np.concatenate(
        [scaler.transform(p) - scaler.transform(n) for p, n, _ in pairs]
    )
    weight = np.concatenate([w for _, _, w in pairs])
    weight *= len(weight) / weight.sum()
    model = LogisticRegression(
        C=0.1, fit_intercept=False, max_iter=200, tol=1e-4, solver="lbfgs"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(
            np.concatenate([design, -design]),
            np.repeat([1, 0], len(design)),
            sample_weight=np.tile(weight, 2),
        )
    if model.n_iter_[0] >= model.max_iter:
        return None  # Report unsupported fit, rather than a misleading accuracy.
    return dict(model=model, scaler=scaler)


def retrieval_metrics(event, candidates, method, fit):
    if not candidates:
        return {}
    anchors = np.array(list(candidates))
    a = np.concatenate([np.repeat(key, len(candidates[key])) for key in anchors])
    b = np.concatenate([candidates[key] for key in anchors])
    scores = score_edges(event, method, fit, a, b)
    top, competing, start = [], [], 0
    for anchor in anchors:
        c = candidates[anchor]
        s = scores[start : start + len(c)]
        positive = event["labels"][c] == event["labels"][anchor]
        top.append(float(positive[s == s.max()].mean()))
        competing.append(positive.any() and not positive.all())
        start += len(c)
    top, competing = np.array(top), np.array(competing)
    ret = dict(
        top1_all_candidate_anchors=float(
            particle_weights(event["labels"][anchors]) @ top
        )
    )
    if competing.any():
        ret["top1_competing"] = float(
            particle_weights(event["labels"][anchors[competing]]) @ top[competing]
        )
    return ret
