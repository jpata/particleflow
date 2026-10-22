import numpy as np

from scripts.studies.probe_hit_instances import (
    candidate_edges,
    cluster,
    instance_metrics,
    pair_features,
)


def test_permutation_invariant_instances():
    truth = np.array([1, 1, 2, 2, 3])
    result = instance_metrics(truth, np.array([9, 9, 8, 8, 7]))
    assert result["ari"] == 1
    assert result["instance_f1_iou0p5"] == 1
    assert result["bcubed_f1"] == 1
    merged = instance_metrics(truth, np.zeros(5, dtype=int))
    assert merged["bcubed_recall"] == 1
    assert merged["bcubed_precision"] < 1
    assert merged["instance_f1_iou0p5"] == 0


def test_graph_oracle_and_singletons():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    labels = np.array([1, 1, 2, 2, 3])
    pred = cluster(5, edges, labels[edges[:, 0]] == labels[edges[:, 1]])
    assert instance_metrics(labels, pred)["ari"] == 1
    assert len(np.unique(cluster(5, edges, np.zeros(3, dtype=bool)))) == 5


def test_candidate_graph_and_symmetric_pairs():
    x = np.random.default_rng(1).normal(size=(7, 16))
    edges = candidate_edges(x, 2)
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len(edges) == len(np.unique(edges, axis=0))
    np.testing.assert_equal(
        pair_features({"raw": x}, "raw", edges),
        pair_features({"raw": x}, "raw", edges[:, ::-1]),
    )
    assert candidate_edges(x[:1], 3).shape == (0, 2)
