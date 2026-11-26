"""Numerical contracts for offline embedding and probe diagnostics."""

import numpy as np
import pytest

from scripts.local.plot_ssl_evolution import (
    load_cohort,
    nearest_neighbors,
    neighbor_overlap,
    precision_recall,
    probe_group_metrics,
    quantile_groups,
)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_neighbors_exclude_self_and_preserve_rotations(metric):
    rng = np.random.default_rng(23)
    x = rng.normal(size=(20, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    neighbors = nearest_neighbors(x, 4, metric)
    assert not (neighbors == np.arange(len(x))[:, None]).any()
    assert neighbor_overlap(neighbors, nearest_neighbors(x @ rotation, 4, metric)) == 1
    assert nearest_neighbors(x[:2], 15, metric).tolist() == [[1], [0]]


def test_precision_recall_groups_ties_and_handles_no_positives():
    target = np.array([True, False, True, False])
    recall, precision, ap = precision_recall(target, np.ones(4))
    np.testing.assert_allclose(recall, [0, 1])
    np.testing.assert_allclose(precision, [1, 0.5])
    assert ap == 0.5
    _, _, ap = precision_recall(target[:3], np.array([0.9, 0.8, 0.7]))
    assert ap == pytest.approx((1 + 2 / 3) / 2)
    _, _, ap = precision_recall(np.zeros(4, dtype=bool), np.ones(4))
    assert ap is None


def test_probe_metrics_exclude_absent_targets_from_pid():
    data = {
        "targets": np.array([[0], [1], [2]]),
        "presence_logits": np.array([[0, 1], [0, 1], [1, 0]]),
        "pid_logits": np.eye(3)[[0, 1, 1]],
    }
    actual = probe_group_metrics(data, np.ones(3, dtype=bool))
    assert (actual["tp"], actual["fp"], actual["fn"]) == (1, 1, 1)
    assert actual["presence_f1"] == 0.5
    assert actual["pid_accuracy"] == 0.5
    assert probe_group_metrics(data, np.zeros(3, dtype=bool))["pid_accuracy"] is None


def test_quantile_bins_partition_ties_and_constant_inputs():
    for values in [np.ones(12), np.array([0, 0, 1, 1, 1, 2, 3, 3, 4])]:
        groups = quantile_groups(values, "value")
        np.testing.assert_array_equal(
            np.stack([mask for _, mask in groups]).sum(0), np.ones(len(values))
        )


def test_cohort_reordering_rejected_even_with_same_digest(tmp_path):
    a = {
        "student": np.eye(3),
        "X": np.eye(3),
        "dataset_index": np.zeros(3),
        "event_index": np.zeros(3),
        "element_index": np.arange(3),
        "cohort_sha256": np.asarray("same"),
    }
    first, second = tmp_path / "step-0.npz", tmp_path / "step-1.npz"
    np.savez(first, **a)
    np.savez(second, **{**a, "element_index": np.arange(3)[::-1]})
    with pytest.raises(ValueError, match="element_index"):
        load_cohort([first, second])
