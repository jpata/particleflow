import numpy as np

from mlpf.model.calo_probe import (
    accuracy,
    build_candidates,
    differences,
    particle_weights,
)


def toy():
    x = np.zeros((5, 15), dtype=np.float32)
    x[:, 0] = 2
    x[:, 4] = 1
    x[:, 12] = 10
    x[:, 14] = [0, 1, 1, 0, 1]
    x[:, 6] = [0, 10, 11, 1, 10]
    x[:, 2] = [0, 0.1, 0.11, 0.01, 0.1]
    x[4, 13] = 1  # different detector side
    return x, np.array([1, 1, 2, 3, 1])


def test_candidates_are_geometry_only_and_respect_surfaces():
    x, labels = toy()
    candidates, triples, support = build_candidates(x, labels)
    np.testing.assert_equal(candidates[0], [1, 2])
    np.testing.assert_equal(triples[triples[:, 0] == 0], [[0, 1, 2]])
    other = build_candidates(x, np.arange(5) + 1)[0]
    for key in candidates:
        np.testing.assert_equal(candidates[key], other[key])
    assert support["target_layer_exists"] == 2
    assert support["same_particle_in_target_layer"] == 1
    assert support["positive_in_candidates"] == 1


def test_requires_both_distances_to_match():
    x, labels = toy()
    x[2, 2] = 1.0  # spatially near, directionally far
    assert len(build_candidates(x, labels)[1]) == 0
    x, labels = toy()
    x[2, 6] = 30  # directionally near, spatially far
    assert len(build_candidates(x, labels)[1]) == 0


def test_particle_weights_and_ties():
    w = particle_weights(np.array([10, 10, 10, 20]))
    np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    np.testing.assert_equal(accuracy(np.array([-1.0, 0.0, 1.0])), [0, 0.5, 1])
    z = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    p, n = differences(z, np.array([[0, 1, 2]]))
    np.testing.assert_equal(p - n, [[-2, -2]])


def test_no_target_layer_reports_zero_coverage():
    x, labels = toy()
    candidates, triples, support = build_candidates(x, labels, layer_gap=5)
    assert candidates == {}
    assert triples.shape == (0, 3)
    assert support["hits"] == 5
    assert support["target_layer_exists"] == 0
