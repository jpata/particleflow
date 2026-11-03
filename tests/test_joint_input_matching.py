"""Sparse majority-energy matching and shared allocator input checks."""

from itertools import product

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlpf.data.target_building import assign_genparticles_to_obj_and_merge, match_majority_tracks_then_energy
from tests.test_postprocessing_merge import make_eventdata


@pytest.mark.parametrize("n_particle,n_track,n_cluster", [(0, 0, 0), (0, 2, 3), (2, 0, 0), (3, 2, 4)])
def test_empty_graphs(n_particle, n_track, n_cluster):
    result = match_majority_tracks_then_energy(csr_matrix((n_particle, n_track)), csr_matrix((n_particle, n_cluster)), np.zeros(n_particle))
    assert result.shape == (n_particle, 2)
    assert np.all(result == -1)


def test_energy_objective_matches_exhaustive_search():
    rng = np.random.default_rng(123)
    for _ in range(20):
        deposits = rng.uniform(0, 5, (3, 2))
        deposits[rng.random((3, 2)) < 0.4] = 0
        energy = rng.uniform(1, 20, 3)
        scores = energy[:, None] + deposits
        candidates = [[-1, *np.flatnonzero(deposits[i] > 0)] for i in range(3)]
        best = 0.0
        for columns in product(*candidates):
            used = [c for c in columns if c >= 0]
            if len(set(used)) != len(used):
                continue
            best = max(best, sum(scores[i, c] for i, c in enumerate(columns) if c >= 0))
        result = match_majority_tracks_then_energy(csr_matrix((3, 0)), csr_matrix(deposits), energy)
        value = sum(scores[i, c] for i, (_, c) in enumerate(result) if c >= 0)
        assert value == pytest.approx(best)


def test_explicit_matching_uses_existing_merges():
    event = make_eventdata([10.0, 2.0, 1.0], [4.0, 2.0, 0.0])
    cleaned, objects, *_ = assign_genparticles_to_obj_and_merge(event, input_assignment=np.array([[-1, 0], [-1, -1], [-1, -1]]))
    assert objects.tolist() == [[-1, 0]]
    assert cleaned.gen_features["energy"].tolist() == [12.0]
    assert cleaned.gp_merges[1].tolist() == [1]


@pytest.mark.parametrize(
    "assignment",
    [np.array([[-1, 0], [-1, 0]]), np.array([[-1, 1], [-1, -1]]), np.array([[-1, 0], [-1, -1]], dtype=float), np.array([[0, 0], [-1, -1]])],
)
def test_invalid_explicit_matching(assignment):
    with pytest.raises(ValueError):
        assign_genparticles_to_obj_and_merge(make_eventdata([10.0, 2.0], [4.0, 2.0]), input_assignment=assignment)
