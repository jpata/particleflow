"""One sparse ownership matching for tracks, clusters and unmatched particles."""

from itertools import product

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlpf.data.target_building import match_particles_to_inputs
from mlpf.data.target_building import assign_genparticles_to_obj_and_merge
from tests.test_postprocessing_merge import make_eventdata


def match(tracks, deposits, cluster_energy, particle_deposit):
    return match_particles_to_inputs(csr_matrix(tracks), csr_matrix(deposits), cluster_energy, particle_deposit)


def test_track_preference_frees_cluster_for_another_particle():
    result = match([[0.9], [0.1]], [[100.0], [80.0]], [180.0], [100.0, 80.0])
    assert result.tolist() == [[0, -1], [-1, 0]]


def test_global_cluster_match_avoids_greedy_conflict():
    result = match(np.empty((2, 0)), [[9.0, 8.0], [8.0, 0.0]], [17.0, 8.0], [17.0, 8.0])
    assert result.tolist() == [[-1, 1], [-1, 0]]


def test_weak_track_links_cannot_be_used():
    result = match([[0.5], [0.5]], np.empty((2, 0)), [], [0.0, 0.0])
    assert result.tolist() == [[-1, -1], [-1, -1]]


def test_duplicate_track_cannot_go_to_minority_particle():
    result = match([[0.9, 0.8], [0.1, 0.2]], np.empty((2, 0)), [], [0.0, 0.0])
    assert result.tolist() == [[0, -1], [-1, -1]]


@pytest.mark.parametrize("n_particle,n_track,n_cluster", [(0, 0, 0), (0, 2, 3), (2, 0, 0), (3, 2, 4)])
def test_empty_graphs(n_particle, n_track, n_cluster):
    result = match(np.zeros((n_particle, n_track)), np.zeros((n_particle, n_cluster)), np.zeros(n_cluster), np.zeros(n_particle))
    assert result.shape == (n_particle, 2)
    assert np.all(result == -1)


def test_unmatched_choice_and_unique_inputs():
    result = match([[0.9], [0.1], [0.0]], [[0.0], [3.0], [2.0]], [5.0], [0.0, 3.0, 2.0])
    assert result.tolist() == [[0, -1], [-1, 0], [-1, -1]]


@pytest.mark.parametrize("deposits,cluster_energy,particle_deposit", [([[2.0]], [1.0], [2.0]), ([[2.0]], [2.0], [1.0]), ([[1.0]], [0.0], [0.0])])
def test_invalid_energy_totals(deposits, cluster_energy, particle_deposit):
    with pytest.raises(ValueError):
        match([[0.0]], deposits, cluster_energy, particle_deposit)


def test_objective_matches_exhaustive_search():
    rng = np.random.default_rng(123)
    for _ in range(20):
        # The extra Dirichlet component represents hits without a selected owner.
        tracks = rng.dirichlet(np.ones(4), size=2).T[:3]
        deposits = rng.uniform(0, 5, (3, 2))
        deposits[rng.random((3, 2)) < 0.4] = 0
        energy = deposits.sum(axis=0) + 1
        total = deposits.sum(axis=1) + 1
        scores = np.column_stack((1 + tracks, 2 * deposits / (energy[None, :] + total[:, None])))
        allowed = np.column_stack((tracks > 0.5, deposits > 0))
        candidates = [[-1, *np.flatnonzero(allowed[i])] for i in range(3)]
        best = 0.0
        for columns in product(*candidates):
            used = [c for c in columns if c >= 0]
            if len(set(used)) != len(used):
                continue
            best = max(best, sum(scores[i, c] for i, c in enumerate(columns) if c >= 0))
        result = match(tracks, deposits, energy, total)
        value = sum(scores[i, t] if t >= 0 else scores[i, 2 + c] if c >= 0 else 0 for i, (t, c) in enumerate(result))
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
