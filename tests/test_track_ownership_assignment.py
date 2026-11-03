"""Track ownership, duplicate tracks, and cluster fallback in target construction."""

import awkward as ak
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mlpf.data.target_building import assign_genparticles_to_obj_and_merge
from scripts.studies.compare_colliderml_matching import assign_tracks_by_hit_ownership
from tests.test_postprocessing_merge import make_eventdata


def test_minority_cannot_claim_track():
    matrix = csr_matrix([[0.875], [0.125]])
    assert assign_tracks_by_hit_ownership(matrix, [8]).tolist() == [0, -1]


def test_equal_shares_and_unowned_hits():
    matrix = csr_matrix([[0.5, 0.49, 0.6], [0.5, 0.0, 0.0]])
    assert assign_tracks_by_hit_ownership(matrix, [6, 10, 10]).tolist() == [2, -1]


def test_duplicate_tracks_have_no_minority_fallback():
    matrix = csr_matrix([[0.9, 0.8], [0.1, 0.2]])
    assert assign_tracks_by_hit_ownership(matrix, [10, 20]).tolist() == [0, -1]


def test_ties_use_owned_hit_count_then_track_index():
    matrix = csr_matrix([[0.8, 0.8, 0.8]])
    assert assign_tracks_by_hit_ownership(matrix, [5, 10, 10]).tolist() == [1]


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0), (2, 3)])
def test_empty_links(shape):
    assert assign_tracks_by_hit_ownership(csr_matrix(shape), np.zeros(shape[1])).tolist() == [-1] * shape[0]


@pytest.mark.parametrize("values", [[[0.7], [0.7]], [[float("nan")]], [[-0.1]]])
def test_invalid_hit_fractions(values):
    with pytest.raises(ValueError):
        assign_tracks_by_hit_ownership(csr_matrix(values), [10])


def test_zero_hit_track_cannot_get_target():
    assert assign_tracks_by_hit_ownership(csr_matrix([[1.0]]), [0]).tolist() == [-1]


def test_rejected_track_link_falls_back_to_cluster_and_preserves_energy():
    event = make_eventdata([1.0, 10.0, 2.0], [1.0, 2.0, 1.0])
    event.gen_features = ak.with_field(event.gen_features, event.genparticle_to_hit[2], "gp_to_cluster")
    event.track_features = ak.Array({"type": [1]})
    event.genparticle_to_track = (np.array([0, 1]), np.array([0, 0]), np.array([0.875, 0.125]))
    old, old_objects, *_ = assign_genparticles_to_obj_and_merge(event)
    new, new_objects, *_ = assign_genparticles_to_obj_and_merge(event, assignment="bipartite", cluster_energy=[4.0])
    # The 1 GeV majority particle gets the track; the 10 GeV minority contributor
    # gets the cluster and absorbs the remaining 2 GeV particle.
    assert ak.to_list(old.gen_features["energy"])[old_objects[:, 0].tolist().index(0)] == 10.0
    assert new_objects.tolist() == [[0, -1], [-1, 0]]
    assert ak.to_list(new.gen_features["energy"]) == [1.0, 12.0]
    assert float(ak.sum(old.gen_features["energy"])) == pytest.approx(13.0)
    assert float(ak.sum(new.gen_features["energy"])) == pytest.approx(13.0)
    assert new.gp_merges[1].tolist() == [2]


def test_rule_is_independent_of_particle_order():
    matrix = csr_matrix([[0.8, 0.0], [0.2, 0.9]])
    selected = assign_tracks_by_hit_ownership(matrix, [10, 10])
    permuted = assign_tracks_by_hit_ownership(matrix[[1, 0]], [10, 10])
    assert np.array_equal(selected, permuted[[1, 0]])


def test_roundoff_cannot_assign_one_track_twice():
    selected = assign_tracks_by_hit_ownership(csr_matrix([[0.5000001], [0.5000001]]), [10])
    assert np.sum(selected >= 0) == 1


def test_particle_without_cluster_fallback_is_explicitly_dropped(capsys):
    event = make_eventdata([1.0, 10.0], [1.0, 0.0])
    event.gen_features = ak.with_field(event.gen_features, event.genparticle_to_hit[2], "gp_to_cluster")
    event.track_features = ak.Array({"type": [1]})
    event.genparticle_to_track = (np.array([0, 1]), np.array([0, 0]), np.array([0.6, 0.4]))
    cleaned, objects, *_ = assign_genparticles_to_obj_and_merge(event, assignment="bipartite", cluster_energy=[1.0])
    assert objects.tolist() == [[0, -1]]
    assert float(ak.sum(cleaned.gen_features["energy"])) == 1.0
    assert "Dropped 1 unmatched genparticles" in capsys.readouterr().out
