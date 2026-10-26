"""Contract tests for the independent reconstructed-track/calo-hit target view."""

import awkward as ak
import fastjet
import numpy as np
import pytest

from mlpf.conf import ParticleFeatures
from mlpf.data.target_building import EventData
from mlpf.data.track_calo_targets import _match, build_track_calo_targets

NAMES = ParticleFeatures.get_names()
PN = NAMES.index("particle_number")
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


def event(energies, hit_edges, track_edges=([], [], []), n_tracks=0, calo_mask=None):
    n = len(energies)
    if calo_mask is None:
        calo_mask = np.ones(max(hit_edges[1], default=-1) + 1, dtype=bool)
    features = {name: np.zeros(n) for name in NAMES}
    features.update(PDG=np.full(n, 211), charge=np.ones(n), pt=np.asarray(energies), energy=np.asarray(energies), phi=np.zeros(n))
    return EventData(
        ak.Record(features),
        ak.Record({"type": np.zeros(len(calo_mask))}),
        ak.Record({"type": []}),
        ak.Record({"type": np.zeros(n_tracks)}),
        hit_edges,
        track_edges,
        ([], [], []),
        ([], []),
    ), np.asarray(calo_mask)


def build(*args, **kwargs):
    data, mask = event(*args, **kwargs)
    result = build_track_calo_targets(data, mask, jetdef=JETDEF, jet_ptcut=0.1)
    targets = result["track_calo_target_particles"]
    rows = np.concatenate((result["ytarget_track_calo_track"], result["ytarget_track_calo_hit"]))
    representatives = rows[rows[:, 0] != 0]
    assert sorted(representatives[:, PN]) == list(range(1, len(targets) + 1))
    assert np.isfinite(rows).all()
    np.testing.assert_allclose(representatives[:, 6].sum(), targets[:, 6].sum())
    for row in representatives:
        np.testing.assert_array_equal(row, targets[int(row[PN]) - 1])
    a = result["track_calo_energy_accounting"]
    assert a[0] == pytest.approx(a[1:].sum())
    return result


def test_track_and_calo_single_representative_and_duplicate_tracks():
    result = build([10], ([0, 0], [0, 1], [4, 3]), ([0, 0], [0, 1], [1, 0.8]), n_tracks=2)
    assert result["ytarget_track_calo_track"][:, 0].tolist() == [211, 0]
    assert result["ytarget_track_calo_track"][:, PN].tolist() == [1, 1]
    assert result["ytarget_track_calo_hit"][:, PN].tolist() == [1, 1]
    assert not np.any(result["ytarget_track_calo_hit"][:, 0])


def test_cluster_independent_targets_with_separate_hits():
    data, mask = event([10, 5], ([0, 1], [0, 1], [5, 3]))
    data.cluster_features = ak.Record({"type": [0]})
    data.hit_to_cluster = (np.array([0, 1]), np.array([0, 0]), np.ones(2))
    result = build_track_calo_targets(data, mask, jetdef=JETDEF, jet_ptcut=0.1)
    assert len(result["track_calo_target_particles"]) == 2
    assert result["ytarget_track_calo_hit"][:, 0].tolist() == [130, 130]
    assert result["ytarget_track_calo_hit"][:, 1].tolist() == [0, 0]


def test_multistep_augmenting_path():
    representatives, owners = _match([[0, 1], [1, 2], [0]], [0, 1, 2], 3)
    assert representatives.tolist() == [1, 2, 0]
    assert owners.tolist() == [2, 0, 1]


def test_merge_transfers_membership_before_argmax():
    # Only two representative cells for three particles. Third merges to gp0.
    # On cell1 gp1=4, gp0=3 and merged gp2=2: merged gp0 becomes dominant.
    result = build([20, 10, 5], ([0, 0, 1, 2, 2], [0, 1, 1, 0, 1], [8, 3, 4, 3, 2]), ([0, 1, 2], [0, 0, 0], [0.1, 0.15, 0.1]), n_tracks=1)
    assert result["track_calo_original_to_target"].tolist() == [0, 1, 0]
    assert result["track_calo_target_particles"][:, 6].tolist() == [25, 10]
    # Duplicate/unclaimed track is below representative threshold, but retains
    # membership and must choose the summed host contribution .2 > .15.
    assert result["ytarget_track_calo_track"][0, PN] == 1
    assert result["ytarget_track_calo_track"][0, 0] == 0


def test_tracker_only_visibility_is_removed():
    result = build([10], ([0], [0], [8]), calo_mask=[False])
    assert len(result["track_calo_target_particles"]) == 0
    assert result["track_calo_energy_accounting"].tolist() == [10, 10, 0, 0]


def test_track_only_supported_particle_and_empty_event():
    result = build([10], ([], [], []), ([0], [0], [1]), n_tracks=1)
    assert result["ytarget_track_calo_track"][0, 0] == 211
    assert build([], ([], [], []))["track_calo_target_particles"].shape == (0, len(NAMES))


def test_track_conflict_without_calo_is_explicit_drop():
    result = build([10, 5], ([], [], []), ([0, 1], [0, 0], [0.6, 0.4]), n_tracks=1)
    assert result["track_calo_original_to_target"].tolist() == [0, -1]
    assert result["track_calo_energy_accounting"].tolist() == [15, 0, 5, 10]


def test_duplicate_edges_and_jet_indices():
    result = build([10, 5], ([0, 0, 1], [0, 0, 1], [2, 3, 3]))
    assert result["track_calo_target_particles"][:, 12].tolist() == [0, 0]
    assert len(result["targetjet_track_calo"]) == 1


def test_input_unchanged_and_invalid_mask():
    data, mask = event([10], ([0], [0], [5]))
    before = ak.to_list(data.gen_features)
    build_track_calo_targets(data, mask, jetdef=JETDEF, jet_ptcut=0.1)
    assert ak.to_list(data.gen_features) == before
    with pytest.raises(ValueError, match="calo_mask"):
        build_track_calo_targets(data, [], jetdef=JETDEF, jet_ptcut=0.1)


def test_matching_cardinality_against_scipy():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    rng = np.random.default_rng(42)
    for _ in range(50):
        n, m = rng.integers(1, 25, size=2)
        edges = rng.random((n, m)) < 0.15
        candidates = [np.flatnonzero(row).tolist() for row in edges]
        representatives, _ = _match(candidates, rng.permutation(n), m)
        reference = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
        assert np.sum(representatives >= 0) == np.sum(reference >= 0)


def test_validator_key4hep_record_with_empty_tracks(tmp_path):
    from scripts.studies.validate_track_calo_targets import validate

    result = build([10], ([0], [0], [5]))
    result.update(
        X_track=np.zeros((0, 16)),
        X_hit_calo=np.zeros((1, 15)),
        ytarget_track=np.zeros((0, len(NAMES))),
        ytarget_cluster=result["track_calo_target_particles"],
    )
    path = tmp_path / "targets.parquet"
    ak.to_parquet(ak.Record({name: ak.from_iter([value]) for name, value in result.items()}), path)
    report = validate(path)
    assert report["events"] == 1
    assert report["targets"] == 1
    assert report["track_representatives"] == 0
