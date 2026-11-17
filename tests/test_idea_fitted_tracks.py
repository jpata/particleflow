"""Fitted IDEA inputs must be measured, quality-selected and truth-labelled separately."""

import awkward as ak
import numpy as np
import pytest

from mlpf.data.key4hep import postprocessing as pp
from mlpf.data.key4hep.idea_tracks import (
    TRACK_COLLECTION,
    TRACK_HIT_COLLECTIONS,
    TRACK_HIT_LINKS,
    fitted_track_selection,
    fitted_track_truth,
    preferred_track_representatives,
)


def event_fixture():
    ids = {"MCParticles": 1}
    ids.update({name: 10 + i for i, name in enumerate(TRACK_HIT_COLLECTIONS)})
    ids.update({name: 20 + i for i, name in enumerate(TRACK_HIT_LINKS.values())})
    t = {
        "type": [0, 0],
        "chi2": [2.0, 3.0],
        "ndf": [10, 10],
        "trackStates_begin": [0, 2],
        "trackStates_end": [2, 4],
        "trackerHits_begin": [0, 4],
        "trackerHits_end": [4, 8],
    }
    s = {
        "location": [3, 1, 3, 1],
        "omega": [0.03, 0.000599584916, 0.02, 0.000299792458],
        "phi": [2.0, 0.4, 2.0, -0.3],
        "tanLambda": [1.0, 0.2, 1.0, -0.4],
        "D0": [0.0, 1.0, 0.0, 2.0],
        "Z0": [0.0, 2.0, 0.0, 3.0],
        "time": [0.0] * 4,
        "referencePoint.x": [100.0, 0.0, 100.0, 0.0],
        "referencePoint.y": [0.0] * 4,
        "referencePoint.z": [0.0] * 4,
    }
    prop = {
        TRACK_COLLECTION: [ak.Array({f"{TRACK_COLLECTION}.{k}": v for k, v in t.items()})],
        f"_{TRACK_COLLECTION}_trackStates": [ak.Array({f"_{TRACK_COLLECTION}_trackStates.{k}": v for k, v in s.items()})],
    }

    def put(key, values):
        prop[key] = [ak.Array(values)]

    put("MCParticles.PDG", [211, -13, -211])
    put("MCParticles.generatorStatus", [1, 1, 0])
    put("MCParticles.charge", [1.0, -1.0, -1.0])
    for i, (links, sim) in enumerate(TRACK_HIT_LINKS.items()):
        reco = TRACK_HIT_COLLECTIONS[i]
        n = 4 if i in (0, 4) else 0
        for direction, indices, coll in [("from", list(range(n)), ids[reco]), ("to", list(range(n)), ids[sim])]:
            put(f"_{links}_{direction}/_{links}_{direction}.index", indices)
            put(f"_{links}_{direction}/_{links}_{direction}.collectionID", [coll] * n)
        put(links + ".weight", [1.0] * n)
        particles = [0, 0, 0, -1] if i == 0 else [1] * n
        put(f"_{sim}_particle/_{sim}_particle.index", particles)
        put(f"_{sim}_particle/_{sim}_particle.collectionID", [ids["MCParticles"]] * n)
        for axis in "xyz":
            put(f"{reco}.position.{axis}", [float((i + 1) * 20 + j) if axis == "x" else 0.0 for j in range(n)])
    put(f"_{TRACK_COLLECTION}_trackerHits/_{TRACK_COLLECTION}_trackerHits.index", [0, 1, 2, 3, 0, 1, 2, 3])
    put(f"_{TRACK_COLLECTION}_trackerHits/_{TRACK_COLLECTION}_trackerHits.collectionID", [ids["VTXBDigis"]] * 4 + [ids["DCHDigis"]] * 4)
    return prop, ids


def replace_field(prop, collection, field, index, value):
    arr = prop[collection][0]
    fields = {name: ak.to_numpy(arr[name]).copy() for name in arr.fields}
    fields[f"{collection}.{field}"][index] = value
    prop[collection][0] = ak.Array(fields)


def test_features_use_ip_fit_and_never_proxy_or_mc_momentum():
    prop, _ = event_fixture()
    selection = fitted_track_selection(prop, 0, 2.0)
    np.testing.assert_array_equal(selection[1], [1, 3])
    features = pp.idea_track_to_features(prop, 0, 2.0, selection=selection)
    np.testing.assert_allclose(features["pt"], [1.0, 2.0])
    np.testing.assert_allclose(features["phi"], [0.4, -0.3])
    np.testing.assert_allclose(features["D0"], [1.0, 2.0])
    np.testing.assert_allclose(features["Z0"], [2.0, 3.0])
    assert "TracksFromGenParticles" not in prop
    assert "MCParticles.momentum.x" not in prop


@pytest.mark.parametrize("defect", ["ndf", "chi2", "nan", "zero_curvature", "no_ip", "two_ip", "no_hits"])
def test_selection_rejects_bad_fit_without_truth_cuts(defect):
    prop, _ = event_fixture()
    if defect == "ndf":
        replace_field(prop, TRACK_COLLECTION, "ndf", 0, -5)
    elif defect == "chi2":
        replace_field(prop, TRACK_COLLECTION, "chi2", 0, -1.0)
    elif defect == "nan":
        replace_field(prop, f"_{TRACK_COLLECTION}_trackStates", "phi", 1, np.nan)
    elif defect == "zero_curvature":
        replace_field(prop, f"_{TRACK_COLLECTION}_trackStates", "omega", 1, 0.0)
    elif defect == "no_ip":
        replace_field(prop, f"_{TRACK_COLLECTION}_trackStates", "location", 1, 3)
    elif defect == "two_ip":
        replace_field(prop, f"_{TRACK_COLLECTION}_trackStates", "location", 0, 1)
    elif defect == "no_hits":
        replace_field(prop, TRACK_COLLECTION, "trackerHits_end", 0, 0)
    selected, states = fitted_track_selection(prop, 0, 2.0)
    np.testing.assert_array_equal(selected, [1])
    np.testing.assert_array_equal(states, [3])


def test_hit_truth_keeps_collection_identity_and_unknown_hit_denominator():
    prop, ids = event_fixture()
    relation, radius, hits = fitted_track_truth(prop, 0, ids, [0, 1])
    np.testing.assert_array_equal(relation[0], [0, 1])
    np.testing.assert_array_equal(relation[1], [0, 1])
    np.testing.assert_allclose(relation[2], [0.75, 1.0])
    np.testing.assert_allclose(radius, [20.0, 100.0])
    np.testing.assert_array_equal(hits, [4, 4])
    # Quality filtering compresses indices; labels must follow the selected rows.
    compressed, _, _ = fitted_track_truth(prop, 0, ids, [1])
    np.testing.assert_array_equal(compressed[0], [1])
    np.testing.assert_array_equal(compressed[1], [0])


def test_secondary_and_low_purity_tracks_stay_inputs_but_have_no_primary_label():
    prop, ids = event_fixture()
    prop["_VertexBarrelCollection_particle/_VertexBarrelCollection_particle.index"][0] = ak.Array([0, 0, 1, -1])
    prop["_DCHCollection_particle/_DCHCollection_particle.index"][0] = ak.Array([2, 2, 2, 2])
    selected, _ = fitted_track_selection(prop, 0, 2.0)
    assert len(selected) == 2
    relation, _, _ = fitted_track_truth(prop, 0, ids, selected)
    assert len(relation[0]) == 0  # no ancestor momentum assigned to secondary 2


def test_unlinked_hits_do_not_inflate_truth_purity():
    prop, ids = event_fixture()
    prop["_VTXBSimDigiLinks_from/_VTXBSimDigiLinks_from.index"][0] = ak.Array([0, 1, 8, 9])
    relation, _, _ = fitted_track_truth(prop, 0, ids, [0])
    assert len(relation[0]) == 0  # two of four measurements, not two of two linked hits


def test_invalid_original_hit_reference_is_rejected():
    prop, ids = event_fixture()
    prop[f"_{TRACK_COLLECTION}_trackerHits/_{TRACK_COLLECTION}_trackerHits.collectionID"][0] = ak.Array([999] * 8)
    with pytest.raises(ValueError, match="unknown or invalid"):
        fitted_track_truth(prop, 0, ids, [0])


def test_duplicate_representative_prefers_purity_then_hit_count_then_fit_quality():
    relation = (np.array([4, 4, 5, 5]), np.array([0, 1, 2, 3]), np.array([0.75, 1.0, 1.0, 1.0]))
    tracks = {"chi2": np.array([1.0, 5.0, 5.0, 1.0]), "ndf": np.full(4, 10)}
    representatives = preferred_track_representatives(relation, tracks, [30, 20, 20, 20])
    assert representatives == {4: 1, 5: 3}


def test_empty_fitted_collection_has_empty_features_without_proxy_fallback():
    prop, ids = event_fixture()
    for collection in [TRACK_COLLECTION, f"_{TRACK_COLLECTION}_trackStates"]:
        arr = prop[collection][0]
        prop[collection][0] = ak.Array({name: ak.to_numpy(arr[name])[:0] for name in arr.fields})
    selection = fitted_track_selection(prop, 0, 2.0)
    features = pp.idea_track_to_features(prop, 0, 2.0, selection=selection)
    assert len(features["pt"]) == 0
    relation, radii, counts = fitted_track_truth(prop, 0, ids, selection[0])
    assert len(relation[0]) == len(radii) == len(counts) == 0


def test_default_reader_refuses_root_with_only_proxy_tracks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class Branch:
        def __init__(self, value):
            self.value = value

        def array(self):
            return [self.value]

    prefix = "events___CollectionTypeInfo/events___CollectionTypeInfo."
    metadata = {prefix + "name": Branch(["MCParticles", "TopoClusterAllCells", "TracksFromGenParticles"]), prefix + "collectionID": Branch([1, 2, 3])}
    monkeypatch.setattr(pp.uproot, "open", lambda *args, **kwargs: {"events": SimpleNamespace(num_entries=1), "podio_metadata": metadata})
    with pytest.raises(ValueError, match="no FittedTracks"):
        pp.process_one_file_idea("legacy.root", str(tmp_path / "new.parquet"))


@pytest.mark.parametrize("marked", [True, False])
def test_cached_proxy_parquet_cannot_satisfy_fitted_postprocessing(tmp_path, marked):
    data = {"X_track": ak.Array([[]])}
    if marked:
        data["metadata_track_source"] = ak.Array(["truth_seeded"])
    cached = tmp_path / "cached.parquet"
    ak.to_parquet(ak.Record(data), cached)
    with pytest.raises(ValueError, match="Cached IDEA parquet"):
        pp.process_one_file("unused.root", str(cached), "idea")
