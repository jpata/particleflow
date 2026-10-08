import awkward as ak
import numpy as np
import pytest

from mlpf.conf import EDM4HEP
from mlpf.data.key4hep.postprocessing import (
    cluster_to_features,
    decode_cellid_field,
    get_hit_matrix_and_genadj,
    hit_cluster_adj,
    hits_to_features,
    parse_cellid_encoding,
)
from mlpf.heptfds.edm4hep_utils.utils_hits import X_FEATURES


HIT_FEATURES = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]
TRACKER_ENCODING = "system:5,side:-2,layer:6,module:11,sensor:8"


def make_hit_data(collection):
    values = {
        "type": [0, 0],
        "cellID": [1, 2],
        "energy": [0.1, 0.2],
        "energyError": [0.0, 0.0],
        "time": [0.0, 0.0],
        "position.x": [1.0, 2.0],
        "position.y": [0.0, 0.0],
        "position.z": [1.0, 2.0],
    }
    return ak.Array({f"{collection}.{feature}": [value] for feature, value in values.items()})


@pytest.mark.parametrize(
    ("collection", "expected_subdetector", "expected_elemtype"),
    [
        ("VXDTrackerHits", 3, 1),
        ("ITrackerEndcapHits", 3, 1),
        ("ECALBarrel", 0, 2),
        ("HCALBarrel", 1, 2),
        ("HCALOther", 1, 2),
        ("MUON", 2, 2),
        ("LumiCal_Hits", 2, 2),
        # MAIA names its calorimeters Ecal*/Hcal*, which a case-sensitive prefix match used to
        # send to subdetector 2, leaving energy_ecal and energy_hcal zero on every MAIA cluster
        ("EcalBarrelCollectionRec", 0, 2),
        ("EcalEndcapCollectionRec", 0, 2),
        ("HcalBarrelCollectionRec", 1, 2),
        ("HcalEndcapCollectionRec", 1, 2),
        ("IBTrackerHits", 3, 1),
        ("VETrackerHits", 3, 1),
    ],
)
def test_hit_elemtype_follows_subdetector(collection, expected_subdetector, expected_elemtype):
    features = hits_to_features(make_hit_data(collection), 0, collection, HIT_FEATURES, TRACKER_ENCODING)

    np.testing.assert_array_equal(features["subdetector"], [expected_subdetector, expected_subdetector])
    np.testing.assert_array_equal(features["elemtype"], [expected_elemtype, expected_elemtype])


def test_cellid_encoding_parser_supports_implicit_and_explicit_offsets():
    expected = {"system": (0, 5), "side": (5, -2), "layer": (7, 6), "module": (13, 11), "sensor": (24, 8)}

    assert parse_cellid_encoding(TRACKER_ENCODING) == expected
    assert parse_cellid_encoding("system:0:5,side:5:-2,layer:7:6,module:13:11,sensor:24:8") == expected


def test_tracker_surface_fields_are_decoded_from_cellid():
    system = np.array([1, 4, 6], dtype=np.uint64)
    side = np.array([0, -1, 1], dtype=np.int64)
    layer = np.array([5, 3, 2], dtype=np.uint64)
    encoded_side = np.where(side < 0, side + 4, side).astype(np.uint64)
    cellids = system | (encoded_side << np.uint64(5)) | (layer << np.uint64(7))
    hit_data = make_hit_data("ITrackerEndcapHits")
    hit_data = ak.with_field(hit_data, ak.Array([cellids[:2].tolist()]), "ITrackerEndcapHits.cellID")

    features = hits_to_features(hit_data, 0, "ITrackerEndcapHits", HIT_FEATURES, TRACKER_ENCODING)

    np.testing.assert_array_equal(features["system"], system[:2])
    np.testing.assert_array_equal(features["side"], side[:2])
    np.testing.assert_array_equal(features["layer"], layer[:2])
    np.testing.assert_array_equal(decode_cellid_field(cellids, TRACKER_ENCODING, "side"), side)


def test_tracker_without_cellid_encoding_raises_by_default():
    with pytest.raises(RuntimeError, match="CellIDEncoding"):
        hits_to_features(make_hit_data("IBTrackerHits"), 0, "IBTrackerHits", HIT_FEATURES, None)


def test_tracker_without_cellid_encoding_falls_back_to_zero_when_not_required():
    features = hits_to_features(make_hit_data("IBTrackerHits"), 0, "IBTrackerHits", HIT_FEATURES, None, require_cellid_encoding=False)

    np.testing.assert_array_equal(features["subdetector"], [3, 3])
    for field in ("system", "side", "layer"):
        np.testing.assert_array_equal(features[field], [0, 0])


def test_only_lcio_converted_detectors_skip_the_cellid_encoding_requirement():
    assert not EDM4HEP.DETECTORS["maia"].require_cellid_encoding
    for name in ("clic", "cld"):
        assert EDM4HEP.DETECTORS[name].require_cellid_encoding


def make_calo_hits(collection, energies):
    n = len(energies)
    values = {
        "type": [0] * n,
        "cellID": list(range(n)),
        "energy": energies,
        "energyError": [0.0] * n,
        "time": [0.0] * n,
        "position.x": [1.0] * n,
        "position.y": [0.0] * n,
        "position.z": [1.0] * n,
    }
    return ak.Array({f"{collection}.{feature}": [value] for feature, value in values.items()})


def make_clustered_event():
    # A CLIC-like event. LumiCal shares subdetector 2 with the muon system, so only selecting
    # muon hits by collection keeps it out of the muon features.
    collection_ids = {"ECALBarrel": 11, "HCALBarrel": 12, "MUON": 13, "LumiCal_Hits": 14}
    hit_data = {
        "ECALBarrel": make_calo_hits("ECALBarrel", [1.0, 2.0]),
        "HCALBarrel": make_calo_hits("HCALBarrel", [3.0]),
        "MUON": make_calo_hits("MUON", [1e-4, 2e-4, 3e-4, 4e-4]),  # the last one is in no cluster
        "LumiCal_Hits": make_calo_hits("LumiCal_Hits", [5.0, 6.0]),
    }
    hit_features, _, local_to_global = get_hit_matrix_and_genadj(hit_data, ak.Array([{"unused": 0}]), None, 0, collection_ids, 0)

    # Pandora's cluster -> hit relation, stored as (collectionID, index) as in the ROOT files
    references = [
        [(11, 0), (12, 0), (13, 0), (13, 1)],  # ECAL + HCAL + two muon-system hits
        [(14, 0), (14, 1), (11, 1)],  # LumiCal + ECAL, no muon-system hit
        [(13, 2)],  # a lone muon-system hit
    ]
    begin = np.cumsum([0] + [len(r) for r in references[:-1]]).tolist()
    clusters = [
        {
            "PandoraClusters.type": 0,
            "PandoraClusters.position.x": 1.0,
            "PandoraClusters.position.y": 0.0,
            "PandoraClusters.position.z": 1.0,
            "PandoraClusters.iTheta": 1.0,
            "PandoraClusters.phi": 0.0,
            "PandoraClusters.energy": 1.0,
            "PandoraClusters.hits_begin": b,
            "PandoraClusters.hits_end": b + len(r),
        }
        for b, r in zip(begin, references)
    ]
    prop_data = ak.Array(
        [
            {
                "PandoraClusters": clusters,
                "_PandoraClusters_hits/_PandoraClusters_hits.collectionID": [c for r in references for c, _ in r],
                "_PandoraClusters_hits/_PandoraClusters_hits.index": [i for r in references for _, i in r],
            }
        ]
    )
    hit_to_cluster = hit_cluster_adj(prop_data, local_to_global, 0, {v: k for k, v in collection_ids.items()})
    return prop_data, hit_features, hit_to_cluster, collection_ids


def test_cluster_muon_features_follow_pandora_hit_references():
    prop_data, hit_features, hit_to_cluster, collection_ids = make_clustered_event()

    features = cluster_to_features(prop_data, hit_features, hit_to_cluster, 0, (collection_ids["MUON"],))

    np.testing.assert_array_equal(features["num_muon_hits"], [2, 0, 1])
    np.testing.assert_allclose(features["energy_muon"], [3e-4, 0.0, 3e-4])
    # LumiCal does land in energy_other with the muon hits, which is why subdetector 2 cannot be used
    np.testing.assert_allclose(features["energy_other"], [3e-4, 11.0, 3e-4])


def test_cluster_muon_features_are_zero_without_muon_collections():
    prop_data, hit_features, hit_to_cluster, _ = make_clustered_event()

    features = cluster_to_features(prop_data, hit_features, hit_to_cluster, 0)

    np.testing.assert_array_equal(features["num_muon_hits"], [0, 0, 0])
    np.testing.assert_array_equal(features["energy_muon"], [0.0, 0.0, 0.0])


def test_cluster_schema_appends_muon_features():
    names = EDM4HEP.ClusterFeatures.get_names()

    assert names[-2:] == ["num_muon_hits", "energy_muon"]
    # appended, so every existing column keeps its index
    assert names.index("energy_other") == 12 and names.index("sigma_z") == 16


def test_registered_muon_collections_are_read():
    for name, detector in EDM4HEP.DETECTORS.items():
        assert set(detector.muon_collections).issubset(detector.hit_collections), name
    for name in ("clic", "cld", "maia"):
        assert EDM4HEP.DETECTORS[name].muon_collections == ("MUON",)


def test_tfds_hit_schema_retains_detector_surface_fields():
    assert EDM4HEP.HitFeatures.get_names()[-3:] == ["system", "side", "layer"]
    assert X_FEATURES[-3:] == ["system", "side", "layer"]
