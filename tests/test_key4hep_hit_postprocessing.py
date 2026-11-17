import awkward as ak
import numpy as np
import pytest

from mlpf.conf import EDM4HEP
from mlpf.data.key4hep.postprocessing import (
    CALOHIT_LINK_FIELDS,
    append_relation_truth_links,
    cluster_to_features,
    decode_cellid_field,
    get_hit_matrix_and_genadj,
    hit_cluster_adj,
    hits_to_features,
    parse_cellid_encoding,
    track_to_features,
)
from mlpf.data.target_building import represented_pid
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


def make_clustered_event(position=(1.0, 0.0, 1.0), itheta=1.0, phi=0.0):
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
            "PandoraClusters.position.x": position[0],
            "PandoraClusters.position.y": position[1],
            "PandoraClusters.position.z": position[2],
            "PandoraClusters.iTheta": itheta,
            "PandoraClusters.phi": phi,
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


def test_cluster_angles_can_come_from_the_centroid():
    # Pandora's iTheta/phi (here pointing along +x) are the shower axis; the cluster sits at +y, z = r
    prop_data, hit_features, hit_to_cluster, _ = make_clustered_event(position=(0.0, 2.0, 2.0), itheta=np.pi / 2, phi=0.0)

    axis = cluster_to_features(prop_data, hit_features, hit_to_cluster, 0)
    centroid = cluster_to_features(prop_data, hit_features, hit_to_cluster, 0, angles_from_position=True)

    np.testing.assert_allclose(axis["eta"], 0.0, atol=1e-6)
    np.testing.assert_allclose(axis["sin_phi"], 0.0, atol=1e-6)
    np.testing.assert_allclose(centroid["eta"], np.arcsinh(1.0), rtol=1e-6)
    np.testing.assert_allclose(centroid["sin_phi"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(centroid["cos_phi"], 0.0, atol=1e-6)
    # et follows the same polar angle; the inertia-axis iTheta stays a feature
    np.testing.assert_allclose(centroid["et"], np.sin(np.pi / 4), rtol=1e-6)
    np.testing.assert_allclose(centroid["iTheta"], np.pi / 2)


def make_track_event():
    # track 0 has a state AtCalorimeter (location 4) at (1000, 1000, 500) mm; track 1 has none
    states = [
        (1, 0.0, 0.0, 0.0),
        (2, 30.0, 0.0, 0.0),
        (3, 1400.0, 0.0, 0.0),
        (4, 1000.0, 1000.0, 500.0),
        (1, 0.0, 0.0, 0.0),
        (2, 30.0, 0.0, 0.0),
    ]
    trk = "SiTracks_Refitted"
    n_states = len(states)
    # as uproot reads a collection: per event, one record whose fields are the per-object arrays
    return ak.Array(
        [
            {
                trk: {
                    f"{trk}.type": [0, 0],
                    f"{trk}.chi2": [1.0, 1.0],
                    f"{trk}.ndf": [10, 10],
                    f"{trk}.trackStates_begin": [0, 4],
                    f"{trk}.trackStates_end": [4, 6],
                },
                f"{trk}_dQdx": {f"{trk}_dQdx.dQdx.value": [0.0, 0.0], f"{trk}_dQdx.dQdx.error": [0.0, 0.0]},
                f"_{trk}_trackStates": {
                    f"_{trk}_trackStates.location": [s[0] for s in states],
                    f"_{trk}_trackStates.referencePoint.x": [s[1] for s in states],
                    f"_{trk}_trackStates.referencePoint.y": [s[2] for s in states],
                    f"_{trk}_trackStates.referencePoint.z": [s[3] for s in states],
                    f"_{trk}_trackStates.tanLambda": [0.5] * n_states,
                    f"_{trk}_trackStates.D0": [0.0] * n_states,
                    f"_{trk}_trackStates.phi": [0.3] * n_states,
                    f"_{trk}_trackStates.omega": [1e-4] * n_states,
                    f"_{trk}_trackStates.Z0": [0.0] * n_states,
                    f"_{trk}_trackStates.time": [0.0] * n_states,
                },
            }
        ]
    )


def test_track_ecal_entry_is_the_state_at_the_calorimeter():
    prop_data = make_track_event()

    features = track_to_features(prop_data, 0, 5.0, ecal_entry=True)

    np.testing.assert_allclose(features["eta_ecal"], [np.arcsinh(500.0 / np.hypot(1000.0, 1000.0)), 0.0], rtol=1e-6)
    np.testing.assert_allclose(features["phi_ecal"], [np.pi / 4, 0.0], rtol=1e-6)
    # the vertex direction is unchanged
    np.testing.assert_allclose(features["sin_phi"], np.sin(0.3))
    assert "eta_ecal" not in track_to_features(prop_data, 0, 5.0).fields


@pytest.mark.parametrize(
    ("pdg", "charge", "on_track", "expected"),
    [
        (211, 1, True, (211, False)),
        (11, -1, True, (11, False)),
        (22, 0, False, (22, False)),
        (2112, 0, False, (130, False)),
        # charged particles without a track become neutral, like a conventional PF candidate
        (211, 1, False, (130, True)),
        (13, -1, False, (130, True)),
        (11, 1, False, (22, True)),
        # neutral particles that own a track (conversion, V0) become charged hadrons
        (22, 0, True, (211, True)),
        (310, 0, True, (211, True)),
    ],
)
def test_represented_pid_flags_charge_flips(pdg, charge, on_track, expected):
    assert represented_pid(pdg, charge, on_track, not on_track) == expected


def test_only_maia_changes_track_inputs_cluster_angles_and_target_assignment():
    for name, detector in EDM4HEP.DETECTORS.items():
        if name == "maia":
            assert detector.track_ecal_entry and detector.cluster_angles_from_position
            assert detector.target_assignment == "majority_energy"
            assert detector.track_feature_names()[-2:] == ["eta_ecal", "phi_ecal"]
        else:
            assert not detector.track_ecal_entry and not detector.cluster_angles_from_position, name
            assert detector.target_assignment == "greedy", name
            assert detector.track_feature_names() == EDM4HEP.TrackFeatures.get_names(), name
    # the extra track columns fill the padding of the shared track/cluster layout
    assert len(EDM4HEP.DETECTORS["maia"].track_feature_names()) <= len(EDM4HEP.ClusterFeatures.get_names())


def test_registered_muon_collections_are_read():
    for name, detector in EDM4HEP.DETECTORS.items():
        assert set(detector.muon_collections).issubset(detector.hit_collections), name
    for name in ("clic", "cld", "maia"):
        assert EDM4HEP.DETECTORS[name].muon_collections == ("MUON",)


class FakeTree:
    """The one uproot TTree method append_relation_truth_links uses."""

    def __init__(self, branches):
        self.branches = branches

    def arrays(self, keys):
        return ak.Array({k: self.branches[k] for k in keys})


# A MAIA-like file: CalohitMCTruthLink covers ECAL only; MUON reaches truth through RelationMuonHit ->
# Yoke sim hits -> CaloHitContributions -> MCParticle. Two events, the second empty.
MAIA_LIKE_IDS = {
    "ECALBarrel": 11,
    "MUON": 13,
    "YokeBarrelCollection": 21,
    "YokeEndcapCollection": 22,
    "AllCaloHitContributionsCombined": 30,
    "MCParticle": 40,
    "OtherParticles": 41,
}
MAIA_LIKE_TREE = FakeTree(
    {
        "_RelationMuonHit_from/_RelationMuonHit_from.collectionID": [[13, 13, 13], []],
        "_RelationMuonHit_from/_RelationMuonHit_from.index": [[0, 1, 2], []],
        "_RelationMuonHit_to/_RelationMuonHit_to.collectionID": [[21, 22, 21], []],
        "_RelationMuonHit_to/_RelationMuonHit_to.index": [[0, 0, 1], []],
        "YokeBarrelCollection.contributions_begin": [[0, 2], []],
        "YokeBarrelCollection.contributions_end": [[2, 3], []],
        "_YokeBarrelCollection_contributions/_YokeBarrelCollection_contributions.collectionID": [[30, 30, 30], []],
        "_YokeBarrelCollection_contributions/_YokeBarrelCollection_contributions.index": [[0, 1, 2], []],
        "YokeEndcapCollection.contributions_begin": [[0], []],
        "YokeEndcapCollection.contributions_end": [[1], []],
        "_YokeEndcapCollection_contributions/_YokeEndcapCollection_contributions.collectionID": [[30], []],
        "_YokeEndcapCollection_contributions/_YokeEndcapCollection_contributions.index": [[3], []],
        "AllCaloHitContributionsCombined.energy": [[2e-4, 1e-4, 5e-4, 4e-4], []],
        # the third contribution belongs to another particle collection and must be skipped
        "_AllCaloHitContributionsCombined_particle/_AllCaloHitContributionsCombined_particle.collectionID": [[40, 40, 41, 40], []],
        "_AllCaloHitContributionsCombined_particle/_AllCaloHitContributionsCombined_particle.index": [[5, 7, 9, 5], []],
    }
)


def make_calohit_links(links):
    """links: one list of (weight, particle collection, particle index, hit collection, hit index) per event"""
    return ak.Array({k: [[link[i] for link in event] for event in links] for i, k in enumerate(CALOHIT_LINK_FIELDS)})


def test_relation_truth_links_give_muon_hits_their_contributing_particles():
    ecal_only = make_calohit_links([[(5.0, 40, 0, 11, 0)], []])

    links = append_relation_truth_links(ecal_only, MAIA_LIKE_TREE, "RelationMuonHit", MAIA_LIKE_IDS, MAIA_LIKE_IDS["MCParticle"])

    assert links.fields == list(CALOHIT_LINK_FIELDS)
    rows = list(zip(*(links[k][0].to_list() for k in CALOHIT_LINK_FIELDS)))
    assert rows == [(5.0, 40, 0, 11, 0), (2e-4, 40, 5, 13, 0), (1e-4, 40, 7, 13, 0), (4e-4, 40, 5, 13, 1)]
    assert all(len(links[k][1]) == 0 for k in CALOHIT_LINK_FIELDS)

    # the hit -> particle matrix splits each muon hit's energy by the contributed-energy shares
    hit_data = {"ECALBarrel": make_calo_hits("ECALBarrel", [1.0]), "MUON": make_calo_hits("MUON", [3e-4, 2e-4, 1e-4])}
    _, (gen_idx, hit_idx, weight), _ = get_hit_matrix_and_genadj(hit_data, links, None, 0, MAIA_LIKE_IDS, MAIA_LIKE_IDS["MCParticle"])
    got = {(int(g), int(h)): float(w) for g, h, w in zip(gen_idx, hit_idx, weight)}
    # global hit order: ECALBarrel 0, then MUON 0, 1, 2; MUON 2 has no MCParticle contribution
    assert got.keys() == {(0, 0), (5, 1), (7, 1), (5, 2)}
    np.testing.assert_allclose([got[(5, 1)], got[(7, 1)], got[(5, 2)], got[(0, 0)]], [2e-4, 1e-4, 2e-4, 1.0], rtol=1e-5)


def test_relation_truth_links_refuse_hits_that_calohit_links_already_cover():
    # mixing the two weight conventions on one hit would skew its shares
    muon_already_linked = make_calohit_links([[(5.0, 40, 0, 13, 0)], []])

    with pytest.raises(ValueError, match="already links"):
        append_relation_truth_links(muon_already_linked, MAIA_LIKE_TREE, "RelationMuonHit", MAIA_LIKE_IDS, MAIA_LIKE_IDS["MCParticle"])


def test_only_maia_follows_the_muon_hit_relation():
    # CLIC and CLD already link MUON in CalohitMCTruthLink
    assert EDM4HEP.DETECTORS["maia"].hit_truth_relations == ("RelationMuonHit",)
    for name in ("clic", "cld"):
        assert EDM4HEP.DETECTORS[name].hit_truth_relations == ()


def test_tfds_hit_schema_retains_detector_surface_fields():
    assert EDM4HEP.HitFeatures.get_names()[-3:] == ["system", "side", "layer"]
    assert X_FEATURES[-3:] == ["system", "side", "layer"]
