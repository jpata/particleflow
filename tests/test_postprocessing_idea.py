"""Focused tests for IDEA truth-link postprocessing."""

import numpy as np
import awkward as ak
import pytest

from mlpf.data.key4hep import postprocessing as pp


def test_cluster_energy_uses_stored_truth_fractions_without_renormalizing():
    # Row 0 is a status-1 ancestor carrying a propagated copy of row 1's link.
    # The column consequently sums to more than one by construction.  Each row
    # must retain its own 25% contribution; column normalization would wrongly
    # reduce both contributions to 10 GeV.
    weights = np.array([[0.25], [0.25]])
    cluster_energy = np.array([40.0])

    attributed_energy = pp.idea_cluster_energy_by_genparticle(weights, cluster_energy)

    np.testing.assert_allclose(attributed_energy, [10.0, 10.0])


def test_cluster_energy_sums_fragments_for_each_particle():
    weights = np.array([[0.50, 0.25], [0.25, 0.75]])
    cluster_energy = np.array([20.0, 40.0])

    attributed_energy = pp.idea_cluster_energy_by_genparticle(weights, cluster_energy)

    np.testing.assert_allclose(attributed_energy, [20.0, 35.0])


def test_dual_readout_cluster_shape_parameters_are_split_by_channel():
    clusters = ak.Array(
        {
            "TopoClusterAll.shapeParameters_begin": [0, 3],
            "TopoClusterAll.shapeParameters_end": [3, 6],
        }
    )
    prop_data = {"_TopoClusterAll_shapeParameters": [ak.Array([0.1, 2.0, 3.0, 0.2, 5.0, 7.0])]}

    cherenkov, scintillation = pp._idea_dual_readout_cluster_energies(prop_data, clusters, 0)

    np.testing.assert_allclose(cherenkov, [2.0, 5.0])
    np.testing.assert_allclose(scintillation, [3.0, 7.0])


def test_idea_clusters_fill_every_schema_column_with_no_muon_hits():
    clusters = ak.Array(
        {
            "TopoClusterAll.energy": [4.0, 6.0],
            "TopoClusterAll.position.x": [100.0, 0.0],
            "TopoClusterAll.position.y": [0.0, 100.0],
            "TopoClusterAll.position.z": [10.0, -10.0],
            "TopoClusterAll.hits_begin": [0, 3],
            "TopoClusterAll.hits_end": [3, 5],
            "TopoClusterAll.shapeParameters_begin": [0, 3],
            "TopoClusterAll.shapeParameters_end": [3, 6],
        }
    )
    cells = ak.Array(
        {
            "TopoClusterAllCells.energy": [1.0, 1.0, 2.0, 3.0, 3.0],
            "TopoClusterAllCells.position.x": [100.0, 100.0, 100.0, 0.0, 0.0],
            "TopoClusterAllCells.position.y": [0.0, 0.0, 0.0, 100.0, 100.0],
            "TopoClusterAllCells.position.z": [10.0, 10.0, 10.0, -10.0, -10.0],
        }
    )
    prop_data = {
        "TopoClusterAll": [clusters],
        "TopoClusterAllCells": [cells],
        "_TopoClusterAll_hits/_TopoClusterAll_hits.index": [ak.Array([0, 1, 2, 3, 4])],
        "_TopoClusterAll_hits/_TopoClusterAll_hits.collectionID": [ak.Array([42] * 5)],
        "_TopoClusterAll_shapeParameters": [ak.Array([0.1, 2.0, 3.0, 0.2, 5.0, 7.0])],
    }

    features = pp.idea_cluster_to_features(prop_data, 0, cell_collection_id=42)

    assert set(pp.cluster_feature_order).issubset(features.fields)
    np.testing.assert_array_equal(features["num_muon_hits"], [0, 0])
    np.testing.assert_array_equal(features["energy_muon"], [0.0, 0.0])


def test_legacy_idea_clusters_leave_channel_energies_empty():
    clusters = ak.Array(
        {
            "TopoClusterAll.shapeParameters_begin": [0, 1],
            "TopoClusterAll.shapeParameters_end": [1, 2],
        }
    )
    prop_data = {"_TopoClusterAll_shapeParameters": [ak.Array([0.1, 0.2])]}

    cherenkov, scintillation = pp._idea_dual_readout_cluster_energies(prop_data, clusters, 0)

    np.testing.assert_array_equal(cherenkov, [0.0, 0.0])
    np.testing.assert_array_equal(scintillation, [0.0, 0.0])


def test_idea_cluster_widths_use_associated_cell_energy_and_positions():
    clusters = ak.Array(
        {
            "TopoClusterAll.energy": [4.0, 2.0],
            "TopoClusterAll.position.x": [3.0, 10.0],
            "TopoClusterAll.position.y": [0.0, 0.0],
            "TopoClusterAll.position.z": [0.0, 0.0],
            "TopoClusterAll.hits_begin": [0, 2],
            "TopoClusterAll.hits_end": [2, 3],
            "TopoClusterAll.shapeParameters_begin": [0, 3],
            "TopoClusterAll.shapeParameters_end": [3, 6],
        }
    )
    cells = ak.Array(
        {
            "TopoClusterAllCells.energy": [1.0, 3.0, 2.0],
            "TopoClusterAllCells.position.x": [0.0, 4.0, 10.0],
            "TopoClusterAllCells.position.y": [0.0, 0.0, 0.0],
            "TopoClusterAllCells.position.z": [0.0, 0.0, 0.0],
        }
    )
    prop_data = {
        "TopoClusterAll": [clusters],
        "TopoClusterAllCells": [cells],
        "_TopoClusterAll_hits/_TopoClusterAll_hits.index": [ak.Array([0, 1, 2])],
        "_TopoClusterAll_hits/_TopoClusterAll_hits.collectionID": [ak.Array([42, 42, 42])],
        "_TopoClusterAll_shapeParameters": [ak.Array([0.0, 2.0, 2.0, 0.0, 1.0, 1.0])],
    }

    features = pp.idea_cluster_to_features(prop_data, 0, cell_collection_id=42)

    np.testing.assert_allclose(features["sigma_x"], [np.sqrt(3.0), 0.0])
    np.testing.assert_array_equal(features["sigma_y"], [0.0, 0.0])
    np.testing.assert_array_equal(features["sigma_z"], [0.0, 0.0])
    with pytest.raises(ValueError, match="do not all refer to TopoClusterAllCells"):
        pp.idea_cluster_to_features(prop_data, 0, cell_collection_id=43)


def _idea_hit_fixture():
    from mlpf.data.key4hep.idea_full import CHANNELS, LINK

    data = {}
    ids = {"MCParticles": 99}
    for collection, energies, cid in zip(CHANNELS, ([4.0, 0.0, 6.0], [3.0, 1.0, 2.0]), (10, 11)):
        ids[collection] = cid
        n = len(energies)
        values = dict(energy=energies, time=[1.0] * n)
        values.update({f"position.{axis}": [10.0 if axis == "x" else 0.0] * n for axis in "xyz"})
        data[collection] = ak.Array({f"{collection}.{name}": [value] for name, value in values.items()})
    data[f"{LINK}.weight"] = [[0.5, 1.0, 0.25, 1.0]]
    for end, indices, collections in (("from", [0, 2, 0, 1], [10, 10, 11, 11]), ("to", [0, 1, 1, 2], [99] * 4)):
        name = f"_{LINK}_{end}"
        data[f"{name}/{name}.index"] = [indices]
        data[f"{name}/{name}.collectionID"] = [collections]
    return data, ids


def test_idea_hit_channels_use_collection_identity_and_stored_fractions():
    from mlpf.data.key4hep.idea_full import hit_inputs_and_links

    data, ids = _idea_hit_fixture()
    x, links = hit_inputs_and_links(data, 0, ids, 3)
    assert x[:, 5].tolist() == [4, 6, 3, 1, 2]  # Zero-signal cells removed; unlinked positive cell kept.
    assert x[:, 11].tolist() == [1, 1, 2, 2, 2]
    np.testing.assert_array_equal(links[0], [0, 1, 1, 2])
    np.testing.assert_array_equal(links[1], [0, 1, 2, 3])
    np.testing.assert_allclose(links[2], [2, 6, 0.75, 1])  # Do not renormalize a 25% response to 100%.
    assert not x[:, 12:15].any()


@pytest.mark.parametrize("field,value", [("collectionID", 555), ("index", 50)])
def test_idea_hit_links_reject_invalid_object_ids(field, value):
    from mlpf.data.key4hep.idea_full import hit_inputs_and_links, LINK

    data, ids = _idea_hit_fixture()
    data[f"_{LINK}_from/_{LINK}_from.{field}"][0][0] = value
    with pytest.raises(ValueError, match="IDEA hit truth link"):
        hit_inputs_and_links(data, 0, ids, 3)


def test_idea_tracker_hits_follow_digi_sim_mc_links():
    from mlpf.data.key4hep.idea_full import TRACKER_DIGIS, tracker_hit_features

    data = {}
    for digi, (sim, link, system) in TRACKER_DIGIS.items():
        n = 2 if digi == "VTXBDigis" else 0
        data[f"{digi}/{digi}.eDep"] = [[1e-5] * n]
        data[f"{digi}/{digi}.time"] = [[0.1] * n]
        data[f"{digi}/{digi}.position.x"] = [[20.0, 40.0][:n]]
        data[f"{digi}/{digi}.position.y"] = [[0.0] * n]
        data[f"{digi}/{digi}.position.z"] = [[5.0] * n]
        # digi 0 -> sim 1 -> MC 7 (quality bit 30 set); digi 1 -> sim 0 -> MC 3
        data[f"_{link}_from/_{link}_from.index"] = [[0, 1][:n]]
        data[f"_{link}_to/_{link}_to.index"] = [[1, 0][:n]]
        data[f"_{sim}_particle/_{sim}_particle.index"] = [[3, 7][:n]]
        data[f"_{sim}_particle/_{sim}_particle.collectionID"] = [[99, 99][:n]]
        data[f"{sim}/{sim}.quality"] = [[0, 1 << 30][:n]]
        for axis, values in zip("xyz", ([1.0, -2.0], [0.0, 0.0], [0.5, 0.5])):
            data[f"{sim}/{sim}.momentum.{axis}"] = [values[:n]]
    x, producer, momentum, unsaved = tracker_hit_features(data, 0, 99)
    assert x.shape == (2, 15) and np.all(x[:, 0] == 1) and np.all(x[:, 10] == 3) and np.all(x[:, 12] == 1)
    assert producer.tolist() == [7, 3]
    np.testing.assert_allclose(momentum[:, 0], [-2.0, 1.0])
    assert unsaved.tolist() == [True, False]
