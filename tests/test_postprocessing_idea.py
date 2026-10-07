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
