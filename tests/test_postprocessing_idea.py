"""Focused tests for IDEA truth-link postprocessing."""

import numpy as np
import awkward as ak

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
