"""Detector-specific ownership units and CMS fallback remain explicit."""

import copy
import contextlib
import io

import awkward as ak
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from mlpf.data.cms import postprocessing2 as cms
from mlpf.data.target_building import match_majority_tracks_then_energy
from scripts.studies.validate_detector_matching import clean_scores, shared_bundle
from tests.test_postprocessing_merge import make_eventdata


def cms_event(kind=1):
    graph = nx.DiGraph()
    values = {k: 0.0 for k in cms.elem_branches}
    values.update(typ=kind, pt=5.0, eta=0.1, phi=0.2, energy=5.1)
    graph.add_node(("elem", 0), **values)
    particle = {k: 0.0 for k in cms.particle_feature_order}
    particle.update(pid=11 if kind == 6 else 211, charge=1.0, pt=5.0, eta=0.1, phi=0.2, energy=5.1, particle_number=1)
    graph.add_node(("cp", 0), **particle)
    graph.add_edge(("cp", 0), ("elem", 0), weight=5.1)
    return graph


def test_cms_explicit_native_assignment_reproduces_default():
    def native(graph, particles, elements):
        forward, reverse = {}, {}
        for kind in [6, 1, 4, 5, 8, 9, 10, 11]:
            cms.find_representative_elements(graph, forward, reverse, kind)
        return forward, reverse

    with contextlib.redirect_stdout(io.StringIO()):
        expected = cms.prepare_normalized_table(cms_event(), 0)
        actual = cms.prepare_normalized_table(cms_event(), 0, assignment_fn=native)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_cms_unmatched_choice_uses_native_fallback_without_key_error():
    with contextlib.redirect_stdout(io.StringIO()):
        x, _, y = cms.prepare_normalized_table(cms_event(), 0, assignment_fn=lambda *args: ({}, {}))
    assert y["pid"].tolist() == [211]
    np.testing.assert_allclose(y["energy"], [5.1])


def test_cms_gsf_restriction_precedes_assignment():
    graph = cms_event(6)
    other = copy.deepcopy(graph.nodes[("cp", 0)])
    other["pid"] = 211
    graph.add_node(("cp", 1), **other)
    graph.add_edge(("cp", 1), ("elem", 0), weight=5.1)

    def choose(graph, particles, elements):
        assert list(graph.predecessors(("elem", 0))) == [("cp", 0)]
        return {("elem", 0): ("cp", 0)}, {("cp", 0): ("elem", 0)}

    with contextlib.redirect_stdout(io.StringIO()):
        _, _, y = cms.prepare_normalized_table(graph, 0, assignment_fn=choose)
    assert y["pid"].tolist() == [11]


def test_overcommitted_tracks_are_excluded_not_renormalized():
    bundle = {"track": csr_matrix([[1.0], [1.0]]), "cluster": csr_matrix((2, 0)), "signal": np.empty(0)}
    track, _ = clean_scores(bundle)
    assert track.nnz == 0
    assert bundle["invalid_track_columns"] == 1
    np.testing.assert_array_equal(bundle["track"].toarray(), [[1.0], [1.0]])


def test_key4hep_ownership_uses_hit_energy_instead_of_corrected_cluster_energy():
    gd = make_eventdata([2.0], [1.2])
    gd.hit_features = ak.Record({"type": [0], "energy": [1.5]})
    gd.track_features = ak.Record({"type": [], "eta": [], "phi": []})
    gd.cluster_features = ak.Record({"type": [0], "energy": [1.0], "eta": [0.0], "phi": [0.0]})
    bundle = shared_bundle(gd, "cld", 0, np.empty((0, 4)))
    np.testing.assert_allclose(bundle["signal"], [1.5])
    result = match_majority_tracks_then_energy(bundle["track"], bundle["cluster"], [2.0])
    assert result.tolist() == [[-1, 0]]
