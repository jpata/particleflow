import numpy as np
import pytest

from mlpf.data.cms.postprocessing_truth import Element, Event, Link, Particle, _nano_particles, four_vector, process_event
from mlpf.data.cms.truth_postprocessing_validation import compare_event, generator_closure, legacy_reference, target_vectors


def particle(index=0, pid=211, pt=10, eta=0, phi=0, charge=1):
    return Particle(index, pid, charge, four_vector(pt, eta, phi, pt * np.cosh(eta)))


def element(kind="Track", index=0, pt=10, phi=0, eta=0):
    return Element(kind, index, four_vector(pt, eta, phi, pt * np.cosh(eta)))


def event(truth, elements, links):
    return Event((1, 1, 1), truth, elements, links, truth, [[p.truth_id] for p in truth])


def test_nano_truth_prefers_cartesian_components_near_phi_pi_over_two():
    reference = np.array([0.00026209248085506953, 20.715230876468723, -0.11901917652134789, 20.71557279342274])
    pt = np.hypot(*reference[:2])
    fields = dict(pdgId=211, charge=1, isSignal=True, pt=pt, eta=np.arcsinh(reference[2] / pt), phi=np.arctan2(reference[1], reference[0]))
    fields.update(zip(("px", "py", "pz", "energy"), reference))
    arrays = {f"TruthBranch_{key}": np.array([[value]], dtype=np.float32) for key, value in fields.items()}
    particle = _nano_particles(arrays, 0, "TruthBranch", [42])[0]
    np.testing.assert_allclose(particle.p4, reference, rtol=1e-5, atol=1e-6)
    assert particle.truth_id == 42
    angular = {key: value for key, value in arrays.items() if key not in ("TruthBranch_px", "TruthBranch_py", "TruthBranch_pz")}
    fallback = _nano_particles(angular, 0, "TruthBranch", [42])[0]
    assert not np.allclose(fallback.p4, reference, rtol=1e-5, atol=1e-6)


def test_track_and_two_clusters_produce_one_whole_particle():
    ev = event([particle()], [element(), element("ECAL"), element("HCAL")], [Link(0, 0, 10), Link(0, 1, 20), Link(0, 2, 30)])
    result = process_event(ev)
    assert result["target_id"].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(result["target_p4"][0], ev.truth[0].p4)
    assert result["provenance"].tolist() == [(0, 0, 1.0)]


def test_track_split_preserves_vector_and_parent_but_has_distinct_target_ids():
    ev = event([particle()], [element(pt=6), element(index=1, pt=4)], [Link(0, 0, 3), Link(0, 1, 1)])
    result = process_event(ev)
    np.testing.assert_allclose(result["target_p4"], [0.75 * ev.truth[0].p4, 0.25 * ev.truth[0].p4])
    assert result["provenance"].tolist() == [(0, 0, 0.75), (1, 0, 0.25)]
    assert result["target_id"].tolist() == [1, 2]
    assert result["split_truth"] == [0]
    reference, _ = legacy_reference(ev)
    np.testing.assert_allclose(result["target_p4"], target_vectors(reference), atol=1e-6)


def test_electron_prefers_gsf_and_is_not_split_across_ordinary_tracks():
    ev = event([particle(pid=11, charge=-1)], [element(), element(index=1), element("GSF")], [Link(0, 0, 10), Link(0, 1, 9), Link(0, 2, 8)])
    result = process_event(ev)
    assert result["target_id"].tolist() == [0, 0, 3]
    assert result["split_truth"] == []


def test_neutral_calorimeter_fragmentation_does_not_duplicate_energy():
    ev = event([particle(pid=22, charge=0)], [element("ECAL", pt=7), element("ECAL", index=1, pt=3)], [Link(0, 0, 7), Link(0, 1, 3)])
    result = process_event(ev)
    assert np.count_nonzero(result["target_id"]) == 1
    np.testing.assert_allclose(result["target_p4"].sum(axis=0), ev.truth[0].p4)


def test_merge_preserves_both_parents_and_uses_consistent_dominant_pid_charge():
    truth = [particle(pid=22, charge=0, pt=10), particle(1, pid=211, charge=1, pt=5, phi=0.1)]
    ev = event(truth, [element("ECAL")], [Link(0, 0, 1), Link(1, 0, 9)])
    result = process_event(ev)
    np.testing.assert_allclose(result["target_p4"][0], sum((p.p4 for p in truth)))
    assert result["provenance"].tolist() == [(0, 0, 1.0), (0, 1, 1.0)]
    assert result["ytarget"]["pid"][0] == 22
    assert result["ytarget"]["charge"][0] == 0
    assert result["mixed"].tolist() == [True]
    assert result["merged_rows"] == [0]
    comparison, _, _ = compare_event(ev, result)
    assert comparison["same_legacy_p4"] and comparison["same_legacy_pid"]
    assert not comparison["same_legacy_charge"]
    assert comparison["legacy_charge_convention_rows"] == [dict(row=0, energy_parent=0, strongest_edge_parents=[1])]
    assert comparison["unexpected_legacy_mismatch_rows"] == []


def test_split_and_merge_in_same_event_preserve_every_original_vector():
    truth = [particle(pt=10), particle(1, pt=5)]
    ev = event(truth, [element(pt=9), element(index=1, pt=6)], [Link(0, 0, 6), Link(0, 1, 4), Link(1, 0, 3), Link(1, 1, 2)])
    result = process_event(ev)
    assert result["split_truth"] == [0, 1]
    assert result["merged_rows"] == [0, 1]
    np.testing.assert_allclose(result["target_p4"].sum(axis=0), sum((p.p4 for p in truth)))
    np.testing.assert_allclose(np.bincount(result["provenance"]["truth"], weights=result["provenance"]["fraction"]), [1, 1])


def test_orphan_truth_is_explicit_and_weak_tracks_do_not_force_assignment():
    ev = event([particle()], [element()], [Link(0, 0, 1, purity=0.75)])
    result = process_event(ev)
    assert result["unmatched_truth"] == [0]
    assert result["target_id"].tolist() == [0]
    assert result["truth"][0].truth_id == 0
    assert len(result["provenance"]) == 0


def test_split_delta_phi_wraparound_and_edge_order_are_stable():
    ev = event([particle(phi=np.pi - 0.01)], [element(phi=-np.pi + 0.01), element(index=1, phi=np.pi - 0.02)], [Link(0, 0, 2), Link(0, 1, 1)])
    first = process_event(ev)
    ev.links.reverse()
    second = process_event(ev)
    assert first["split_truth"] == [0]
    np.testing.assert_array_equal(first["target_p4"], second["target_p4"])
    np.testing.assert_array_equal(first["provenance"], second["provenance"])


def test_empty_event_and_bad_associations():
    result = process_event(event([], [], []))
    assert result["target_p4"].shape == (0, 4)
    for link in [Link(0, 1, 1), Link(0, 0, np.nan), Link(0, 0, -1)]:
        with pytest.raises(ValueError):
            process_event(event([particle()], [element()], [link]))


def test_duplicate_truth_or_element_identity_is_rejected():
    with pytest.raises(ValueError, match="Truth particle IDs"):
        process_event(event([particle(), particle()], [element()], []))
    with pytest.raises(ValueError, match="Input collection/index"):
        process_event(event([particle()], [element(), element()], []))


def test_legacy_reference_agrees_on_mixed_charged_and_neutral_event():
    truth = [particle(), particle(1, pid=22, pt=7, charge=0, phi=1)]
    ev = event(truth, [element(), element("ECAL", phi=1, pt=6), element("HCAL", pt=2)], [Link(0, 0, 10), Link(0, 2, 2), Link(1, 1, 6)])
    result = process_event(ev)
    reference, _ = legacy_reference(ev)
    np.testing.assert_allclose(result["target_p4"], target_vectors(reference), atol=1e-6)
    np.testing.assert_array_equal(result["ytarget"]["pid"], reference["pid"])
    np.testing.assert_array_equal(result["ytarget"]["charge"], reference["charge"])


def test_gen_closure_distinguishes_unsupported_roots_after_split_and_merge():
    ev = Event(
        (1, 1, 1),
        [particle(pt=6), particle(1, pt=3), particle(2, pt=1)],
        [element(pt=5), element(index=1, pt=4)],
        [Link(0, 0, 1), Link(0, 1, 1), Link(1, 0, 1)],
        [particle(42, pt=10)],
        [[42], [42], [42]],
    )
    result = process_event(ev)
    rows, ambiguous = generator_closure(result)
    assert result["split_truth"] == [0]
    assert result["merged_rows"] == [0]
    assert result["unmatched_truth"] == [2]
    assert ambiguous == []
    assert len(rows) == 1
    assert rows[0]["energy_ratio"] == pytest.approx(0.9)
    assert rows[0]["original_root_energy_ratio"] == pytest.approx(1.0)


def test_transverse_scalar_sum_changes_under_merging_while_vector_sum_closes():
    from scripts.cmssw.validate_transverse_activity import event_metrics

    ev = event([particle(pt=10), particle(1, pt=10, phi=np.pi / 2)], [element("ECAL")], [Link(0, 0, 1), Link(1, 0, 1)])
    metrics = event_metrics(process_event(ev))
    assert metrics["gen_visible"]["scalar_sum_pt"] == pytest.approx(20)
    assert metrics["targets"]["scalar_sum_pt"] == pytest.approx(np.sqrt(200))
    assert metrics["represented_vector_residual_GeV"] == pytest.approx(0)
    assert metrics["merge_scalar_pt_reduction_GeV"] == pytest.approx(20 - np.sqrt(200))
    assert metrics["ancestry_fiducial_pt_ratio"] == pytest.approx(1)


def test_transverse_visible_reference_excludes_neutrinos():
    from scripts.cmssw.validate_transverse_activity import event_metrics

    ev = event([particle(pt=10)], [element()], [Link(0, 0, 1)])
    ev.gen = list(ev.gen)
    ev.gen.append(particle(9, pid=14, pt=10, phi=np.pi, charge=0))
    metrics = event_metrics(process_event(ev))
    assert metrics["gen_visible"]["scalar_sum_pt"] == pytest.approx(10)
    assert metrics["gen_all"]["scalar_sum_pt"] == pytest.approx(20)
    assert metrics["gen_all"]["vector_sum_pt"] == pytest.approx(0, abs=1e-12)
    assert metrics["gen_visible"]["vector_sum_pt"] == pytest.approx(10)


def test_transverse_missing_ancestry_keeps_global_target_vector_and_flags_it():
    from scripts.cmssw.validate_transverse_activity import event_metrics

    ev = event([particle(pt=10)], [element()], [Link(0, 0, 1)])
    ev.gen_ancestors = [[]]
    metrics = event_metrics(process_event(ev))
    assert metrics["targets"]["scalar_sum_pt"] == pytest.approx(10)
    assert metrics["represented_vector_residual_GeV"] == pytest.approx(0)
    assert metrics["unattributed_truth_rows"] == [0]
    assert metrics["unattributed_p4"][3] == pytest.approx(10)
    assert metrics["ancestry_fiducial_pt_ratio"] == pytest.approx(0)
