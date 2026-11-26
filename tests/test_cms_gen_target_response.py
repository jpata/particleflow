import numpy as np
import pytest

from mlpf.data.cms.gen_target_response import ResponseConfig, analyze_event, bootstrap_stats, binned_stats, cluster_jets, met_quantities, robust_stats
from mlpf.data.cms.postprocessing_truth import Particle, four_vector


def make_result(scale=1):
    vectors = np.array([four_vector(60, 0.3, 0, 60 * np.cosh(0.3)), four_vector(30, -0.5, 1.7, 30 * np.cosh(0.5))])
    return dict(
        event_id=(1, 1, 1),
        gen=[Particle(i, 211, 1, p) for i, p in enumerate(vectors)] + [Particle(9, 14, 0, four_vector(20, 0, 2, 20))],
        target_p4=scale * vectors,
        target_id=np.array([1, 2]),
        ytarget=np.array([(211.0,), (211.0,)], dtype=[("pid", "f4")]),
    )


def test_identical_visible_particles_give_unit_jet_and_met_response():
    event = analyze_event(make_result(), 0, ResponseConfig())
    assert len(event["jet_matches"]) == 2
    assert event["gen_particles"] == 2
    assert all(pair["response"] == pytest.approx(1) for pair in event["jet_matches"])
    assert event["met"]["magnitude_response"] == pytest.approx(1)
    assert event["met"]["projected_response"] == pytest.approx(1)
    assert event["met"]["delta_parallel"] == pytest.approx(0)
    assert event["met"]["delta_perp"] == pytest.approx(0)


def test_uniform_momentum_scale_has_known_jet_and_met_response():
    event = analyze_event(make_result(0.9), 0, ResponseConfig())
    assert all(pair["response"] == pytest.approx(0.9) for pair in event["jet_matches"])
    assert event["met"]["magnitude_response"] == pytest.approx(0.9)
    assert event["met"]["projected_response"] == pytest.approx(0.9)
    assert event["met"]["delta_parallel"] == pytest.approx(-0.1 * event["met"]["gen_met"])
    assert event["met"]["delta_perp"] == pytest.approx(0)


def test_low_target_jet_threshold_does_not_hide_low_response_matches():
    event = analyze_event(make_result(0.5), 0, ResponseConfig())
    assert len(event["jet_matches"]) == 2
    assert any(p["target_pt"] < 20 for p in event["jet_matches"])
    assert all(p["response"] == pytest.approx(0.5) for p in event["jet_matches"])


def test_one_target_jet_is_not_reused_by_two_gen_jets():
    config = ResponseConfig(match_dr=0.25)
    result = make_result()
    result["gen"] = [Particle(i, 211, 1, four_vector(30, 0, phi, 30)) for i, phi in enumerate((-0.205, 0.205))]
    result["target_p4"] = np.array([four_vector(30, 0, 0, 30)])
    result["target_id"] = np.array([1])
    result["ytarget"] = result["ytarget"][:1]
    event = analyze_event(result, 0, config)
    assert len(event["gen_jets"]) == 2
    assert len(event["jet_matches"]) == 1
    assert sum(j["matched"] for j in event["gen_jets"]) == 1


def test_zero_reference_met_has_no_response_ratio_and_retains_residuals():
    gen = np.array([[10, 0, 0, 10], [-10, 0, 0, 10]])
    target = np.array([[9, 0, 0, 9], [-10, 0, 0, 10]])
    row = met_quantities(gen, target)
    assert row["gen_met"] == 0
    assert row["target_met"] == 1
    assert row["magnitude_response"] is None
    assert row["projected_response"] is None
    assert row["delta_x"] == 1


def test_met_projection_is_signed_and_separates_angular_error():
    row = met_quantities(np.array([[-10, 0, 0, 10]]), np.array([[0, -10, 0, 10]]))
    assert row["magnitude_response"] == 1
    assert row["projected_response"] == 0
    assert row["delta_parallel"] == -10
    assert row["delta_perp"] == 10
    assert row["delta_phi"] == pytest.approx(np.pi / 2)


def test_empty_targets_are_unmatched_instead_of_dropped_from_efficiency():
    result = make_result()
    result["target_p4"] = np.empty((0, 4))
    result["target_id"] = np.empty(0, dtype=int)
    result["ytarget"] = result["ytarget"][:0]
    event = analyze_event(result, 0, ResponseConfig())
    assert len(event["gen_jets"]) == 2
    assert len(event["jet_matches"]) == 0
    assert all(not j["matched"] for j in event["gen_jets"])
    assert event["met"]["magnitude_response"] == 0


def test_jet_clustering_wraps_phi_and_preserves_four_vector():
    particles = np.array([four_vector(25, 0, np.pi - 0.01, 25), four_vector(25, 0, -np.pi + 0.01, 25)])
    jets = cluster_jets(particles, 5, ResponseConfig())
    assert len(jets) == 1
    np.testing.assert_allclose(jets[0], particles.sum(axis=0))


def test_fiducial_cuts_apply_independently_and_exclude_neutrinos():
    result = make_result()
    result["gen"] += [Particle(10, 22, 0, four_vector(5, 3.3, 1, 5 * np.cosh(3.3))), Particle(11, 22, 0, four_vector(0.2, 0, 1, 0.2))]
    event = analyze_event(result, 0, ResponseConfig())
    assert event["gen_particles"] == 2


def test_robust_resolution_definition_and_zero_median():
    stat = robust_stats([0, 1, 2, 3, 4])
    assert stat["median"] == 2
    assert stat["sigma68"] == pytest.approx(1.36)
    assert stat["relative_sigma68"] == pytest.approx(0.68)
    assert robust_stats([0, 0])["relative_sigma68"] is None
    assert robust_stats([])["median"] is None


def test_event_bootstrap_is_deterministic_and_counts_clusters():
    values = [0.9, 1.1, 1.0, 1.2]
    groups = [0, 0, 1, 1]
    a = bootstrap_stats(values, groups, 50)
    assert a == bootstrap_stats(values, groups, 50)
    assert a["independent_events"] == 2
    assert "median" in a["intervals"]


def test_low_statistics_bins_are_reported_but_not_plotted():
    rows = [dict(event_index=0, x=25, response=1), dict(event_index=1, x=40, response=0.9)]
    bins = binned_stats(rows, "x", "response", [20, 30, 50, 80], replicas=0, min_count=2)
    assert [b["n"] for b in bins] == [1, 1, 0]
    assert not any(b["plot"] for b in bins)


def test_jet_axis_acceptance_and_threshold_config_are_checked():
    with pytest.raises(ValueError):
        ResponseConfig(jet_eta_max=3.3)
    with pytest.raises(ValueError):
        ResponseConfig(targetjet_pt_min=30)


def test_full_hgcal_forward_band_and_eta_endpoint_are_included():
    result = make_result()
    vectors = np.array([four_vector(40, eta, phi, 40*np.cosh(eta)) for eta, phi in [(3.1,0),(-3.2,2)]])
    result['gen'] = [Particle(i, 211, 1, p) for i,p in enumerate(vectors)]
    result['target_p4'] = vectors.copy()
    event = analyze_event(result,0,ResponseConfig())
    assert event['gen_particles'] == 2
    assert event['target_particles'] == 2
    assert len(event['jet_matches']) == 2
    assert all(p['response'] == pytest.approx(1) for p in event['jet_matches'])
