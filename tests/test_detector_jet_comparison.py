import numpy as np

from scripts.compare_detector_features import binned_fraction, collect, jet_matching_values


def test_jet_comparison_wraps_phi_without_response_cut():
    gen = np.array([[10.0, 0.0, np.pi - 0.01, 12.0], [20.0, 2.0, 0.0, 30.0]])
    target = np.array([[30.0, 0.0, -np.pi + 0.01, 35.0]])
    values = jet_matching_values(gen, target, 0.1)
    np.testing.assert_array_equal(values["gen_matched"], [True, False])
    np.testing.assert_array_equal(values["target_matched"], [True])
    np.testing.assert_allclose(values["pt_ratio"], [3.0])
    np.testing.assert_array_equal(values["matched_gen_pt"], [10.0])


def test_jet_comparison_keeps_unmatched_denominators_and_unique_pairs():
    gen = np.array([[10.0, 0.0, 0.0, 10.0], [20.0, 0.01, 0.0, 20.0]])
    target = np.array([[15.0, 0.0, 0.0, 15.0]])
    values = jet_matching_values(gen, target, 0.1)
    assert len(values["gen_pt"]) == 2
    assert values["gen_matched"].sum() == 1
    assert values["target_matched"].sum() == 1
    empty = jet_matching_values(gen, np.empty((0, 4)), 0.1)
    assert not empty["gen_matched"].any()
    assert not len(empty["pt_ratio"])


def test_matching_fraction_includes_unmatched_and_handles_empty_bins():
    fraction, lower, upper, passed, total = binned_fraction(
        np.array([5.0, 6.0, 15.0, 16.0]), np.array([True, False, True, True]), np.array([0.0, 10.0, 20.0, 30.0])
    )
    np.testing.assert_array_equal(passed, [1, 2, 0])
    np.testing.assert_array_equal(total, [2, 2, 0])
    np.testing.assert_allclose(fraction[:2], [0.5, 1.0])
    assert lower[0] < 0.5 < upper[0]
    assert lower[1] < 1.0 and np.isclose(upper[1], 1.0)
    assert np.isnan(fraction[2]) and np.isnan(lower[2]) and np.isnan(upper[2])


def test_comparison_uses_updated_colliderml_track_columns():
    x = np.zeros((1, 17), dtype=np.float32)
    x[0, :6] = [1.0, 2.0, 0.5, 0.0, 1.0, 2 * np.cosh(0.5)]
    x[0, 10:14] = [np.sinh(0.5), 0.00045, 50.0, 8.0]
    event = {"X": x, "ytarget": np.zeros((1, 14)), "genmet": 0.0, "genjets": np.empty((0, 4)), "targetjets": np.empty((0, 4))}
    values, _ = collect([event], "colliderml")
    np.testing.assert_allclose(values["track/tanLambda"], [np.sinh(0.5)])
    np.testing.assert_allclose(values["track/omega"], [0.00045])
    np.testing.assert_allclose(values["track/radiusOfInnermostHit"], [50.0])
    np.testing.assert_allclose(values["diagnostic/tanLambda_minus_sinh_eta"], [0.0], atol=1e-7)
    assert "track/tanLambda_slot" not in values
