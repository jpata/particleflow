# Unit tests for the ColliderML track adapter (ACTS perigee -> MLPF features).
import numpy as np

from mlpf.data.colliderml.tracks import B_FIELD, track_features_cml
from mlpf.data.key4hep.postprocessing import track_pt


def _tracker_hits(n=20):
    # n hits on a unit-spaced x grid (1, 2, ..., n) in the y=0 plane
    return {
        "x": np.arange(1, n + 1, dtype=np.float64),
        "y": np.zeros(n, dtype=np.float64),
    }


def _single_track(d0=0.01, z0=0.02, phi=0.3, theta=1.0, qop=0.5, nmeas=10):
    return {
        "d0": np.array([d0]),
        "z0": np.array([z0]),
        "phi": np.array([phi]),
        "theta": np.array([theta]),
        "qop": np.array([qop]),
        "hit_ids": [[i for i in range(nmeas)]],
    }


def _args(tracks, tracker=None):
    return (tracks, tracker if tracker is not None else _tracker_hits())


def test_track_pt_eta_invariant_mass():
    tr = track_features_cml(*_args(_single_track(qop=0.25, theta=np.pi / 2)))  # theta=pi/2 -> eta=0
    assert np.isclose(tr["pt"][0], 4.0, atol=1e-6), f"pt={tr['pt']}"
    assert np.isclose(tr["eta"][0], 0.0, atol=1e-6), f"eta={tr['eta']}"
    # p = |1/qop|
    assert np.isclose(tr["p"][0], 4.0, atol=1e-6)
    # energy-like scale should be positive
    assert tr["p"][0] > 0


def test_track_pt_decreases_with_qop():
    f1 = track_features_cml(*_args(_single_track(qop=0.5)))
    f2 = track_features_cml(*_args(_single_track(qop=0.25)))
    assert f2["pt"][0] > f1["pt"][0]


def test_track_zero_qop_safe():
    f = track_features_cml(*_args(_single_track(qop=0.0)))
    # we protect with a small epsilon rather than crash; result should not be inf/nan
    assert np.isfinite(f["pt"][0]) and np.isfinite(f["eta"][0])


def test_n_meas_matches_hit_count():
    f = track_features_cml(*_args(_single_track(nmeas=7)))
    assert f["n_meas"][0] == 7


def test_eta_symmetry():
    forward = track_features_cml(*_args(_single_track(theta=np.pi / 4)))
    backward = track_features_cml(*_args(_single_track(theta=3 * np.pi / 4)))
    assert np.isclose(forward["eta"][0], -backward["eta"][0], atol=1e-6)


def test_tanlambda_matches_theta_and_key4hep_relation():
    tr = track_features_cml(*_args(_single_track(theta=np.pi / 4, qop=0.25)))
    # tanLambda = 1/tan(theta) = sinh(eta), and key4hep recovers eta = arcsinh(tanLambda)
    assert np.isclose(tr["tanLambda"][0], 1.0 / np.tan(np.pi / 4), atol=1e-6)
    assert np.isclose(np.arcsinh(tr["tanLambda"][0]), tr["eta"][0], atol=1e-6)
    backward = track_features_cml(*_args(_single_track(theta=3 * np.pi / 4, qop=0.25)))
    assert backward["tanLambda"][0] < 0  # backwards track -> negative tanLambda, like EDM4hep


def test_omega_roundtrips_key4hep_track_pt_and_keeps_charge_sign():
    pos = track_features_cml(*_args(_single_track(qop=0.25, theta=np.pi / 2)))
    neg = track_features_cml(*_args(_single_track(qop=-0.25, theta=np.pi / 2)))
    # key4hep recovers pt from omega with track_pt(omega, b_field)
    assert np.isclose(track_pt(pos["omega"], B_FIELD)[0], pos["pt"][0], rtol=1e-5)
    assert pos["omega"][0] > 0 and neg["omega"][0] < 0
    assert np.isclose(np.abs(pos["omega"][0]), np.abs(neg["omega"][0]), atol=1e-12)


def test_zero_qop_omega_stays_finite():
    f = track_features_cml(*_args(_single_track(qop=0.0)))
    assert np.isfinite(f["omega"][0]) and np.isfinite(f["tanLambda"][0])


def test_radius_of_innermost_hit():
    # the single track owns hits 2, 5, 9 -> innermost at x = 3 mm
    tracks = _single_track(nmeas=3)
    tracks["hit_ids"] = [[2, 5, 9]]
    f = track_features_cml(*_args(tracks))
    assert np.isclose(f["radiusOfInnermostHit"][0], 3.0, atol=1e-6)


def test_radius_of_innermost_hit_uses_transverse_radius_not_z():
    tracks = _single_track(nmeas=2)
    tracks["hit_ids"] = [[0, 1]]
    tracker = {"x": np.array([5.0, 1.0]), "y": np.array([0.0, 0.0])}
    f = track_features_cml(tracks, tracker)
    # picks the smaller transverse radius (hit 1 at x=1), not the first-listed hit
    assert np.isclose(f["radiusOfInnermostHit"][0], 1.0, atol=1e-6)


def test_radius_of_innermost_hit_zero_for_hitless_track():
    tracks = _single_track(nmeas=0)
    tracks["hit_ids"] = [[]]
    f = track_features_cml(tracks, _tracker_hits())
    assert f["radiusOfInnermostHit"][0] == 0.0
    assert f["n_meas"][0] == 0
