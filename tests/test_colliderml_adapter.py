# Unit tests for the ColliderML track adapter (ACTS perigee -> MLPF features).
import numpy as np

from mlpf.data.colliderml.tracks import track_features_cml


def _single_track(d0=0.01, z0=0.02, phi=0.3, theta=1.0, qop=0.5, nmeas=10):
    return {
        "d0": np.array([d0]),
        "z0": np.array([z0]),
        "phi": np.array([phi]),
        "theta": np.array([theta]),
        "qop": np.array([qop]),
        "hit_ids": [[0] * nmeas],
    }


def test_track_pt_eta_invariant_mass():
    tr = track_features_cml(_single_track(qop=0.25, theta=np.pi / 2))  # theta=pi/2 -> eta=0
    assert np.isclose(tr["pt"][0], 4.0, atol=1e-6), f"pt={tr['pt']}"
    assert np.isclose(tr["eta"][0], 0.0, atol=1e-6), f"eta={tr['eta']}"
    # p = |1/qop|
    assert np.isclose(tr["p"][0], 4.0, atol=1e-6)
    # energy-like scale should be positive
    assert tr["p"][0] > 0


def test_track_pt_decreases_with_qop():
    f1 = track_features_cml(_single_track(qop=0.5))
    f2 = track_features_cml(_single_track(qop=0.25))
    assert f2["pt"][0] > f1["pt"][0]


def test_track_zero_qop_safe():
    f = track_features_cml(_single_track(qop=0.0))
    # we protect with a small epsilon rather than crash; result should not be inf/nan
    assert np.isfinite(f["pt"][0]) and np.isfinite(f["eta"][0])


def test_n_meas_matches_hit_count():
    f = track_features_cml(_single_track(nmeas=7))
    assert f["n_meas"][0] == 7


def test_eta_symmetry():
    forward = track_features_cml(_single_track(theta=np.pi / 4))
    backward = track_features_cml(_single_track(theta=3 * np.pi / 4))
    assert np.isclose(forward["eta"][0], -backward["eta"][0], atol=1e-6)
