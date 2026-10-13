import numpy as np
import pytest

from scripts.studies.diagnose_tracker_fits import direction_fit, fit_groups


@pytest.mark.parametrize('turn,slope', [(1, .5), (-1, -.8)])
def test_diagnostic_helix_direction_and_production_pt(turn, slope):
    radius = 10.
    distance = np.linspace(.1, 1.6, 24)
    angle = distance/radius
    phi = .4
    xy = np.column_stack([radius*np.sin(angle), turn*radius*(1-np.cos(angle))])
    rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    x = np.zeros((24, 15))
    x[:, 0] = 1
    x[:, 6:8] = xy@rotation.T*1000
    x[:, 8] = slope*distance*1000
    x[:, 14] = np.arange(24)
    x = x[np.random.default_rng(42).permutation(24)]
    result = fit_groups(x, np.zeros(24, dtype=int), 1, 2.)[0]
    assert result['valid']
    assert result['pt'] == pytest.approx(.299792458*2*radius, rel=1e-4)
    assert result['direction']['eta'] == pytest.approx(np.arcsinh(slope), abs=1e-5)
    assert result['direction']['phi'] == pytest.approx(phi, abs=1e-5)
    assert result['direction']['z_rms_m'] < 1e-6


def test_diagnostic_direction_rejects_insufficient_geometry():
    assert direction_fit(np.zeros((3, 15))) is None
    assert direction_fit(np.zeros((8, 15))) is None
