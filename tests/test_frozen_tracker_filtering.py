import numpy as np
import pytest
import torch

from mlpf.model.tracker_group_readout import tracker_group_geometry
from scripts.studies.frozen_tracker_filtering import circle, filter_fit, jet_metrics


def arc(radius=10.0):
    s = np.linspace(0.1, 1.6, 24)
    x = np.zeros((24, 15), dtype=np.float64)
    x[:, 0] = 1
    x[:, 6] = radius * np.sin(s / radius) * 1000
    x[:, 7] = radius * (1 - np.cos(s / radius)) * 1000
    x[:, 8] = s * 500
    x[:, 14] = np.arange(24)
    return x


@pytest.mark.parametrize("radius", [1.0, 10.0, 100.0])
def test_numpy_circle_matches_production(radius):
    x = arc(radius)
    x[:, 6:9] += np.random.default_rng(2).normal(0, 0.005, (24, 3))
    _, p, v = tracker_group_geometry(torch.tensor(x), torch.zeros(24, dtype=torch.long), 1, 2.0)
    result = circle(x)
    assert bool(v[0]) == (result is not None)
    assert result["pt"] == pytest.approx(float(p[0].exp()), rel=1e-5)


@pytest.mark.parametrize("family", ["confidence", "geometry", "combined"])
def test_filters_remove_obvious_contamination(family):
    x = arc()
    bad = np.array([3, 8, 13, 18])
    x[bad, 7] += 100
    x[bad, 8] += 200
    probability = np.full(24, 0.95)
    probability[bad] = 0.1
    config = dict(family=family)
    if family != "geometry":
        config.update(score="probability", threshold=0.8)
    if family != "confidence":
        config["tolerance_m"] = 0.001
    result, keep = filter_fit(x, probability, np.ones(24), config)
    assert result is not None
    assert not keep[bad].any()
    assert keep.sum() == 20
    assert result["pt"] == pytest.approx(0.299792458 * 20, rel=1e-4)


def test_jet_diagnostic_identical_collections():
    p = np.array([[1, 30, 0, 0, 30], [1, 30, 0, np.pi, 30]], dtype=float)
    result = jet_metrics([(p, p.copy())])
    assert result["response_qualified_f1"] == pytest.approx(1.0)
    assert result["response_iqr"] == pytest.approx(0.0)
