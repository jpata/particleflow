from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from scripts.visualize_colliderml import adapt_event
from scripts.visualize_key4hep import DETECTORS, _detector_config, _event, _particle_trajectory, _track_paths, render_event


def sample_record():
    return {
        "tracks": {"theta": np.array([np.pi / 4]), "qop": np.array([0.1]), "phi": np.array([0.0]), "d0": np.array([0.0]), "z0": np.array([0.0])},
        "calo_hits": {
            "detector": np.array([10]),
            "total_energy": np.array([1.0]),
            "x": np.array([1400.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
        },
        "tracker_hits": {"detector": np.array([0]), "x": np.array([100.0]), "y": np.array([0.0]), "z": np.array([100.0])},
        "particles": {
            "particle_id": np.array([1]),
            "pdg_id": np.array([22]),
            "parent_id": np.array([0]),
            "primary": np.array([True]),
            "charge": np.array([0.0]),
            "mass": np.array([0.0]),
            "energy": np.array([1.0]),
            "px": np.array([1.0]),
            "py": np.array([0.0]),
            "pz": np.array([0.0]),
        },
    }


def test_maia_autodetection_is_distinct_from_cld():
    tree = {"SiTracks_Refitted": [], "PandoraClusters": [], "EcalBarrelCollectionRec": [], "MCParticle": []}
    assert _detector_config(tree).key == "maia"
    assert _detector_config({"SiTracks_Refitted": [], "PandoraClusters": []}).key == "cld"
    assert DETECTORS["maia"].particle_collection == "MCParticle"


@pytest.mark.parametrize("use_saved_clusters", [False, True])
def test_colliderml_adapter_and_shared_renderer(tmp_path, use_saved_clusters):
    clusters = np.zeros((1, 17)) if use_saved_clusters else None
    if clusters is not None:
        clusters[0, 5:9] = [1.0, 1400.0, 0.0, 0.0]
    tree = adapt_event(sample_record(), 7, 10, clusters)
    assert _detector_config(tree).key == "colliderml"
    state = "_ActsTracks_trackStates/_ActsTracks_trackStates."
    np.testing.assert_allclose(_event(tree, state + "tanLambda", 7), [1.0])
    np.testing.assert_allclose(_event(tree, state + "omega", 7), [2.99792458e-4 * 3 * 0.1 / np.sin(np.pi / 4)])
    paths = _track_paths(tree, 7, DETECTORS["colliderml"])
    assert len(paths) == 1 and len(paths[0][0]) > 0
    # The initial direction is +x and +z, with positive-charge bending to -y.
    assert paths[0][0][1] > 0 and paths[0][1][1] < 0 and paths[0][2][1] > 0
    output = tmp_path / "display.png"
    assert render_event(Path("release"), 7, output, detector="colliderml", tree=tree) == "colliderml"
    assert output.stat().st_size > 1000
    with pytest.raises(IndexError):
        _event(tree, state + "omega", 0)


@pytest.mark.parametrize("detector", ["clic", "cld", "idea", "colliderml", "maia"])
def test_charged_particle_guide_has_expected_radius_and_charge_sign(detector):
    field = DETECTORS[detector].magnetic_field_tesla
    x, y, z = _particle_trajectory(1.0, 0.0, 0.5, 1.0, 1000.0, field)
    xn, yn, zn = _particle_trajectory(1.0, 0.0, 0.5, -1.0, 1000.0, field)
    radius = 1 / (2.99792458e-4 * field)
    assert y[-1] < 0 and yn[-1] > 0 and z[-1] > 0
    np.testing.assert_allclose(x * x + (y + radius) ** 2, radius**2)
    np.testing.assert_allclose(x, xn)
    np.testing.assert_allclose(y, -yn)
    np.testing.assert_allclose(z, zn)


@pytest.mark.parametrize("charge,field", [(0.0, 3.0), (1.0, 0.0)])
def test_neutral_or_zero_field_particle_guide_is_straight(charge, field):
    x, y, z = _particle_trajectory(3.0, 4.0, 0.0, charge, 1000.0, field)
    np.testing.assert_allclose([x[-1], y[-1], z[-1]], [600.0, 800.0, 0.0])


def test_low_pt_guide_is_sampled_and_limited_to_one_revolution():
    x, y, z = _particle_trajectory(0.001, 0.0, 0.001, 1.0, 6000.0, 5.0)
    assert len(x) > 100
    np.testing.assert_allclose([x[-1], y[-1]], [0.0, 0.0], atol=1e-12)
    assert 0 < z[-1] < 10
