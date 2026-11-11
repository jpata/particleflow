import math
from pathlib import Path

import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.calo_entry import TrackCaloEntry, calo_entry_eta_phi
from mlpf.model.mlpf import MLPF


def _t(*values):
    return torch.tensor(values, dtype=torch.float32)


def test_straight_track_keeps_its_direction():
    tan_lambda = _t(0.0, 0.5, -2.0)
    eta, phi = calo_entry_eta_phi(_t(0.3, -1.0, 2.5), _t(0, 0, 0), tan_lambda, _t(0, 0, 0), radius=1500.0, endcap_z=2300.0)
    torch.testing.assert_close(eta, torch.asinh(tan_lambda))
    torch.testing.assert_close(phi, _t(0.3, -1.0, 2.5))


def test_barrel_entry_of_a_bending_track():
    # a central track enters the barrel at azimuth phi0 - asin(omega * R / 2)
    omega, radius = 1e-4, 1500.0
    eta, phi = calo_entry_eta_phi(_t(0.2, 0.2), _t(0, 0), _t(0, 0), _t(omega, -omega), radius=radius, endcap_z=2300.0)
    turn = math.asin(omega * radius / 2)
    torch.testing.assert_close(phi, _t(0.2 - turn, 0.2 + turn))
    torch.testing.assert_close(eta, _t(0, 0))


def test_forward_track_enters_the_endcap():
    # tanLambda 5: reaches |z| = 2300 mm after an arc of 460 mm < 1500 mm, so at r = chord(460 mm)
    omega, tan_lambda, endcap_z = 1e-3, 5.0, 2300.0
    eta, phi = calo_entry_eta_phi(_t(0.0), _t(0.0), _t(tan_lambda), _t(omega), radius=1500.0, endcap_z=endcap_z)
    arc = endcap_z / tan_lambda
    chord = 2 * math.sin(omega * arc / 2) / omega
    torch.testing.assert_close(eta, _t(math.asinh(endcap_z / chord)))
    torch.testing.assert_close(phi, _t(-omega * arc / 2))


def test_curling_track_reaches_no_barrel():
    # 2 / omega < radius: never reaches the barrel; without pz it keeps its vertex direction
    eta, phi = calo_entry_eta_phi(_t(1.0), _t(0.0), _t(0.0), _t(1e-2), radius=1500.0, endcap_z=2300.0)
    torch.testing.assert_close(eta, _t(0.0))
    torch.testing.assert_close(phi, _t(1.0))


def _heptv2(dataset, **heptv2):
    config = MLPFConfig.model_validate(
        {
            "dataset": dataset,
            "data_dir": "/tmp",
            "model": {
                "type": "heptv2",
                "heptv2": {"num_convs": 1, "num_heads": 2, "embedding_dim": 16, "width": 16, "block_size": 8, **heptv2},
            },
            "conv_type": "heptv2",
        }
    )
    torch.manual_seed(0)
    return config, MLPF(config)


def _inputs(config):
    torch.manual_seed(1)
    X = torch.randn(2, 16, config.input_dim)
    elem_types = torch.tensor(config.elemtypes_nonzero)
    X[..., 0] = elem_types[torch.arange(32).reshape(2, 16) % len(elem_types)].to(X.dtype)
    X[..., 1] = torch.exp(X[..., 1])
    X[..., 5] = torch.exp(X[..., 5])
    phi = torch.atan2(X[..., 3], X[..., 4])
    X[..., 3], X[..., 4] = torch.sin(phi), torch.cos(phi)
    if config.dataset.value != "cms":
        omega = TrackCaloEntry(config.dataset.value).columns[2]
        X[..., omega] = X[..., omega] * 1e-3  # 1/mm: a few hundred MeV to a few GeV
    return X, torch.ones(2, 16, dtype=torch.bool)


@pytest.mark.parametrize("dataset", ["colliderml", "clic", "cld", "cms"])
def test_track_hash_at_calo_only_moves_tracks_in_the_backbone(dataset):
    config_off, model_off = _heptv2(dataset)
    config_on, model_on = _heptv2(dataset, track_hash_at_calo=True)
    assert model_off.track_calo_entry is None
    assert model_off.state_dict().keys() == model_on.state_dict().keys()
    model_on.load_state_dict(model_off.state_dict())
    model_off.eval(), model_on.eval()

    X, mask = _inputs(config_on)
    seen = []
    model_on.backbone[0].register_forward_pre_hook(lambda module, args: seen.append(args[2].clone()))
    out_on = model_on(X, mask)
    out_off = model_off(X, mask)

    is_track = X[..., 0] == 1
    backbone_X = seen[0]
    assert not torch.allclose(backbone_X[is_track][:, 3], X[is_track][:, 3])
    torch.testing.assert_close(backbone_X[~is_track], X[~is_track])
    torch.testing.assert_close(backbone_X[..., 5:], X[..., 5:])
    # the moved tracks still carry a unit (sin_phi, cos_phi)
    torch.testing.assert_close(backbone_X[is_track][:, 3] ** 2 + backbone_X[is_track][:, 4] ** 2, torch.ones(int(is_track.sum())))
    assert not torch.allclose(out_on[2], out_off[2])


def test_cms_tracks_without_ecal_extrapolation_keep_their_direction():
    config, _ = _heptv2("cms")
    entry = TrackCaloEntry("cms")
    X, _ = _inputs(config)
    X[..., 0] = 1
    X[0, :, entry.columns] = 0
    out = entry(X)
    torch.testing.assert_close(out[0], X[0])
    torch.testing.assert_close(out[1, :, 2], X[1, :, entry.columns[0]])


def test_geometry_override_and_unsupported_datasets():
    _, model = _heptv2("clic", track_hash_at_calo=True, track_hash_calo_radius_mm=1000.0)
    assert (model.track_calo_entry.radius, model.track_calo_entry.endcap_z) == (1000.0, 2408.0)

    with pytest.raises(ValueError, match="no calorimeter geometry"):
        _heptv2("idea", track_hash_at_calo=True)
    _, model = _heptv2("idea", track_hash_at_calo=True, track_hash_calo_radius_mm=2100.0, track_hash_calo_endcap_z_mm=2500.0)
    assert model.track_calo_entry.mode == "helix"

    with pytest.raises(ValueError, match="not supported"):
        _heptv2("clic_hits", track_hash_at_calo=True)


@pytest.mark.parametrize("dataset", ["clic", "cld"])
def test_cluster_hash_at_centroid_only_moves_clusters_in_the_backbone(dataset):
    _, model_off = _heptv2(dataset)
    config_on, model_on = _heptv2(dataset, cluster_hash_at_centroid=True)
    assert model_off.cluster_centroid is None
    assert model_off.state_dict().keys() == model_on.state_dict().keys()
    model_on.load_state_dict(model_off.state_dict())
    model_off.eval(), model_on.eval()

    X, mask = _inputs(config_on)
    seen = []
    model_on.backbone[0].register_forward_pre_hook(lambda module, args: seen.append(args[2].clone()))
    out_on = model_on(X, mask)
    out_off = model_off(X, mask)

    is_cluster = X[..., 0] == 2
    backbone_X = seen[0]
    x, y, z = (X[is_cluster][:, column] for column in model_on.cluster_centroid.columns)
    torch.testing.assert_close(backbone_X[is_cluster][:, 2], torch.asinh(z / torch.hypot(x, y).clamp_min(1.0)))
    torch.testing.assert_close(torch.atan2(backbone_X[is_cluster][:, 3], backbone_X[is_cluster][:, 4]), torch.atan2(y, x))
    torch.testing.assert_close(backbone_X[~is_cluster], X[~is_cluster])
    torch.testing.assert_close(backbone_X[..., 5:], X[..., 5:])
    assert not torch.allclose(out_on[2], out_off[2])


def test_track_and_cluster_coordinates_combine():
    config, model = _heptv2("clic", track_hash_at_calo=True, cluster_hash_at_centroid=True)
    X, _ = _inputs(config)
    expected = model.cluster_centroid(model.track_calo_entry(X))
    seen = []
    model.backbone[0].register_forward_pre_hook(lambda module, args: seen.append(args[2].clone()))
    model.eval()
    model(X, torch.ones(X.shape[:2], dtype=torch.bool))
    torch.testing.assert_close(seen[0], expected)


def test_cluster_hash_at_centroid_rejects_position_clusters():
    with pytest.raises(ValueError, match="not supported"):
        _heptv2("cms", cluster_hash_at_centroid=True)
    with pytest.raises(ValueError, match="not supported"):
        _heptv2("colliderml", cluster_hash_at_centroid=True)


SPEC_FILE = Path(__file__).resolve().parents[1] / "particleflow_spec.yaml"


@pytest.mark.parametrize(
    "model_name, production, centroid",
    [
        ("pyg-cms-v1", "cms_run3", False),
        ("pyg-clic-v1", "clic", True),
        ("pyg-cld-v1", "cld", True),
        ("pyg-colliderml-v1", "colliderml", False),
        ("pyg-colliderml-pu200-v1", "colliderml", False),
        ("pyg-colliderml-pu0pu200-groups-v1", "colliderml", False),
    ],
)
def test_spec_pf_models_hash_at_calo_when_switched_to_heptv2(model_name, production, centroid):
    heptv2 = MLPFConfig.from_spec(SPEC_FILE, model_name, production, extra_args=["--model.type", "heptv2"]).model.heptv2
    assert heptv2.track_hash_at_calo
    assert heptv2.cluster_hash_at_centroid == centroid


def test_spec_hit_models_keep_vertex_hashing():
    heptv2 = MLPFConfig.from_spec(SPEC_FILE, "pyg-cld-hits-heptv2-set-v1", "cld").model.heptv2
    assert not heptv2.track_hash_at_calo
    assert not heptv2.cluster_hash_at_centroid
