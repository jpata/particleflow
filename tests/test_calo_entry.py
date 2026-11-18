import math
from pathlib import Path

import pytest
import torch

from mlpf.conf import X_FEATURES, MLPFConfig, SOURCE_IDS
from mlpf.model.calo_entry import MultiSourceTrackCaloEntry, TrackCaloEntry, calo_entry_eta_phi
from mlpf.model.mlpf import MLPF


def _t(*values):
    return torch.tensor(values, dtype=torch.float32)


def test_straight_track_keeps_its_direction():
    tan_lambda = _t(0.0, 0.5, -2.0)
    eta, phi, reached = calo_entry_eta_phi(_t(0.3, -1.0, 2.5), _t(0, 0, 0), tan_lambda, _t(0, 0, 0), radius=1500.0, endcap_z=2300.0)
    torch.testing.assert_close(eta, torch.asinh(tan_lambda))
    torch.testing.assert_close(phi, _t(0.3, -1.0, 2.5))
    assert reached.all()


def test_barrel_entry_of_a_bending_track():
    # a central track enters the barrel at azimuth phi0 - asin(omega * R / 2)
    omega, radius = 1e-4, 1500.0
    eta, phi, reached = calo_entry_eta_phi(_t(0.2, 0.2), _t(0, 0), _t(0, 0), _t(omega, -omega), radius=radius, endcap_z=2300.0)
    turn = math.asin(omega * radius / 2)
    torch.testing.assert_close(phi, _t(0.2 - turn, 0.2 + turn))
    torch.testing.assert_close(eta, _t(0, 0))


def test_forward_track_enters_the_endcap():
    # tanLambda 5: reaches |z| = 2300 mm after an arc of 460 mm < 1500 mm, so at r = chord(460 mm)
    omega, tan_lambda, endcap_z = 1e-3, 5.0, 2300.0
    eta, phi, reached = calo_entry_eta_phi(_t(0.0), _t(0.0), _t(tan_lambda), _t(omega), radius=1500.0, endcap_z=endcap_z)
    arc = endcap_z / tan_lambda
    chord = 2 * math.sin(omega * arc / 2) / omega
    torch.testing.assert_close(eta, _t(math.asinh(endcap_z / chord)))
    torch.testing.assert_close(phi, _t(-omega * arc / 2))


def test_curling_track_reaches_no_barrel():
    # 2 / omega < radius: never reaches the barrel; without pz it reaches no surface and keeps its vertex direction
    # (also away from z0 = 0, where the vertex eta is not asinh(z0 / chord))
    eta, phi, reached = calo_entry_eta_phi(_t(1.0, 1.0), _t(0.0, 100.0), _t(0.0, 0.0), _t(1e-2, 1e-2), radius=1500.0, endcap_z=2300.0)
    torch.testing.assert_close(eta, _t(0.0, 0.0))
    torch.testing.assert_close(phi, _t(1.0, 1.0))
    assert not reached.any()


def test_curling_track_reaches_the_endcap_after_several_turns():
    # 2 / omega < radius, tanLambda 0.2: reaches |z| = 2300 mm after an arc of 11500 mm, 1.83 turns at omega 1e-3
    omega, tan_lambda, endcap_z = 1e-3, 0.2, 2300.0
    eta, phi, reached = calo_entry_eta_phi(_t(0.0), _t(0.0), _t(tan_lambda), _t(omega), radius=2500.0, endcap_z=endcap_z)
    arc = endcap_z / tan_lambda
    # direct helix position: the center sits at distance 1 / omega from the origin, perpendicular to phi0
    x, y = math.sin(omega * arc) / omega, (math.cos(omega * arc) - 1) / omega
    assert reached.all()
    torch.testing.assert_close(phi, _t(math.atan2(y, x)))
    torch.testing.assert_close(eta, _t(math.asinh(endcap_z / math.hypot(x, y))))


def test_track_starting_beyond_the_endcap_keeps_its_vertex_direction():
    # |z0| >= endcap_z: the track starts outside the calorimeter's front, curling (reaches no surface) or not
    tan_lambda = _t(0.5, -0.3, 0.5)
    eta, phi, reached = calo_entry_eta_phi(
        _t(0.3, -1.0, 2.0), _t(2600.0, -3000.0, 2300.0), tan_lambda, _t(1e-2, 1e-4, 1e-4), radius=1500.0, endcap_z=2300.0
    )
    torch.testing.assert_close(eta, torch.asinh(tan_lambda))
    torch.testing.assert_close(phi, _t(0.3, -1.0, 2.0))
    assert not reached.any()


def test_straight_displaced_track_enters_the_barrel_off_its_direction():
    # a straight track with impact parameter d0 crosses r = R after a transverse arc sqrt(R^2 - d0^2)
    radius, d0, phi0 = 1500.0, 300.0, 0.4
    eta, phi, reached = calo_entry_eta_phi(_t(phi0), _t(0.0), _t(0.0), _t(0.0), radius=radius, endcap_z=2300.0, d0=_t(d0))
    arc = math.sqrt(radius**2 - d0**2)
    x, y = -d0 * math.sin(phi0) + arc * math.cos(phi0), d0 * math.cos(phi0) + arc * math.sin(phi0)
    assert reached.all()
    torch.testing.assert_close(phi, _t(math.atan2(y, x)))
    torch.testing.assert_close(eta, _t(0.0))


def _helix_scan(phi0, d0, z0, tan_lambda, omega, radius, endcap_z, steps=200_000):
    """Reference: walk the helix from its perigee in small steps until it reaches the barrel or an endcap plane."""
    s = torch.linspace(0, 4 * math.pi / abs(omega), steps, dtype=torch.float64)
    x = -d0 * math.sin(phi0) + (math.sin(phi0) - torch.sin(phi0 - omega * s)) / omega
    y = d0 * math.cos(phi0) + (torch.cos(phi0 - omega * s) - math.cos(phi0)) / omega
    z = z0 + s * tan_lambda
    out = (torch.hypot(x, y) >= radius) | (z.abs() >= endcap_z)
    i = int(out.to(torch.int8).argmax())
    return math.asinh(z[i] / math.hypot(x[i], y[i])), math.atan2(y[i], x[i])


@pytest.mark.parametrize(
    "phi0, d0, z0, tan_lambda, omega",
    [
        (0.3, 0.0, 0.0, 0.2, 2e-4),  # prompt, barrel
        (0.3, 400.0, -50.0, 0.2, 2e-4),  # displaced, barrel
        (-2.0, -400.0, 50.0, -0.3, -5e-4),  # displaced, other charge and side, barrel
        (1.0, 150.0, 0.0, 0.4, 2e-3),  # displaced curler: endcap after several turns
        (2.5, -600.0, 200.0, 1.5, -1e-4),  # displaced, endcap
    ],
)
def test_helix_entry_matches_a_step_by_step_helix(phi0, d0, z0, tan_lambda, omega):
    radius, endcap_z = 1500.0, 2300.0
    eta, phi, reached = calo_entry_eta_phi(_t(phi0), _t(z0), _t(tan_lambda), _t(omega), radius=radius, endcap_z=endcap_z, d0=_t(d0))
    ref_eta, ref_phi = _helix_scan(phi0, d0, z0, tan_lambda, omega, radius, endcap_z)
    assert reached.all()
    torch.testing.assert_close(eta, _t(ref_eta), atol=2e-3, rtol=0)
    torch.testing.assert_close(phi, _t(ref_phi), atol=2e-3, rtol=0)


def test_perigee_outside_the_barrel_keeps_its_direction():
    eta, phi, reached = calo_entry_eta_phi(_t(0.3), _t(0.0), _t(0.5), _t(1e-4), radius=1500.0, endcap_z=2300.0, d0=_t(1600.0))
    assert not reached.any()
    torch.testing.assert_close(eta, torch.asinh(_t(0.5)))
    torch.testing.assert_close(phi, _t(0.3))


def test_tracks_that_reach_no_surface_keep_their_features():
    entry = TrackCaloEntry("clic")
    z0, tan_lambda, omega, _ = entry.columns
    X = torch.zeros(1, 2, len(X_FEATURES["clic"]))
    X[..., 0] = 1
    X[..., 2] = _t(0.7, 0.7)  # stored eta, deliberately not asinh(tan_lambda)
    X[..., 3], X[..., 4] = math.sin(1.0), math.cos(1.0)
    X[..., z0], X[..., tan_lambda], X[..., omega] = _t(0.0, 3000.0), _t(0.5, 0.5), _t(1e-4, 1e-2)
    out = entry(X)
    assert not torch.allclose(out[0, 0, 2:5], X[0, 0, 2:5])  # reaches the barrel: moved
    torch.testing.assert_close(out[0, 1], X[0, 1])  # starts beyond the endcap plane: untouched


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


@pytest.mark.parametrize("dataset", ["colliderml", "clic", "cld", "idea", "cms"])
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

    # non-finite extrapolations also keep the vertex direction
    X[1, :2, entry.columns[0]] = float("nan")
    X[1, 2:4, entry.columns[1]] = float("inf")
    out = entry(X)
    torch.testing.assert_close(out[1, :4], X[1, :4], equal_nan=True)
    assert torch.isfinite(out[..., 2:5]).all()


def test_missing_columns_name_the_flag_and_dataset(monkeypatch):
    import mlpf.model.calo_entry as calo_entry

    monkeypatch.setattr(calo_entry, "HELIX_COLUMNS", ("z0", "tanlambda", "no_such_column"))
    with pytest.raises(ValueError, match=r"track_hash_at_calo needs the X_FEATURES columns \['no_such_column'\].*clic"):
        TrackCaloEntry("clic")


def test_geometry_override_and_unsupported_datasets():
    _, model = _heptv2("clic", track_hash_at_calo=True, track_hash_calo_radius_mm=1000.0)
    assert (model.track_calo_entry.radius, model.track_calo_entry.endcap_z) == (1000.0, 2408.0)

    _, model = _heptv2("idea", track_hash_at_calo=True)
    assert (model.track_calo_entry.radius, model.track_calo_entry.endcap_z) == (2500.0, 2500.0)

    with pytest.raises(ValueError, match="no calorimeter geometry"):
        _heptv2("maia", track_hash_at_calo=True)
    _, model = _heptv2("maia", track_hash_at_calo=True, track_hash_calo_radius_mm=2000.0, track_hash_calo_endcap_z_mm=2450.0)
    assert model.track_calo_entry.mode == "helix"

    with pytest.raises(ValueError, match="not supported"):
        _heptv2("clic_hits", track_hash_at_calo=True)


@pytest.mark.parametrize("backbone_mode", ["shared", "split"])
@pytest.mark.parametrize("method", ["forward", "encode_backbone", "encode_backbone_layers"])
def test_joint_hashing_uses_each_events_source(backbone_mode, method):
    sources = ("clic", "cld", "colliderml", "cld")
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld",
            "data_dir": "/tmp",
            "input_dim": 19,
            "model": {
                "type": "heptv2",
                "input_encoding": "split",
                "backbone": {"mode": backbone_mode},
                "heptv2": {"num_convs": 1, "num_heads": 2, "embedding_dim": 16, "width": 16, "block_size": 8, "track_hash_at_calo": True},
            },
            "conv_type": "heptv2",
            "test_dataset": {f"{source}_fixture_pf": {"version": "1.0.0", "splits": ["fixture"]} for source in sources},
        }
    )
    model = MLPF(config).eval()
    X = torch.zeros(len(sources), 8, config.input_dim)
    X[..., 0] = 1
    X[..., 4] = 1
    X[:, 1, 0] = 2
    X[:, -1, 0] = 0
    for event, source in enumerate(sources):
        z0, tan_lambda, omega, d0 = TrackCaloEntry(source).columns
        X[event, :, z0] = 0
        X[event, :, d0] = 0
        X[event, :, tan_lambda] = 0.2
        X[event, :, omega] = 1e-4
    original = X.clone()
    source_id = torch.tensor([SOURCE_IDS[source] for source in sources])
    expected = torch.cat([TrackCaloEntry(source)(X[event : event + 1]) for event, source in enumerate(sources)])
    seen = []
    backbones = [model.conv_id] if backbone_mode == "shared" or method != "forward" else [model.conv_id, model.conv_reg]
    for backbone in backbones:
        backbone[0].register_forward_pre_hook(lambda module, args: seen.append(args[2].clone()))

    getattr(model, method)(X, X[..., 0] != 0, source_id=source_id)

    assert len(seen) == len(backbones)
    for coordinates in seen:
        torch.testing.assert_close(coordinates, expected)
    torch.testing.assert_close(X, original)
    torch.testing.assert_close(expected[X[..., 0] != 1], X[X[..., 0] != 1])
    torch.testing.assert_close(expected[..., 5:], X[..., 5:])
    assert not torch.allclose(expected[0, 0, 3:5], expected[1, 0, 3:5])


@pytest.mark.parametrize("source_id", [None, torch.tensor([SOURCE_IDS["cld"]])], ids=["missing", "wrong-batch-size"])
def test_multi_source_hashing_requires_one_source_id_per_event(source_id):
    entry = MultiSourceTrackCaloEntry(("cld", "clic"))
    with pytest.raises(ValueError, match="source_id"):
        entry(torch.zeros(2, 8, 19), source_id)


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
        ("pyg-idea-pipeline-v1", "idea", False),
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


def test_portable_asin_matches_fp32_reference():
    from mlpf.model.calo_entry import _asin

    # Dense coverage plus tiny arguments and adjacent FP32 values around the
    # range-reduction boundary and the ill-conditioned endpoints.
    grid = torch.linspace(-1, 1, 131073)
    tiny = torch.logspace(-40, -2, 1000)
    edges = _t(-1, -0.5, 0, 0.5, 1)
    neighbors = torch.cat(
        (torch.nextafter(edges, torch.full_like(edges, -torch.inf)), edges, torch.nextafter(edges, torch.full_like(edges, torch.inf)))
    ).clamp(-1, 1)
    values = torch.cat((grid, tiny, -tiny, neighbors)).sort().values
    expected = torch.asin(values.double()).float()
    actual = _asin(values)
    spacing = torch.nextafter(expected.abs(), torch.full_like(expected, torch.inf)) - expected.abs()
    assert torch.all((actual - expected).abs() <= 2 * spacing)
    assert torch.all(actual[1:] >= actual[:-1])
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("scale", [0.0, 1e-20, 0.3, 1.0, 1e20])
def test_unit_direction_matches_atan2_without_angle_round_trip(scale):
    from mlpf.model.calo_entry import _unit_direction

    # Both axes, all quadrants and zero; include non-unit vectors.
    x = _t(0, 1, -1, 0, 0, 2, -3, -4, 5) * scale
    y = _t(0, 0, 0, 1, -1, 3, 4, -5, -6) * scale
    sine, cosine = _unit_direction(x, y)
    angle = torch.where((x == 0) & (y == 0), torch.zeros_like(x), torch.atan2(y, x))
    torch.testing.assert_close(sine, angle.sin(), atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(cosine, angle.cos(), atol=2e-7, rtol=2e-7)
    assert torch.isfinite(sine).all() and torch.isfinite(cosine).all()
    # Exact axis directions must not acquire the sin(pi) residual that can
    # move deterministic hashing across the phi branch cut.
    assert torch.all(sine[y == 0] == 0)
    assert torch.all(cosine[(x == 0) & (y != 0)] == 0)


def test_direct_track_direction_matches_native_asin_geometry(monkeypatch):
    import mlpf.model.calo_entry as calo

    entry = TrackCaloEntry("cld")
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(2, 256, len(X_FEATURES["cld"]), generator=generator)
    x[..., 0] = 1
    z0, _tan_lambda, omega, d0 = entry.columns
    x[..., z0] *= 3000  # includes tracks outside the endcaps
    x[..., omega] *= 0.002  # includes curlers and both charge signs
    x[..., d0] *= 2000  # includes displaced tracks outside the barrel
    x[..., 3:5] *= 3  # non-unit input directions
    actual = entry(x)
    # Independent native inverse trig for the angle API and barrel arc.
    monkeypatch.setattr(calo, "_asin", torch.asin)
    eta, phi, reached = entry.eta_phi(x)
    expected = x.clone()
    expected[..., 2] = torch.where(reached, eta, x[..., 2])
    expected[..., 3] = torch.where(reached, phi.sin(), x[..., 3])
    expected[..., 4] = torch.where(reached, phi.cos(), x[..., 4])
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(actual[~reached], x[~reached], atol=0, rtol=0)
