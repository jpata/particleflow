import copy
import json
import math

import pytest
import torch

from mlpf.model.mlpf import MLPF
from mlpf.model.ssl import DistillationConfig, StudentTeacher
from mlpf.model.ssl_regions import RegionConfig, RegionSampler, hit_addresses, measurement_loss, pool_teacher, region_mean
from tests.test_set_prediction import make_attention_config


@pytest.fixture
def region_setup(tmp_path):
    regions = [[kind, 3 if kind == 1 else 1, 1 if kind == 1 else 20, 0, layer, eta, phi]
               for kind in [1, 2] for layer in range(4) for eta in range(2) for phi in range(4)]
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(dict(schema="edm4hep-hit-regions-v1", eta_width=1., phi_bins=4, fine_bins=2,
                                    regions=regions, valid_bins=[[True] * 4 for _ in regions])))
    config = RegionConfig(geometry_path=str(path), tracker_surface_fraction=.5, calo_surface_fraction=.5,
                          tracker_queries=6, calo_queries=6, predictor_dim=16)
    rows = []
    for kind in [1, 2]:
        for layer in range(4):
            for eta, phi in [(.25, .2), (1.25, 1.9)]:
                x = torch.zeros(15)
                x[0], x[2], x[3], x[4], x[5] = kind, eta, math.sin(phi), math.cos(phi), 2.
                x[6:9] = torch.tensor([100 * math.cos(phi), 100 * math.sin(phi), 100 * math.sinh(eta)])
                x[10], x[12], x[14] = (3 if kind == 1 else 1), (1 if kind == 1 else 20), layer
                rows.append(x)
    return config, torch.stack(rows)[None]


def sample(sampler, x):
    return sampler(x, x[..., 0] != 0, torch.Generator().manual_seed(17))


def test_hidden_measurements_and_multiplicity_cannot_change_queries_or_visible_inputs(region_setup):
    config, x = region_setup
    sampler = RegionSampler(config)
    a = sample(sampler, x)
    hidden = a["hit_to_query"][0] >= 0
    assert hidden.any() and (~hidden).any()
    # Change energy, exact location within a region, and hidden multiplicity.
    edited = x.clone()
    edited[0, hidden, 5] = 30
    edited[0, hidden, 2] += .1
    edited = torch.cat([edited, edited[:, hidden][:, :1]], 1)
    b = sample(sampler, edited)
    for key in ["view", "mask", "features", "types", "valid"]:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert b["counts"].sum() == a["counts"].sum() + 1
    assert b["energy"].sum() > a["energy"].sum()


def test_targets_include_empty_cells_and_match_removed_measurements(region_setup):
    config, x = region_setup
    sampler = RegionSampler(config)
    result = sample(sampler, x)
    hidden = result["hit_to_query"][0] >= 0
    assert result["counts"].sum() == hidden.sum()
    assert result["energy"].sum() == x[0, hidden, 5].sum()
    assert (result["valid"] & (result["counts"] == 0)).any()
    assert (result["valid"] & (result["counts"] > 0)).any()
    assert result["view"].shape[1] + int(hidden.sum()) == x.shape[1]
    # Padding never becomes a target or alters the random sampler.
    padded = torch.nn.functional.pad(x, (0, 0, 0, 13))
    other = sample(sampler, padded)
    for key in ["view", "features", "types", "counts", "energy", "valid"]:
        torch.testing.assert_close(result[key], other[key])


def test_periodic_phi_boundaries_have_consistent_addresses():
    x = torch.zeros(2, 15)
    x[:, 0] = 1
    x[:, 3] = torch.tensor([0., -0.])
    x[:, 4] = -1
    a, fine = hit_addresses(x, .5, 16, 2)
    torch.testing.assert_close(a[0], a[1])
    assert fine[0] == fine[1]


@pytest.mark.parametrize("latent_weight", [0., .1])
def test_region_gradients_teacher_freeze_and_state_reload(region_setup, latent_weight):
    config, x = region_setup
    config.latent_weight = latent_weight
    architecture = make_attention_config()
    student = MLPF(architecture)
    dimension = student.encode_backbone(x, x[..., 0] != 0).shape[-1]
    model = StudentTeacher(student, dimension, DistillationConfig(mode="regions", regions=config, student_views=2, prototypes=8, projection_dim=4))
    model.train()
    initial = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    loss, metrics = model(x, generator=torch.Generator().manual_seed(15))
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in metrics.values())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.student.backbone.parameters())
    assert model.region_predictor.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.student.set_decoder.parameters())
    for k, v in model.teacher.state_dict().items():
        torch.testing.assert_close(v, initial[k])
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    a = model(x, generator=torch.Generator().manual_seed(28))[0]
    b = clone(x, generator=torch.Generator().manual_seed(28))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_balancing_and_energy_loss_only_uses_calo(region_setup):
    config, x = region_setup
    result = sample(RegionSampler(config), x)
    shape = result["counts"].shape
    prediction = [torch.zeros(shape, requires_grad=True) for _ in range(3)] + [None]
    _, before = measurement_loss(prediction, result, config)
    result["energy"][result["types"] == 1] += 10000
    _, after = measurement_loss(prediction, result, config)
    torch.testing.assert_close(before["energy_loss"], after["energy_loss"])
    values = torch.tensor([[[1., 1.], [3., 3.], [3., 3.]]])
    assert region_mean(values, torch.tensor([[1, 2, 2]]), torch.ones_like(values, dtype=torch.bool)) == 2


def test_teacher_pool_is_only_hidden_rows(region_setup):
    config, x = region_setup
    result = sample(RegionSampler(config), x)
    features = torch.arange(x.shape[1]).float()[None, :, None]
    pooled, present = pool_teacher(features, result)
    for q in torch.where(present[0])[0]:
        expected = features[0, result["hit_to_query"][0] == q].mean()
        torch.testing.assert_close(pooled[0, q, 0], expected)


def test_region_config_requires_geometry():
    with pytest.raises(ValueError, match="requires"):
        DistillationConfig(mode="regions")


def test_region_exact_resume_preview_and_standard_export(region_setup, tmp_path, monkeypatch):
    import mlpf.pretrain as pretrain
    from mlpf.model.losses import make_task_loss_weighter

    region_config, x = region_setup
    architecture = make_attention_config()
    data = [{"X": x[0].numpy(), "source_id": 1}]
    monkeypatch.setattr(pretrain, "build_model_config", lambda _: architecture)
    monkeypatch.setattr(pretrain, "load_inputs", lambda *args: data)
    monkeypatch.setattr(pretrain, "check_splits", lambda _: None)
    settings = pretrain.PretrainConfig(data_dir="/tmp", datasets=["fixture"], steps=2, dtype="float32", batch_size=1,
                                      validation_frequency=1, checkpoint_frequency=1,
                                      ssl=DistillationConfig(mode="regions", regions=region_config, prototypes=8, projection_dim=4))
    pretrain.run(settings, tmp_path / "resumed", "cpu")
    settings.steps = 3
    pretrain.run(settings, tmp_path / "resumed", "cpu", resume=True)
    pretrain.run(settings, tmp_path / "continuous", "cpu")
    a = torch.load(tmp_path / "resumed/latest.pth", weights_only=False)
    b = torch.load(tmp_path / "continuous/latest.pth", weights_only=False)
    for key in a["ssl_state_dict"]:
        torch.testing.assert_close(a["ssl_state_dict"][key], b["ssl_state_dict"][key], rtol=0, atol=0)
    assert (tmp_path / "resumed/region_view.npz").exists()
    exported = torch.load(tmp_path / "resumed/student.pth", weights_only=False)
    model = MLPF(architecture)
    model.task_loss_weighter = make_task_loss_weighter(architecture.task_loss_weights.model_dump())
    model.load_state_dict(exported["model_state_dict"], strict=True)
    assert not any("region_" in k for k in exported["model_state_dict"])
