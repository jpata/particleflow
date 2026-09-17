import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.PFDataset import PFBatch
from mlpf.model.mlpf import MLPF
from mlpf.model.set_losses import hungarian_match, set_event_loss
from mlpf.model.utils import unpack_predictions, unpack_target


REGRESSION_WEIGHTS = {feature: 1.0 for feature in ("pt", "eta", "sin_phi", "cos_phi", "energy")}


def make_config(num_slots=4, **set_decoder_overrides):
    return MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "model": {
                "type": "heptv2",
                "output_mode": "set",
                "input_encoding": "joint",
                "heptv2": {
                    "num_convs": 0,
                    "num_heads": 2,
                    "embedding_dim": 16,
                    "width": 16,
                    "block_size": 8,
                },
                "set_decoder": {
                    "num_slots": num_slots,
                    "num_layers": 2,
                    "num_heads": 2,
                    **set_decoder_overrides,
                },
                "hit_feature_engineering": {"enabled": False},
            },
            "conv_type": "heptv2",
        }
    )


def make_attention_config(num_slots=4):
    return MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "model": {
                "type": "attention",
                "output_mode": "set",
                "input_encoding": "joint",
                "attention": {
                    "num_convs": 1,
                    "num_heads": 2,
                    "head_dim": 8,
                    "use_pre_layernorm": True,
                },
                "set_decoder": {
                    "num_slots": num_slots,
                    "num_layers": 1,
                    "num_heads": 2,
                },
                "hit_feature_engineering": {"enabled": False},
            },
            "conv_type": "attention",
        }
    )


def make_target_tensor(batch_size=1, num_targets=2):
    target = torch.zeros(batch_size, num_targets, 14)
    phi = torch.linspace(-torch.pi, torch.pi, num_targets + 1)[:-1]
    pt = torch.linspace(1.0, 20.0, num_targets)
    target[..., 0] = (torch.arange(num_targets) % 5) + 1
    target[..., 2] = torch.log(pt)
    target[..., 3] = torch.linspace(-2.0, 2.0, num_targets)
    target[..., 4] = torch.sin(phi)
    target[..., 5] = torch.cos(phi)
    target[..., 6] = torch.log(pt + 2.0)
    target[..., 13] = torch.arange(1, num_targets + 1)
    return target


def test_set_config_populates_decoder_defaults():
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "model": {"type": "heptv2", "output_mode": "set", "heptv2": {}},
            "conv_type": "heptv2",
        }
    )

    assert config.model.set_decoder is not None
    assert config.model.set_decoder.num_slots == 256
    assert config.model.set_decoder.no_object_weight == 1.0
    assert config.model.set_decoder.matcher.dr_scale == 0.1


def test_set_config_rejects_local_attention_for_global_queries():
    with pytest.raises(ValueError, match="local_attention_radius requires"):
        make_config(local_attention_radius=0.4)


def test_set_config_rejects_non_hit_datasets():
    with pytest.raises(ValueError, match="supported only for CLD/CLIC hit datasets"):
        MLPFConfig.model_validate(
            {
                "dataset": "cld",
                "data_dir": "/tmp",
                "model": {"type": "heptv2", "output_mode": "set", "heptv2": {}},
                "conv_type": "heptv2",
            }
        )


def test_set_model_output_axis_is_num_slots():
    config = make_config(num_slots=4)
    model = MLPF(config)
    X = torch.randn(2, 11, config.input_dim)
    X[..., 0] = 1
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[1, 7:] = False

    presence, pid, momentum, pileup = model(X, mask)

    assert presence.shape == (2, 4, 2)
    assert pid.shape == (2, 4, config.num_classes)
    assert momentum.shape == (2, 4, 5)
    assert pileup.shape == (2, 4, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(momentum[..., 2:4], dim=-1), torch.ones(2, 4))


def test_input_conditioned_set_decoder_forward_backward():
    config = make_config(
        num_slots=6,
        query_init="input-conditioned",
        local_attention_radius=0.4,
        tracker_query_fraction=0.5,
        num_layers=3,
        auxiliary_loss_weight=0.25,
    )
    model = MLPF(config)
    X = torch.randn(2, 12, config.input_dim)
    X[..., 0] = torch.tensor([1, 2] * 6)
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 9:] = False

    predictions = model(X, mask)
    assert predictions[2].shape == (2, 6, 5)
    assert len(model.set_decoder.auxiliary_outputs) == 2
    assert all(torch.isfinite(output).all() for prediction in predictions for output in [prediction])

    auxiliary = [output for prediction in model.set_decoder.auxiliary_outputs for output in prediction]
    sum(output.square().mean() for output in [*predictions, *auxiliary]).backward()
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


def test_model_step_applies_configured_cardinality_and_auxiliary_losses():
    from mlpf.model.training import model_step

    config = make_config(
        num_slots=6,
        query_init="input-conditioned",
        local_attention_radius=0.4,
        num_layers=3,
        cardinality_loss_weight=0.05,
        auxiliary_loss_weight=0.25,
    )
    model = MLPF(config)
    X = torch.randn(1, 10, config.input_dim)
    X[..., 0] = torch.tensor([1, 2] * 5)
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    batch = PFBatch(X=X, ytarget_set=make_target_tensor(num_targets=2))

    loss, losses, _, _, _, _ = model_step(batch, model, None, REGRESSION_WEIGHTS)
    loss.backward()

    assert torch.isfinite(loss)
    assert losses["Cardinality"] > 0
    assert losses["Auxiliary"] > 0
    assert model.set_decoder.reference_delta_heads[0].weight.grad is not None


def test_attention_set_model_has_no_unused_elementwise_parameters():
    config = make_attention_config()
    model = MLPF(config)
    X = torch.randn(2, 8, config.input_dim)
    X[..., 0] = 1
    mask = torch.ones(2, 8, dtype=torch.bool)

    predictions = model(X, mask)
    sum(prediction.square().mean() for prediction in predictions).backward()

    assert model.classification_norm is None
    assert model.regression_norm is None
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


def test_hungarian_match_finds_permuted_particles():
    ytarget_tensor = make_target_tensor()
    targets = unpack_target(ytarget_tensor, None)
    predictions = {
        "cls_binary": torch.tensor([[[-5.0, 5.0], [-5.0, 5.0]]]),
        "cls_id_onehot": torch.tensor([[[-5.0, -5.0, 5.0, -5.0, -5.0, -5.0], [-5.0, 5.0, -5.0, -5.0, -5.0, -5.0]]]),
        "pt": targets["pt"].flip(1),
        "eta": targets["eta"].flip(1),
        "sin_phi": targets["sin_phi"].flip(1),
        "cos_phi": targets["cos_phi"].flip(1),
        "energy": targets["energy"].flip(1),
    }

    matches = hungarian_match(targets, predictions, torch.ones(1, 2, dtype=torch.bool))

    slot_indices, target_indices = matches[0]
    assert slot_indices.tolist() == [0, 1]
    assert target_indices.tolist() == [1, 0]


def test_set_loss_is_target_permutation_invariant():
    torch.manual_seed(3)
    target_tensor = make_target_tensor()
    batch = PFBatch(X=torch.ones(1, 5, 15), ytarget_set=target_tensor)
    predictions = {
        "cls_binary": torch.randn(1, 4, 2, requires_grad=True),
        "cls_id_onehot": torch.randn(1, 4, 6, requires_grad=True),
        "pt": torch.randn(1, 4, requires_grad=True),
        "eta": torch.randn(1, 4, requires_grad=True),
        "sin_phi": torch.randn(1, 4, requires_grad=True),
        "cos_phi": torch.randn(1, 4, requires_grad=True),
        "energy": torch.randn(1, 4, requires_grad=True),
    }
    targets = unpack_target(target_tensor, None)
    losses, _ = set_event_loss(targets, predictions, batch.target_mask, REGRESSION_WEIGHTS)

    permutation = torch.tensor([1, 0])
    permuted_tensor = target_tensor[:, permutation]
    permuted_targets = unpack_target(permuted_tensor, None)
    permuted_mask = permuted_tensor[..., 0] != 0
    permuted_losses, _ = set_event_loss(permuted_targets, predictions, permuted_mask, REGRESSION_WEIGHTS)

    for key in losses:
        torch.testing.assert_close(losses[key], permuted_losses[key])
    sum(losses.values()).backward()
    assert predictions["cls_binary"].grad is not None


def test_set_loss_rejects_target_overflow():
    targets_tensor = make_target_tensor(num_targets=2)
    targets = unpack_target(targets_tensor, None)
    predictions = {
        "cls_binary": torch.zeros(1, 1, 2),
        "cls_id_onehot": torch.zeros(1, 1, 6),
        "pt": torch.zeros(1, 1),
        "eta": torch.zeros(1, 1),
        "sin_phi": torch.zeros(1, 1),
        "cos_phi": torch.ones(1, 1),
        "energy": torch.zeros(1, 1),
    }

    with pytest.raises(ValueError, match="2 targets.*1 slots"):
        hungarian_match(targets, predictions, torch.ones(1, 2, dtype=torch.bool))


def test_set_loss_supports_an_event_without_targets():
    target_tensor = torch.zeros(1, 0, 14)
    batch = PFBatch(X=torch.ones(1, 3, 15), ytarget_set=target_tensor)
    targets = unpack_target(target_tensor, None)
    predictions = {
        "cls_binary": torch.randn(1, 4, 2, requires_grad=True),
        "cls_id_onehot": torch.randn(1, 4, 6, requires_grad=True),
        "pt": torch.randn(1, 4, requires_grad=True),
        "eta": torch.randn(1, 4, requires_grad=True),
        "sin_phi": torch.randn(1, 4, requires_grad=True),
        "cos_phi": torch.randn(1, 4, requires_grad=True),
        "energy": torch.randn(1, 4, requires_grad=True),
    }

    losses, matches = set_event_loss(targets, predictions, batch.target_mask, REGRESSION_WEIGHTS)
    loss = sum(losses.values())
    loss.backward()

    assert torch.isfinite(loss)
    assert matches[0][0].numel() == 0
    assert losses["Classification"] == 0
    assert losses["Regression_pt"] == 0
    assert all(prediction.grad is not None for prediction in predictions.values())


def test_cardinality_loss_penalizes_excess_present_slots():
    target_tensor = make_target_tensor(num_targets=2)
    targets = unpack_target(target_tensor, None)
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    predictions = {
        "cls_binary": torch.tensor([[[0.0, 5.0], [0.0, 5.0], [5.0, 0.0], [5.0, 0.0]]]),
        "cls_id_onehot": torch.zeros(1, 4, 6),
        "pt": torch.zeros(1, 4),
        "eta": torch.zeros(1, 4),
        "sin_phi": torch.zeros(1, 4),
        "cos_phi": torch.ones(1, 4),
        "energy": torch.zeros(1, 4),
    }
    calibrated_losses, _ = set_event_loss(
        targets,
        predictions,
        target_mask,
        REGRESSION_WEIGHTS,
        cardinality_loss_weight=1.0,
    )
    predictions["cls_binary"] = torch.tensor([[[0.0, 5.0]] * 4])
    excess_losses, _ = set_event_loss(
        targets,
        predictions,
        target_mask,
        REGRESSION_WEIGHTS,
        cardinality_loss_weight=1.0,
    )

    assert calibrated_losses["Cardinality"] < excess_losses["Cardinality"]


def test_predict_particles_restores_absolute_set_kinematics():
    model = MLPF(make_config(num_slots=3)).eval()
    X = torch.ones(1, 6, 15)
    with torch.no_grad():
        prediction = model.predict_particles(X, torch.ones(1, 6, dtype=torch.bool))

    assert prediction["pt"].shape == (1, 3)
    assert prediction["energy"].shape == (1, 3)
    assert torch.all(prediction["pt"] >= 0)
    assert torch.all(prediction["energy"] >= 0)


def test_set_presence_threshold_controls_inference_selection():
    model = MLPF(make_config(num_slots=2, presence_threshold=0.9)).eval()
    presence = torch.tensor([[[0.0, 2.0], [0.0, 4.0]]])
    pid = torch.full((1, 2, model.num_classes), -5.0)
    pid[0, 0, 2] = 5.0
    pid[0, 1, 3] = 5.0
    momentum = torch.zeros(1, 2, 5)
    momentum[..., 3] = 1.0
    model.forward = lambda _features, _mask: (presence, pid, momentum, torch.zeros_like(presence))

    prediction = model.predict_particles(torch.ones(1, 3, 15), torch.ones(1, 3, dtype=torch.bool))

    assert prediction["cls_id"].tolist() == [[0, 3]]
    assert prediction["pt"][0, 0] == 0
    assert prediction["pt"][0, 1] > 0


def test_set_model_10k_inputs_forward_backward():
    torch.manual_seed(7)
    config = make_config(num_slots=256)
    model = MLPF(config)
    X = torch.randn(1, 10_000, config.input_dim)
    X[..., 0] = 1
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    target_tensor = make_target_tensor(num_targets=100)
    batch = PFBatch(X=X, ytarget_set=target_tensor)

    raw_predictions = model(batch.X, batch.mask)
    predictions = unpack_predictions(raw_predictions)
    targets = unpack_target(batch.ytarget_set, model)
    losses, _ = set_event_loss(targets, predictions, batch.target_mask, REGRESSION_WEIGHTS)
    loss = sum(losses.values())
    loss.backward()

    assert torch.isfinite(loss)
    assert model.set_decoder.queries.grad is not None
    assert torch.isfinite(model.set_decoder.queries.grad).all()
