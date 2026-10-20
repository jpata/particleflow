import pytest
import torch

from mlpf.conf import MLPFConfig
from mlpf.model.PFDataset import PFBatch
from mlpf.model.mlpf import MLPF
from mlpf.model.set_decoder import _flash_multihead_attention, _phi_sector
from mlpf.model.set_losses import hungarian_match, set_event_loss
from mlpf.model.set_losses import hit_grouping_loss
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


def test_set_config_accepts_colliderml_hits():
    config = MLPFConfig.model_validate(
        {
            "dataset": "colliderml_hits",
            "data_dir": "/tmp",
            "model": {"type": "heptv2", "output_mode": "set", "heptv2": {}},
            "conv_type": "heptv2",
        }
    )

    assert config.model.set_decoder is not None


def test_set_config_rejects_local_attention_for_global_queries():
    with pytest.raises(ValueError, match="local_attention_radius requires"):
        make_config(local_attention_radius=0.4)


def test_set_config_rejects_sectorized_attention_for_global_queries():
    with pytest.raises(ValueError, match="sectorized attention requires"):
        make_config(attention_mode="sectorized")


def test_hit_grouping_uses_predicted_membership_and_event_local_truth():
    config = make_config(
        num_slots=4, query_init="input-conditioned", hit_grouping=True,
        hit_grouping_dim=8, hit_grouping_chunk_size=2,
    )
    model = MLPF(config)
    features = torch.zeros(1, 6, config.input_dim)
    features[0, :5, 0] = torch.tensor([1, 1, 2, 2, 2])
    features[0, :5, 1] = 1.0
    features[0, :5, 5] = 1.0
    features[0, :5, 3] = 0.0
    features[0, :5, 4] = 1.0
    target = make_target_tensor(num_targets=2)
    hit_target = torch.zeros(1, 6, 14)
    hit_target[0, :5, 13] = torch.tensor([1, 1, 1, 2, 0])
    batch = PFBatch(X=features, ytarget=hit_target, ytarget_set=target)
    output = model(features, batch.mask)
    assert output.hit_grouping is not None
    assert output[2].shape == (1, 4, 5)
    grouping_loss = hit_grouping_loss(
        output.hit_grouping, batch,
        [(torch.tensor([3, 1]), torch.tensor([0, 1]))], chunk_size=2,
    )
    assert torch.isfinite(grouping_loss)
    grouping_loss.backward()
    assert model.set_decoder.hit_grouping_hit.weight.grad is not None
    assert model.set_decoder.hit_grouping_slot.weight.grad is not None
    assert model.set_decoder.hit_grouping_background.weight.grad is not None


@pytest.mark.parametrize("fit_weight", [0.0, 0.1])
def test_hit_grouping_is_included_in_set_training_loss(fit_weight):
    from mlpf.model.training import model_step

    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=True,
                         tracker_geometry_loss_weight=fit_weight, tracker_curvature_loss_weight=fit_weight)
    model = MLPF(config)
    features = torch.zeros(1, 5, config.input_dim)
    features[..., 0] = torch.tensor([1, 1, 2, 2, 2])
    features[..., 1] = 1
    features[..., 4] = 1
    features[..., 5] = 1
    target = make_target_tensor(num_targets=2)
    hit_target = torch.zeros(1, 5, 14)
    hit_target[0, :, 13] = torch.tensor([1, 1, 2, 2, 0])
    batch = PFBatch(X=features, ytarget=hit_target, ytarget_set=target)
    loss, losses, _, _, _, _ = model_step(batch, model, None, REGRESSION_WEIGHTS)
    assert losses["HitGrouping"] > 0
    assert ("TrackerGeometry" in losses) == (fit_weight > 0)
    assert ("TrackerCurvature" in losses) == (fit_weight > 0)
    loss.backward()
    assert model.set_decoder.hit_grouping_projection.weight.grad is not None


def test_hit_grouping_labels_are_local_to_each_event():
    features = torch.ones(2, 3, 15)
    targets = make_target_tensor(batch_size=2, num_targets=2)
    hit_targets = torch.zeros(2, 3, 14)
    hit_targets[..., 13] = torch.tensor([[1, 2, 0], [1, 2, 0]])
    batch = PFBatch(X=features, ytarget=hit_targets, ytarget_set=targets)
    hit_keys = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]]] * 2)
    slot_keys = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
    background = torch.tensor([[[-10.], [-10.], [10.]]] * 2)
    matches = [
        (torch.tensor([0, 1]), torch.tensor([0, 1])),
        (torch.tensor([1, 0]), torch.tensor([0, 1])),
    ]
    loss = hit_grouping_loss((hit_keys, slot_keys, background), batch, matches, chunk_size=2)
    assert loss < 0.01


def test_phi_sectors_wrap_at_periodic_boundary():
    sectors = _phi_sector(torch.tensor([-torch.pi, torch.pi, -torch.pi + 0.01, torch.pi - 0.01]), 32)
    assert sectors.tolist() == [0, 0, 0, 31]


def test_global_set_attention_matches_unmasked_attention_without_dense_pairwise_mask(monkeypatch):
    module = torch.nn.MultiheadAttention(16, 2, batch_first=True, dropout=0.0)
    queries = torch.randn(2, 5, 16, requires_grad=True)
    memory = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 6:] = False
    expected = torch.stack([
        module(queries[index:index + 1], memory[index:index + 1, mask[index]], memory[index:index + 1, mask[index]], need_weights=False)[0][0]
        for index in range(2)
    ])

    def reject_dense_attention(*args, **kwargs):
        raise AssertionError("Set decoder called dense MultiheadAttention.forward")

    monkeypatch.setattr(torch.nn.MultiheadAttention, "forward", reject_dense_attention)
    actual = _flash_multihead_attention(module, queries, memory, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    assert queries.grad is not None and memory.grad is not None


def test_global_set_attention_returns_zero_for_event_without_hits():
    module = torch.nn.MultiheadAttention(16, 2, batch_first=True)
    queries = torch.randn(1, 5, 16)
    memory = torch.randn(1, 9, 16)
    mask = torch.zeros(1, 9, dtype=torch.bool)

    output = _flash_multihead_attention(module, queries, memory, mask)

    torch.testing.assert_close(output, torch.zeros_like(output))


def test_heptv2_sectorized_set_decoder_forward_backward():
    config = make_config(
        num_slots=12,
        query_init="input-conditioned",
        attention_mode="sectorized",
        num_sectors=8,
        sector_neighbors=1,
        num_layers=2,
        proposal_mode="grid-diverse",
        use_aggregate_anchors=True,
    )
    config.model.heptv2.num_convs = 1
    model = MLPF(config)
    features = torch.randn(2, 16, config.input_dim)
    features[..., 0] = torch.tensor([1, 2] * 8)
    features[..., 1] = features[..., 1].abs() + 0.1
    features[..., 5] = features[..., 5].abs() + 0.1
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[1, 10:] = False

    predictions = model(features, mask)
    assert predictions[2].shape == (2, 12, 5)
    assert all(torch.isfinite(output).all() for output in predictions)
    sum(output.square().mean() for output in predictions).backward()
    assert model.set_decoder.layers[0].cross_attention.in_proj_weight.grad is not None
    assert model.set_decoder.layers[0].self_attention.in_proj_weight.grad is not None
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


@pytest.mark.parametrize("override", [{"proposal_mode": "grid-diverse"}, {"use_aggregate_anchors": True}])
def test_set_config_rejects_object_formation_for_global_queries(override):
    with pytest.raises(ValueError, match="query_init='input-conditioned'"):
        make_config(**override)


def test_set_config_rejects_non_hit_datasets():
    with pytest.raises(ValueError, match="supported only for the CLD/CLIC/ColliderML hit datasets"):
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
    assert len(predictions.auxiliary_predictions) == 2
    assert not hasattr(model.set_decoder, "auxiliary_outputs")
    assert all(torch.isfinite(output).all() for prediction in predictions for output in [prediction])

    auxiliary = [output for prediction in predictions.auxiliary_predictions for output in prediction]
    sum(output.square().mean() for output in [*predictions, *auxiliary]).backward()
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


def test_grid_diverse_queries_choose_one_representative_per_angular_cell():
    config = make_config(
        num_slots=4,
        query_init="input-conditioned",
        local_attention_radius=0.4,
        tracker_query_fraction=0.5,
        proposal_mode="grid-diverse",
        proposal_grid_size=0.2,
    )
    decoder = MLPF(config).set_decoder
    features = torch.zeros(6, config.input_dim)
    features[:, 0] = torch.tensor([1, 1, 1, 2, 2, 2])
    features[:, 2] = torch.tensor([0.01, 0.02, 0.45, 0.01, 0.02, 0.45])
    phi = torch.tensor([0.01, 0.02, 0.45, 0.01, 0.02, 0.45])
    features[:, 3] = torch.sin(phi)
    features[:, 4] = torch.cos(phi)
    features[:, 5] = torch.tensor([0.0, 0.0, 0.0, 1.0, 3.0, 2.0])
    features[:, 6] = torch.tensor([20.0, 10.0, 30.0, 1000.0, 1000.0, 1000.0])

    summaries, scores, representatives = decoder._angular_grid_summary(features, torch.ones(6, dtype=torch.bool), 0.2)

    assert representatives.tolist() == [False, True, True, False, True, True]
    assert representatives.sum() == 4
    torch.testing.assert_close(summaries[3, 2], torch.log1p(torch.tensor(4.0)))
    torch.testing.assert_close(scores[3], scores[4])


def test_angular_grid_keeps_detector_types_separate_at_bin_boundaries():
    features = torch.zeros(2, 9)
    features[:, 0] = torch.tensor([1.0, 2.0])
    features[:, 2] = torch.tensor([10.0, -10.0])
    features[:, 4:6] = 1.0

    decoder = MLPF(make_config()).set_decoder
    summaries, _, representatives = decoder._angular_grid_summary(features, torch.ones(2, dtype=torch.bool), 0.1)

    assert representatives.tolist() == [True, True]
    torch.testing.assert_close(summaries[:, 0], torch.log1p(torch.ones(2)))


def test_grid_diverse_queries_use_remaining_cell_representatives_before_duplicate_hits():
    config = make_config(num_slots=4, query_init="input-conditioned", proposal_mode="grid-diverse", tracker_query_fraction=0.5)
    decoder = MLPF(config).set_decoder
    features = torch.zeros(5, config.input_dim)
    features[:, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0])
    features[:, 2] = torch.tensor([0.01, 0.31, 0.61, 1.01, 1.02])
    features[:, 4] = 1.0
    features[:, 5] = torch.tensor([0.0, 0.0, 0.0, 10.0, 20.0])
    memory = torch.zeros(1, 5, decoder.queries.shape[-1])

    _, references, reference_mask, _, _ = decoder._input_conditioned_queries(memory, torch.ones(1, 5, dtype=torch.bool), features.unsqueeze(0))

    assert reference_mask.all()
    torch.testing.assert_close(references[0, :, 0].sort().values, torch.tensor([0.01, 0.31, 0.61, 1.02]))


def test_grid_diverse_queries_prioritize_distinct_high_energy_cells():
    config = make_config(num_slots=2, query_init="input-conditioned", proposal_mode="grid-diverse", tracker_query_fraction=0.0)
    decoder = MLPF(config).set_decoder
    features = torch.zeros(1, 4, config.input_dim)
    features[0, :, 0] = 2
    features[0, :, 2] = torch.tensor([0.01, 0.02, 0.51, 1.01])
    features[0, :, 4] = 1
    features[0, :, 5] = torch.tensor([1.0, 10.0, 5.0, 0.1])
    memory = torch.zeros(1, 4, decoder.queries.shape[-1])

    _, references, reference_mask, _, _ = decoder._input_conditioned_queries(memory, torch.ones(1, 4, dtype=torch.bool), features)

    assert reference_mask.all()
    torch.testing.assert_close(references[0, :, 0].sort().values, torch.tensor([0.02, 0.51]))


def test_aggregate_anchor_set_decoder_forward_backward():
    config = make_config(
        num_slots=6,
        query_init="input-conditioned",
        local_attention_radius=0.4,
        proposal_mode="grid-diverse",
        use_aggregate_anchors=True,
        num_layers=2,
    )
    model = MLPF(config)
    X = torch.randn(2, 12, config.input_dim)
    X[..., 0] = torch.tensor([1, 2] * 6)
    X[..., 1] = X[..., 1].abs() + 0.1
    X[..., 5] = X[..., 5].abs() + 0.1
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 9:] = False

    predictions = model(X, mask)
    sum(output.square().mean() for output in predictions).backward()

    assert all(torch.isfinite(output).all() for output in predictions)
    assert model.set_decoder.aggregate_projection[0].weight.grad is not None
    assert [name for name, parameter in model.named_parameters() if parameter.grad is None] == []


def test_auxiliary_outputs_belong_to_their_forward_pass():
    config = make_config(num_layers=3, auxiliary_loss_weight=0.25)
    model = MLPF(config)
    mask = torch.ones(1, 8, dtype=torch.bool)
    first_input = torch.randn(1, 8, config.input_dim)
    second_input = torch.randn(1, 8, config.input_dim)
    for features in (first_input, second_input):
        features[..., 0] = 1
        features[..., 1] = features[..., 1].abs() + 0.1
        features[..., 5] = features[..., 5].abs() + 0.1

    first_output = model(first_input, mask)
    second_output = model(second_input, mask)

    assert len(first_output.auxiliary_predictions) == 2
    assert len(second_output.auxiliary_predictions) == 2
    assert first_output.auxiliary_predictions is not second_output.auxiliary_predictions
    assert any(
        not torch.equal(first, second)
        for first_layer, second_layer in zip(
            first_output.auxiliary_predictions,
            second_output.auxiliary_predictions,
        )
        for first, second in zip(first_layer, second_layer)
    )


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


def test_hungarian_match_reports_nonfinite_prediction():
    targets = unpack_target(make_target_tensor(num_targets=1), None)
    predictions = {
        "cls_binary": torch.zeros(1, 1, 2),
        "cls_id_onehot": torch.zeros(1, 1, 6),
        "pt": torch.tensor([[float("nan")]]),
        "eta": targets["eta"].clone(),
        "sin_phi": targets["sin_phi"].clone(),
        "cos_phi": targets["cos_phi"].clone(),
        "energy": targets["energy"].clone(),
    }

    with pytest.raises(ValueError, match="event 0.*slot 0, target 0.*prediction fields: pt: count=1.*target fields: none"):
        hungarian_match(targets, predictions, torch.ones(1, 1, dtype=torch.bool))


def test_hungarian_match_reports_nonfinite_target_with_zero_energy_weight():
    targets = unpack_target(make_target_tensor(num_targets=1), None)
    targets["energy"][0, 0] = float("nan")
    predictions = {
        "cls_binary": torch.zeros(1, 1, 2),
        "cls_id_onehot": torch.zeros(1, 1, 6),
        "pt": targets["pt"].clone(),
        "eta": targets["eta"].clone(),
        "sin_phi": targets["sin_phi"].clone(),
        "cos_phi": targets["cos_phi"].clone(),
        "energy": torch.zeros(1, 1),
    }

    with pytest.raises(ValueError, match="event 0.*slot 0, target 0.*prediction fields: none; target fields: energy: count=1"):
        hungarian_match(targets, predictions, torch.ones(1, 1, dtype=torch.bool))


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


def test_huber_momentum_caps_large_log_residuals_without_changing_geometry():
    targets = unpack_target(make_target_tensor(num_targets=1), None)
    predictions = {
        "cls_binary": torch.tensor([[[0.0, 1.0]]]),
        "cls_id_onehot": torch.zeros(1, 1, 6),
        "pt": targets["pt"] + 1.0,
        "eta": targets["eta"] + 0.1,
        "sin_phi": targets["sin_phi"].clone(),
        "cos_phi": targets["cos_phi"].clone(),
        "energy": targets["energy"] + 1.0,
    }
    target_mask = torch.ones(1, 1, dtype=torch.bool)

    mse, _ = set_event_loss(targets, predictions, target_mask, REGRESSION_WEIGHTS)
    huber, _ = set_event_loss(
        targets, predictions, target_mask, REGRESSION_WEIGHTS,
        momentum_loss="huber", momentum_huber_delta=0.2,
    )

    torch.testing.assert_close(huber["Regression_pt"], 0.36 * mse["Regression_pt"])
    torch.testing.assert_close(huber["Regression_energy"], 0.36 * mse["Regression_energy"])
    torch.testing.assert_close(huber["Regression_eta"], mse["Regression_eta"])


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


@pytest.mark.parametrize("grouping", [False, True])
def test_evaluate_reconstruction_diagnostics_respect_batch_limit(grouping):
    from mlpf.model.training import evaluate

    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=grouping, hit_grouping_dim=8)
    config.validation_diagnostics_batches = 1
    config.make_plots = False
    model = MLPF(config)
    features = torch.zeros(1, 5, config.input_dim)
    features[..., 0] = torch.tensor([1, 1, 2, 2, 2])
    features[..., 1] = 1
    features[..., 4] = 1
    features[..., 5] = 1
    hit_targets = torch.zeros(1, 5, 14)
    hit_targets[0, :, 13] = torch.tensor([1, 1, 2, 2, 0])
    batch = PFBatch(X=features, ytarget=hit_targets, ytarget_set=make_target_tensor())
    metrics = evaluate("cpu", 1, model, [batch, batch], 1, config, device_type="cpu")
    assert metrics["metrics/particle/momentum/all/event/samples"] == 1
    assert metrics["metrics/particle/momentum/all/target_samples"] == 2
    if grouping:
        assert metrics["metrics/hit_grouping/tracker/hit_samples"] == 2
        assert metrics["metrics/hit_grouping/calorimeter/hit_samples"] == 3
    else:
        assert not any(key.startswith("metrics/hit_grouping/") for key in metrics)
    config.validation_diagnostics_batches = 0
    ordinary = evaluate("cpu", 1, model, [batch, batch], 1, config, device_type="cpu")
    assert not any("momentum/" in key or "hit_grouping/" in key for key in ordinary)
    assert all(metrics[key] == pytest.approx(value) for key, value in ordinary.items())


@pytest.mark.parametrize(
    "overrides",
    [
        {"hit_grouping_detector_balanced": True},
        {"hit_grouping_tracker_particle_balanced": True},
        {"tracker_overlap_loss_weight": 0.05},
        {"hit_grouping_tracker_particle_balanced": True, "tracker_overlap_loss_weight": 0.05},
        {"tracker_group_readout": True},
        {"tracker_group_readout": True, "tracker_fit_correction": True},
        {"event_consistency_loss_weight": 0.1},
    ],
)
def test_grouped_tracking_variants_forward_backward(overrides):
    from mlpf.model.training import model_step

    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=True, **overrides)
    model = MLPF(config)
    features = torch.ones(1, 8, 15)
    features[0, :, 0] = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    targets = make_target_tensor()
    hit_targets = torch.zeros(1, 8, 14)
    hit_targets[0, :, 13] = torch.tensor([1, 1, 1, 0, 2, 2, 2, 0])
    batch = PFBatch(X=features, ytarget=hit_targets, ytarget_set=targets)
    loss, values, _, _, _, _ = model_step(batch, model, None, REGRESSION_WEIGHTS)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.set_decoder.scale_head.weight.grad is not None
    if overrides.get("tracker_overlap_loss_weight"):
        assert values["TrackerOverlap"] > 0
        assert values["TrackerOverlapCount"] == 1
    if overrides.get("tracker_fit_correction"):
        assert model.set_decoder.tracker_pt_correction.weight.grad is not None
    if overrides.get("tracker_group_readout"):
        assert model.set_decoder.tracker_group_projection.weight.grad is not None
        from mlpf.model.training import evaluate
        config.validation_diagnostics_batches = 1
        config.make_plots = False
        metrics = evaluate("cpu", 1, model, [batch], 1, config, device_type="cpu")
        assert "metrics/set_slot/tracker_fit_fraction" in metrics
        assert "metrics/set_slot/tracker_fits_per_event" in metrics
    if overrides.get("event_consistency_loss_weight"):
        assert values["EventConsistency"] > 0


def test_tracker_readout_supplies_scale_prior_and_invalid_fit_falls_back(monkeypatch):
    import mlpf.model.set_decoder as decoder

    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=True, tracker_group_readout=True)
    model = MLPF(config)
    torch.nn.init.zeros_(model.set_decoder.scale_head.weight)
    torch.nn.init.zeros_(model.set_decoder.scale_head.bias)

    def fit(features, groups, num_slots, magnetic_field):
        return torch.zeros(num_slots, 9), torch.full((num_slots,), 2.0), torch.tensor([True, False, True, False])

    monkeypatch.setattr(decoder, "tracker_group_geometry", fit)
    features = torch.ones(1, 8, 15)
    output = model(features, torch.ones(1, 8, dtype=torch.bool))
    assert output.predictions[2][0, :, 0].tolist() == [2.0, 0.0, 2.0, 0.0]
    assert output.tracker_group_fit_valid.tolist() == [[True, False, True, False]]


def test_group_readout_and_balancing_require_grouping():
    for kwargs in ({"tracker_group_readout": True}, {"hit_grouping_detector_balanced": True}):
        with pytest.raises(ValueError, match="require hit_grouping"):
            make_config(**kwargs)


def test_bounded_tracker_correction_preserves_fit_fallback_and_gradients():
    config = make_config(query_init="input-conditioned", hit_grouping=True,
                         tracker_group_readout=True, tracker_fit_correction=True)
    decoder = MLPF(config).set_decoder
    slots = torch.randn(1, 3, decoder.output_norm.normalized_shape[0], requires_grad=True)
    references = torch.zeros(1, 3, 2)
    anchors = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    fits = (torch.tensor([[2.123456, 3., 4.]]), torch.tensor([[True, False, True]]))
    with torch.no_grad():
        decoder.pid_head.weight.zero_()
        decoder.pid_head.bias.zero_()
        decoder.pid_head.bias[1] = 10
        decoder.scale_head.weight.zero_()
        decoder.scale_head.bias.fill_(.7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ordinary = decoder._predict(slots, references, anchors, mask)[2]
        fitted = decoder._predict(slots, references, anchors, mask, fits)[2]
    assert fitted.dtype == torch.float32
    torch.testing.assert_close(fitted[0, [0, 2], 0], fits[0][0, [0, 2]], rtol=0, atol=1e-7)
    torch.testing.assert_close(fitted[..., 4], ordinary[..., 4])
    torch.testing.assert_close(fitted[:, 1], ordinary[:, 1])
    fitted[..., 0].sum().backward()
    assert decoder.tracker_pt_correction.weight.grad is not None
    assert decoder.tracker_pt_correction.bias.grad.abs().sum() > 0
    assert decoder.scale_head.bias.grad[0] > 0  # invalid fit fallback learns
    assert torch.isfinite(slots.grad).all()
    with torch.no_grad():
        decoder.tracker_pt_correction.bias.fill_(100)
    upper = decoder._predict(slots, references, anchors, mask, fits)[2]
    torch.testing.assert_close(upper[0, [0, 2], 0] - fits[0][0, [0, 2]],
                               torch.full((2,), config.model.set_decoder.tracker_fit_max_log_correction))
    with torch.no_grad():
        decoder.tracker_pt_correction.bias.fill_(-100)
    lower = decoder._predict(slots, references, anchors, mask, fits)[2]
    torch.testing.assert_close(lower[0, [0, 2], 0] - fits[0][0, [0, 2]],
                               torch.full((2,), -config.model.set_decoder.tracker_fit_max_log_correction))
    for pid in (2, 3):
        with torch.no_grad():
            decoder.pid_head.bias.zero_()
            decoder.pid_head.bias[pid] = 10
        fallback = decoder._predict(slots, references, anchors, mask, fits)[2]
        expected = decoder._predict(slots, references, anchors, mask)[2]
        torch.testing.assert_close(fallback, expected)


def test_bounded_tracker_correction_is_wired_into_forward(monkeypatch):
    import mlpf.model.set_decoder as module
    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=True,
                         tracker_group_readout=True, tracker_fit_correction=True)
    model = MLPF(config)
    with torch.no_grad():
        model.set_decoder.pid_head.weight.zero_()
        model.set_decoder.pid_head.bias.zero_()
        model.set_decoder.pid_head.bias[1] = 10
    def fit(features, groups, num_slots, magnetic_field):
        return torch.zeros(num_slots, 9), torch.full((num_slots,), 2.123456), torch.ones(num_slots, dtype=torch.bool)
    monkeypatch.setattr(module, "tracker_group_geometry", fit)
    output = model(torch.ones(1, 8, 15), torch.ones(1, 8, dtype=torch.bool))
    torch.testing.assert_close(output.predictions[2][..., 0], torch.full((1, 4), 2.123456))
    output.predictions[2][..., 0].sum().backward()
    assert model.set_decoder.tracker_pt_correction.bias.grad.abs().sum() > 0


def test_bounded_tracker_correction_configuration():
    with pytest.raises(ValueError, match="requires tracker_group_readout"):
        make_config(tracker_fit_correction=True)
    with pytest.raises(ValueError):
        make_config(tracker_fit_max_log_correction=0)


@pytest.mark.parametrize("grouping", [False, True])
def test_particle_metrics_disabled_preserves_set_diagnostics(grouping, monkeypatch):
    from mlpf.model.training import evaluate

    config = make_config(num_slots=4, query_init="input-conditioned", hit_grouping=grouping, hit_grouping_dim=8)
    config.validation_diagnostics_batches = 1
    config.make_plots = False
    model = MLPF(config)
    features = torch.zeros(1, 5, config.input_dim)
    features[..., 0] = torch.tensor([1, 1, 2, 2, 2])
    features[..., 1] = 1
    features[..., 4] = 1
    features[..., 5] = 1
    hit_targets = torch.zeros(1, 5, 14)
    hit_targets[0, :, 13] = torch.tensor([1, 1, 2, 2, 0])
    batch = PFBatch(X=features, ytarget=hit_targets, ytarget_set=make_target_tensor())
    config.validation_particle_metrics = False
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Disabled particle matching was called")
    monkeypatch.setattr("mlpf.model.training.compute_validation_particle_metrics", fail_if_called)
    metrics = evaluate("cpu", 1, model, [batch], 1, config, device_type="cpu")
    assert not any(key.startswith("metrics/particle/") for key in metrics)
    assert any(key.startswith("metrics/set_slot/") for key in metrics)
    assert any(key.startswith("metrics/hit_grouping/") for key in metrics) == grouping
