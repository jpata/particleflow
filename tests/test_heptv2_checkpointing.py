"""Checkpoint replay must preserve all outputs, dropout RNG and gradients."""

import copy

import pytest
import torch

import mlpf.model.mlpf as mlpf_module
from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF


def make_model(checkpointing=False, dropout=0.2, shared=False):
    config = MLPFConfig.model_validate(
        {
            "dataset": "cld_hits",
            "data_dir": "/tmp",
            "conv_type": "heptv2",
            "model": {
                "type": "heptv2",
                "output_mode": "set",
                "input_encoding": "split",
                "task_queries": False,
                "heptv2": {
                    "num_convs": 2,
                    "embedding_dim": 32,
                    "width": 32,
                    "num_heads": 2,
                    "block_size": 8,
                    "n_hashes": 3,
                    "dropout_ff": dropout,
                    "activation_checkpointing": checkpointing,
                    "share_position_embedding": shared,
                },
                "hit_feature_engineering": {"enabled": False},
                "set_decoder": {
                    "num_slots": 4,
                    "num_layers": 2,
                    "num_heads": 2,
                    "attention_mode": "global-flash",
                    "query_init": "input-conditioned",
                    "hit_grouping": True,
                },
            },
        }
    )
    return MLPF(config)


def make_inputs(device):
    x = torch.randn(3, 24, 15, device=device)
    mask = torch.arange(24, device=device)[None] < torch.tensor([19, 13, 5], device=device)[:, None]
    x[..., 0] = torch.arange(24, device=device) % 2 + 1
    x[..., 1] = x[..., 1].exp()
    x[..., 5] = x[..., 5].exp()
    x[..., 10] = torch.where(x[..., 0] == 1, 3, 1)
    x[~mask] = 0
    return x, mask


def output_tensors(output):
    return [*output, *(t for aux in output.auxiliary_predictions for t in aux), *output.hit_grouping]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required"))])
@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_checkpoint_outputs_gradients_and_rng(device, freeze_encoder):
    torch.manual_seed(123)
    baseline = make_model().to(device).train()
    replayed = copy.deepcopy(baseline)
    replayed.config.heptv2.activation_checkpointing = True
    # Checkpointing adds no parameters or buffers; old checkpoints load strictly.
    replayed.load_state_dict(baseline.state_dict(), strict=True)
    if freeze_encoder:
        baseline.nn0.requires_grad_(False)
        replayed.nn0.requires_grad_(False)
    x, mask = make_inputs(device)

    def run(model):
        torch.manual_seed(42)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            outputs = output_tensors(model(x, mask))
            loss = sum(t.float().square().mean() for t in outputs)
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return [t.detach() for t in outputs], gradients, torch.rand(10, device=device)

    expected, expected_grad, expected_rng = run(baseline)
    actual, actual_grad, actual_rng = run(replayed)
    assert len(actual) == len(expected)
    assert actual_grad.keys() == expected_grad.keys()
    assert any(n.startswith("backbone.0") for n in actual_grad)
    assert any(n.startswith("backbone.1") for n in actual_grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    for name in actual_grad:
        tolerance = {"rtol": 0.03, "atol": 0.004} if device == "cuda" else {"rtol": 1e-4, "atol": 1e-5}
        torch.testing.assert_close(actual_grad[name], expected_grad[name], **tolerance)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize("training,grad_enabled,enabled", [(False, True, True), (True, False, True), (True, True, False)])
def test_checkpoint_bypassed_outside_opt_in_training(monkeypatch, training, grad_enabled, enabled):
    model = make_model(checkpointing=enabled).train(training)
    x, mask = make_inputs("cpu")

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError("checkpoint must not run here")

    monkeypatch.setattr(mlpf_module, "checkpoint", unexpected_checkpoint)
    with torch.set_grad_enabled(grad_enabled):
        outputs = model(x, mask)
    assert torch.isfinite(outputs[2]).all()


def test_checkpoint_reuses_coordinates_after_backbone_transform(monkeypatch):
    import mlpf.model.heptv2 as heptv2

    model = make_model(checkpointing=True)
    x, mask = make_inputs("cpu")
    transformed = x.clone()
    transformed[..., 2] += 1

    class TrackTransform(torch.nn.Module):
        def forward(self, features, source_id):
            torch.testing.assert_close(features[..., 2:5], x[..., 2:5])
            return transformed

    def convert(features):
        calls.append(features)
        return reference_conversion(features)

    calls = []
    reference_conversion = heptv2.fixed_point_coordinates
    monkeypatch.setattr(mlpf_module, "fixed_point_coordinates", convert)
    monkeypatch.setattr(heptv2, "fixed_point_coordinates", convert)
    model.track_calo_entry = TrackTransform()

    model.encode_backbone(x, mask).square().mean().backward()
    assert len(calls) == 1
    assert calls[0] is transformed


@pytest.mark.parametrize("packed_checkpoint", [False, True])
def test_shared_position_embedding_matches_tied_layers(packed_checkpoint):
    torch.manual_seed(12)
    reference = make_model(dropout=0.0)
    torch.manual_seed(12)
    shared = make_model(checkpointing=packed_checkpoint, dropout=0.0, shared=True)

    # The paired scenario starts all remaining parameters from the same values.
    assert shared.backbone[1].pe_func is None
    reference_state = reference.state_dict()
    for name, value in shared.state_dict().items():
        torch.testing.assert_close(value, reference_state[name], atol=0, rtol=0)

    reference.backbone[1].pe_func = reference.backbone[0].pe_func
    reference.config.heptv2.pack_padding = packed_checkpoint
    shared.config.heptv2.pack_padding = packed_checkpoint
    calls = []

    def observe_embedding(_module, _args, _output):
        calls.append(1)

    x, mask = make_inputs("cpu")
    expected = reference.encode_backbone(x, mask)
    expected.square().mean().backward()
    with shared.backbone[0].pe_func.register_forward_hook(observe_embedding):
        actual = shared.encode_backbone(x, mask)
        actual.square().mean().backward()
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected)
    for actual_param, expected_param in zip(shared.backbone[0].pe_func.parameters(), reference.backbone[0].pe_func.parameters(), strict=True):
        assert actual_param.grad is not None
        torch.testing.assert_close(actual_param.grad, expected_param.grad)
