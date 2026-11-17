from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from mlpf.conf import SetDecoderConfig
from mlpf.model.set_decoder import ParticleSetDecoder


def make_decoder(**options):
    torch.manual_seed(17)
    return ParticleSetDecoder(
        16,
        6,
        SetDecoderConfig(
            num_slots=4,
            num_layers=2,
            num_heads=2,
            query_init="input-conditioned",
            hit_grouping=True,
            hit_grouping_chunk_size=3,
            auxiliary_loss_weight=0.25,
            **options,
        ),
    )


def inputs():
    torch.manual_seed(29)
    memory = torch.randn(2, 9, 16, requires_grad=True)
    features = torch.zeros(2, 9, 15)
    features[..., 0] = 1
    features[..., 1] = torch.arange(9) + 1  # avoid ambiguous top-k proposal ties
    features[:, 5:, 0] = 2
    features[..., 4] = 1
    features[..., 5] = 2
    features[..., 6:9] = torch.randn(2, 9, 3) * 200
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1] = False
    return memory, mask, features


def test_refinement_requires_grouping():
    with pytest.raises(ValueError, match="refinement requires hit_grouping"):
        SetDecoderConfig(hit_grouping_refinement=True)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("attention_mode", ["global-flash", "sectorized"])
def test_final_queries_drive_membership_pooling_and_readout(sparse, attention_mode):
    d = make_decoder(hit_grouping_refinement=True, tracker_momentum_readout=True, hit_grouping_sparse=sparse, attention_mode=attention_mode)
    memory, mask, features = inputs()
    passes = []
    original = d._group_hits

    def capture(slots, *args):
        result = original(slots, *args)
        passes.append((slots, result))
        return result

    with patch.object(d, "_group_hits", side_effect=capture), patch.object(d, "_predict", wraps=d._predict) as predict:
        output = d(memory, mask, features)

    assert len(passes) == 2
    assert len(output.auxiliary_predictions) == 1
    expected_keys = F.normalize(d.hit_grouping_slot(passes[-1][0]).float(), dim=-1)
    torch.testing.assert_close(output.hit_grouping[1], expected_keys)
    assert not torch.allclose(output.hit_grouping[1], passes[0][1][1][1])
    assert predict.call_args_list[0].args[-1] is passes[0][1][3]
    assert predict.call_args_list[-1].args[-1] is passes[-1][1][3]
    assert predict.call_args_list[-1].args[0] is passes[-1][1][0]

    # A grouping objective must train the decoder layers, rather than only the
    # input proposals. Hard pooling itself deliberately has no affinity gradient.
    h, q, bg = output.hit_grouping
    loss = (h[0] @ q[0].T).square().mean() + bg.square().mean()
    loss.backward()
    for layer in d.layers:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters())
    assert torch.isfinite(memory.grad).all()


def test_disabled_refinement_preserves_checkpoint_and_predictions():
    memory, mask, features = inputs()
    base = make_decoder()
    explicit = make_decoder(hit_grouping_refinement=False)
    refined = make_decoder(hit_grouping_refinement=True)
    # Refinement adds no parameters and supports strict loading of old weights.
    refined.load_state_dict(base.state_dict(), strict=True)
    explicit.load_state_dict(base.state_dict(), strict=True)
    with patch.object(base, "_group_hits", wraps=base._group_hits) as group:
        a = base(memory, mask, features)
    b = explicit(memory, mask, features)
    assert group.call_count == 1
    for left, right in zip((*a, *a.hit_grouping), (*b, *b.hit_grouping)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("fit_correction", [False, True])
def test_refined_invalid_fit_discards_initial_fit_anchor(fit_correction):
    d = make_decoder(hit_grouping_refinement=True, tracker_group_readout=True, tracker_fit_correction=fit_correction)
    memory, mask, features = inputs()
    # All queries are tracker-seeded, so a valid initial fit would set an anchor.
    features[..., 0] = 1
    passes = []
    original = d._group_hits

    def group(slots, *args):
        updated, grouping, _, summary = original(slots, *args)
        valid = torch.full(slots.shape[:2], not passes, dtype=torch.bool)
        passes.append(valid)
        return updated, grouping, (torch.full(slots.shape[:2], 4.0), valid), summary

    with patch.object(d, "_group_hits", side_effect=group), patch.object(d, "_predict", wraps=d._predict) as predict:
        output = d(memory, mask, features)
    assert passes[0].all() and not passes[1].any()
    assert not output.tracker_group_fit_valid.any()
    assert not predict.call_args.args[3].any()  # no stale initial tracker anchors
    if fit_correction:
        assert not predict.call_args.args[4][1].any()


def test_refinement_ignores_padded_hits_and_handles_empty_event():
    d = make_decoder(hit_grouping_refinement=True, tracker_momentum_readout=True)
    memory, mask, features = inputs()
    a = d(memory, mask, features)
    b = d(F.pad(memory, (0, 0, 0, 3), value=1000), F.pad(mask, (0, 3)), F.pad(features, (0, 0, 0, 3), value=1000))
    for left, right in zip(a, b):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(a.hit_grouping[1], b.hit_grouping[1])
