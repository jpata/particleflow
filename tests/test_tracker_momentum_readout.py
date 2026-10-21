import pytest
import torch
from mlpf.conf import SetDecoderConfig
from mlpf.model.set_decoder import ParticleSetDecoder
from mlpf.model.tracker_momentum_readout import association_weights, pool_hit_associations


def inputs():
    torch.manual_seed(31)
    memory = torch.randn(2, 9, 16, requires_grad=True)
    features = torch.zeros(2, 9, 15)
    features[..., 0] = 1
    features[:, 5:, 0] = 2
    features[..., 4] = 1
    features[..., 5] = 2
    features[..., 6:9] = torch.randn(2, 9, 3) * 200
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1] = False
    return memory, mask, features


def decoder(**kwargs):
    return ParticleSetDecoder(
        16,
        6,
        SetDecoderConfig(
            num_slots=4, num_layers=2, num_heads=2, query_init="input-conditioned", hit_grouping=True, hit_grouping_chunk_size=3, **kwargs
        ),
    )


def test_sparse_weights_match_truncated_dense_reference_and_gradients():
    torch.manual_seed(1)
    hits = torch.randn(7, 5, requires_grad=True)
    slots = torch.randn(6, 5, requires_grad=True)
    bg = torch.randn(7, 1, requires_grad=True)
    dest, weights = association_weights(hits, slots, bg, 3, 0.7, True)
    dense = 10 * hits @ slots.T
    expected = torch.cat((dense.gather(1, dest), bg), -1).div(0.7).softmax(-1)[:, :-1]
    torch.testing.assert_close(weights, expected)
    assert (weights.sum(-1) <= 1 + 1e-6).all()
    grad = torch.autograd.grad((weights.square()).sum(), (hits, slots, bg), retain_graph=True)
    ref = torch.autograd.grad(expected.square().sum(), (hits, slots, bg))
    for a, b in zip(grad, ref):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    _, rejected = association_weights(hits, slots, torch.full_like(bg, 1e4), 3, 1.0, True)
    assert rejected.sum() == 0


def test_pooling_is_chunk_order_and_padding_invariant():
    m, mask, x = inputs()
    d = decoder(hit_grouping_sparse=True, tracker_momentum_readout=True)
    grouping = (
        torch.nn.functional.normalize(d.hit_grouping_hit(m).float(), dim=-1),
        torch.nn.functional.normalize(d.hit_grouping_slot(d.queries.expand(2, -1, -1)).float(), dim=-1),
        d.hit_grouping_background(m).float(),
    )

    def pool(mem, msk, feat, g, chunk):
        return pool_hit_associations(mem, msk, feat, g, 4, chunk, 2, 1.0, True, True)

    a = pool(m, mask, x, grouping, 2)
    b = pool(m, mask, x, grouping, 20)
    perm = torch.randperm(9)
    c = pool(m[:, perm], mask[:, perm], x[:, perm], (grouping[0][:, perm], grouping[1], grouping[2][:, perm]), 3)
    for aa, bb, cc in zip(a, b, c):
        torch.testing.assert_close(aa, bb)
        torch.testing.assert_close(aa, cc)
        assert aa[1].abs().sum() == 0
    # Calorimeter values cannot affect the tracker-only summary for fixed affinities.
    mx = m.clone()
    mx[:, 5:] += 100
    xx = x.clone()
    xx[:, 5:, 6:9] += 1000
    torch.testing.assert_close(a[1], pool(mx, mask, xx, grouping, 2)[1])


@pytest.mark.parametrize("sparse", [False, True])
def test_momentum_gradients_reach_selected_affinities_only_when_sparse(sparse):
    m, mask, x = inputs()
    d = decoder(hit_grouping_sparse=sparse, tracker_momentum_readout=True)
    # Move past the deliberately zero-initialized residual head.
    torch.nn.init.normal_(d.tracker_momentum_head.output.weight, std=0.1)
    slots = d.queries.expand(2, -1, -1)
    _, _, _, summary = d._group_hits(slots, m, mask, x)
    d.tracker_momentum_head(summary).square().sum().backward()
    for layer in [d.hit_grouping_hit, d.hit_grouping_slot, d.hit_grouping_background]:
        grad = layer.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert bool(grad.abs().sum() > 0) == sparse
    assert torch.isfinite(m.grad).all() and m.grad[0, :5].abs().sum() > 0


def test_readout_zero_initialization_neutral_fallback_and_empty_tracker():
    m, mask, x = inputs()
    d = decoder(tracker_momentum_readout=True)
    slots, refs, _, anchors, anchor_mask = d._input_conditioned_queries(m, mask, x)
    slots, _, _, summary = d._group_hits(slots, m, mask, x)
    base = d._predict(slots, refs, anchors, anchor_mask)
    initial = d._predict(slots, refs, anchors, anchor_mask, tracker_summary=summary)
    for a, b in zip(base, initial):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        d.tracker_momentum_head.output.bias.fill_(1.0)
        d.pid_head.weight.zero_()
        d.pid_head.bias.zero_()
        d.pid_head.bias[2] = 10  # neutral hadron
    neutral = d._predict(slots, refs, anchors, anchor_mask, tracker_summary=summary)
    neutral_base = d._predict(slots, refs, anchors, anchor_mask)
    torch.testing.assert_close(neutral[2], neutral_base[2], rtol=0, atol=0)
    with torch.no_grad():
        d.pid_head.bias[2] = 0
        d.pid_head.bias[1] = 10
    charged = d._predict(slots, refs, anchors, anchor_mask, tracker_summary=summary)
    assert (charged[2][0, :, 0] != neutral_base[2][0, :, 0]).any()
    torch.testing.assert_close(charged[2][..., 1:], neutral_base[2][..., 1:])
    torch.testing.assert_close(charged[2][1], neutral_base[2][1])


@pytest.mark.parametrize("sparse", [False, True])
def test_full_decoder_backward_under_bfloat16(sparse):
    m, mask, x = inputs()
    d = decoder(hit_grouping_sparse=sparse, tracker_momentum_readout=True, auxiliary_loss_weight=0.1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = d(m, mask, x)
    assert len(result.auxiliary_predictions) == 1
    loss = sum(t.float().square().mean() for t in result)
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in d.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize(
    "options",
    [
        {"tracker_momentum_readout": True},
        {"hit_grouping_sparse": True},
        {"hit_grouping": True, "query_init": "input-conditioned", "hit_grouping_sparse": True, "hit_grouping_topk": 257},
        {"hit_grouping": True, "query_init": "input-conditioned", "tracker_momentum_readout": True, "tracker_group_readout": True},
        {"hit_grouping_temperature": 0},
    ],
)
def test_invalid_configs_rejected(options):
    with pytest.raises(ValueError):
        SetDecoderConfig(**options)


def test_readout_only_preserves_baseline_outputs_at_initialization():
    m, mask, x = inputs()
    torch.manual_seed(41)
    base = decoder()
    torch.manual_seed(41)
    readout = decoder(tracker_momentum_readout=True)
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, readout.state_dict()[k], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a = base(m, mask, x)
        b = readout(m, mask, x)
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa.float(), bb.float(), rtol=0, atol=0)


def test_all_slot_pooling_matches_dense_reference_and_memory_gradients():
    m, mask, x = inputs()
    h = torch.nn.functional.normalize(torch.randn(2, 9, 5), dim=-1).requires_grad_()
    q = torch.nn.functional.normalize(torch.randn(2, 4, 5), dim=-1).requires_grad_()
    bg = torch.randn(2, 9, 1, requires_grad=True)
    actual, _ = pool_hit_associations(m, mask, x, (h, q, bg), 4, 3, 4, 1.0, True)
    probs = torch.cat((10 * h @ q.transpose(1, 2), bg), -1).softmax(-1)[..., :4] * mask[..., None]
    counts = []
    means = []
    for detector in [1, 2]:
        w = probs * (x[..., 0] == detector)[..., None]
        count = w.sum(1)
        counts.append(count)
        means.append(torch.einsum("bhs,bhw->bsw", w, m) / count.clamp_min(1)[..., None])
    energy = torch.einsum("bhs,bh->bs", probs * (x[..., 0] == 2)[..., None], x[..., 5].clamp_min(0))
    expected = torch.cat((*means, torch.stack(counts, -1).log1p(), energy.log1p()[..., None]), -1)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.square().sum(), (m, h, q, bg), retain_graph=True)
    ref = torch.autograd.grad(expected.square().sum(), (m, h, q, bg))
    for a, b in zip(grad, ref):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
