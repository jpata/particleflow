import math

import pytest
import torch

from mlpf.model.hept import (
    HEPTLayer,
    qkv_res as hept_qkv_res,
    sort_to_buckets as hept_sort_to_buckets,
    unsort_from_buckets as hept_unsort_from_buckets,
    invert_permutation as hept_invert_permutation,
)
from mlpf.model.heptv2 import HEPTv2Layer, qkv_res as heptv2_qkv_res


def _make_x_features(batch_size, seq_len, feature_dim=6, *, scale=1.0):
    features = torch.zeros(batch_size, seq_len, feature_dim)
    features[..., 0] = 1.0
    features[..., 1] = 1.0
    features[..., 2] = torch.linspace(-3.0 * scale, 3.0 * scale, seq_len).unsqueeze(0)
    phi = torch.linspace(-math.pi, math.pi, seq_len).unsqueeze(0)
    features[..., 3] = torch.sin(phi)
    features[..., 4] = torch.cos(phi)
    features[..., 5] = 1.0
    return features


def _make_layer(layer_cls):
    kwargs = dict(
        embedding_dim=32,
        num_heads=4,
        width=64,
        dropout=0.0,
        block_size=8,
        n_hashes=2,
        num_regions=10,
        num_w_per_dist=4,
    )
    if layer_cls is HEPTv2Layer:
        kwargs["pe_type"] = "learned"
    else:
        kwargs["pos"] = False
    return layer_cls(**kwargs)


@pytest.mark.parametrize("layer_cls", [HEPTLayer, HEPTv2Layer])
def test_hept_layer_forward_basic_properties(layer_cls):
    torch.manual_seed(1)
    layer = _make_layer(layer_cls)
    layer.eval()

    batch_size, seq_len, embedding_dim = 2, 16, 32
    x = torch.randn(batch_size, seq_len, embedding_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[1, 12:] = False
    features = _make_x_features(batch_size, seq_len)

    out = layer(x, mask, features)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]), atol=1e-7)


@pytest.mark.parametrize("layer_cls", [HEPTLayer, HEPTv2Layer])
def test_hept_layer_backward_has_finite_gradients(layer_cls):
    torch.manual_seed(2)
    layer = _make_layer(layer_cls)
    layer.train()

    batch_size, seq_len, embedding_dim = 2, 16, 32
    x = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 10:] = False
    mask[1, 14:] = False
    features = _make_x_features(batch_size, seq_len)

    out = layer(x, mask, features)
    loss = out[mask].pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.allclose(x.grad[~mask], torch.zeros_like(x.grad[~mask]), atol=1e-7)

    grads = [param.grad for param in layer.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("layer_cls", [HEPTLayer, HEPTv2Layer])
def test_hept_layer_mask_edge_cases_keep_at_least_one_valid_token(layer_cls):
    torch.manual_seed(3)
    layer = _make_layer(layer_cls)
    layer.eval()

    batch_size, seq_len, embedding_dim = 3, 16, 32
    x = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, :] = True
    mask[1, :5] = True
    mask[2, 7] = True
    features = _make_x_features(batch_size, seq_len)

    out = layer(x, mask, features)
    loss = out[mask].sum()
    loss.backward()

    assert torch.isfinite(out).all()
    assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]), atol=1e-7)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.allclose(x.grad[~mask], torch.zeros_like(x.grad[~mask]), atol=1e-7)


@pytest.mark.parametrize("layer_cls", [HEPTLayer, HEPTv2Layer])
def test_hept_layer_extreme_inputs_are_finite(layer_cls):
    torch.manual_seed(4)
    layer = _make_layer(layer_cls)
    layer.train()

    batch_size, seq_len, embedding_dim = 2, 16, 32
    x = (100.0 * torch.randn(batch_size, seq_len, embedding_dim)).requires_grad_(True)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[1, -2:] = False
    features = _make_x_features(batch_size, seq_len, scale=4.0)
    features[:, ::4, 2] = features[:, 0:1, 2]
    features[..., 1] = 1e3
    features[..., 5] = 1e3

    out = layer(x, mask, features)
    loss = out[mask].square().mean()
    loss.backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [param.grad for param in layer.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("layer_cls", [HEPTLayer, HEPTv2Layer])
def test_hept_layer_permutation_equivariance(layer_cls):
    torch.manual_seed(5)
    layer = _make_layer(layer_cls)
    layer.eval()

    batch_size, seq_len, embedding_dim = 2, 16, 32
    x = torch.randn(batch_size, seq_len, embedding_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -3:] = False
    features = _make_x_features(batch_size, seq_len)

    perm = torch.randperm(seq_len)
    inv_perm = torch.argsort(perm)

    with torch.no_grad():
        out = layer(x, mask, features)
        out_perm = layer(x[:, perm], mask[:, perm], features[:, perm])

    assert torch.allclose(out, out_perm[:, inv_perm], atol=1e-4, rtol=1e-4)


def test_hept_qkv_res_backward_finite_with_masks():
    torch.manual_seed(6)
    h, batch, nbuckets, bucketsz, dim = 2, 1, 3, 4, 8
    s_query = torch.randn(h, batch, nbuckets, bucketsz, dim, requires_grad=True)
    s_key = torch.randn(h, batch, nbuckets, bucketsz, dim, requires_grad=True)
    s_value = torch.randn(h, batch, nbuckets, bucketsz, dim, requires_grad=True)
    s_mask_q = torch.ones(h, batch, nbuckets, bucketsz, 1)
    s_mask_k = torch.ones(h, batch, nbuckets, bucketsz, 1)
    s_mask_q[..., -1:, :] = 0
    s_mask_k[..., -2:, :] = 0

    denom, so = hept_qkv_res(s_query, s_key, s_value, s_mask_q=s_mask_q, s_mask_k=s_mask_k)
    loss = denom.sum() + so.square().mean()
    loss.backward()

    assert torch.isfinite(denom).all()
    assert torch.isfinite(so).all()
    for tensor in (s_query, s_key, s_value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_heptv2_qkv_res_backward_finite():
    torch.manual_seed(7)
    c, h, nbuckets, bucketsz, dim = 2, 2, 3, 4, 8
    s_query = torch.randn(c, h, nbuckets, bucketsz, dim, requires_grad=True)
    s_key = torch.randn(c, h, nbuckets, bucketsz, dim, requires_grad=True)
    s_value = torch.randn(c, h, nbuckets, bucketsz, dim, requires_grad=True)

    lse, out = heptv2_qkv_res(s_query, s_key, s_value)
    loss = lse.sum() + out.square().mean()
    loss.backward()

    assert torch.isfinite(lse).all()
    assert torch.isfinite(out).all()
    for tensor in (s_query, s_key, s_value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_hept_bucket_sort_unsort_backward_finite():
    torch.manual_seed(8)
    heads, batch_size, seq_len, dim = 2, 3, 16, 4
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    perm = torch.stack([torch.randperm(seq_len) for _ in range(heads * batch_size)]).view(heads, batch_size, seq_len)

    bucketed = hept_sort_to_buckets(x, perm, bucketsz=4)
    restored = hept_unsort_from_buckets(bucketed, hept_invert_permutation(perm))
    loss = restored.square().mean()
    loss.backward()

    assert torch.allclose(restored, x.unsqueeze(0).expand(heads, -1, -1, -1), atol=1e-6)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad).item() == x.numel()
