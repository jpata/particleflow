import torch
import math
from mlpf.model.heptv2 import (
    HEPTv2Layer,
    Qwen3RMSNorm,
    Qwen3MLP,
    qkv_res,
)


def test_qwen3_rms_norm():
    norm = Qwen3RMSNorm(16)
    x = torch.randn(2, 4, 16)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance is indeed normalized
    var = out.pow(2).mean(-1)
    # Since weight is 1.0, mean variance should be close to 1.0
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_qwen3_mlp():
    mlp = Qwen3MLP(hidden_size=16, intermediate_size=32)
    x = torch.randn(4, 16)
    out = mlp(x)
    assert out.shape == (4, 16)


def test_qkv_res_fallback_vs_compiled():
    # Test that standard scaled dot product attention fallback is correct
    c, h, nbuckets, bucketsz, d = 2, 2, 4, 8, 16
    s_query = torch.randn(c, h, nbuckets, bucketsz, d)
    s_key = torch.randn(c, h, nbuckets, bucketsz, d)
    s_value = torch.randn(c, h, nbuckets, bucketsz, d)

    # Compute using the fallback calculation inside qkv_res manually/directly:
    scores = torch.matmul(s_query, s_key.transpose(-1, -2)) / math.sqrt(d)
    lse_expected = torch.logsumexp(scores, dim=-1, keepdim=True)
    probs = torch.softmax(scores, dim=-1)
    out_expected = torch.matmul(probs, s_value)

    # Run qkv_res with flex_attention set to None to force fallback
    import mlpf.model.heptv2 as heptv2

    original_flex = heptv2.flex_attention
    try:
        heptv2.flex_attention = None
        lse_actual, out_actual = qkv_res(s_query, s_key, s_value)

        assert torch.allclose(lse_actual, lse_expected, atol=1e-5)
        assert torch.allclose(out_actual, out_expected, atol=1e-5)
    finally:
        heptv2.flex_attention = original_flex


def test_heptv2_layer_forward():
    layer = HEPTv2Layer(
        embedding_dim=32, num_heads=4, width=64, dropout=0.0, block_size=8, n_hashes=2, num_regions=10, num_w_per_dist=4, pe_type="learned"
    )

    B, S, D = 2, 16, 32
    x = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)
    # Mask out the last elements in the second event
    mask[1, 12:] = False

    # X_features needs to contain: elem_type, pt, eta, sin_phi, cos_phi, energy, ...
    # eta is at index 2, sin_phi at 3, cos_phi at 4.
    X_features = torch.randn(B, S, 6)
    # Set coordinates
    X_features[..., 2] = torch.linspace(-3.0, 3.0, S).unsqueeze(0).repeat(B, 1)  # eta
    phi = torch.linspace(-math.pi, math.pi, S).unsqueeze(0).repeat(B, 1)
    X_features[..., 3] = torch.sin(phi)
    X_features[..., 4] = torch.cos(phi)

    out = layer(x, mask, X_features)

    assert out.shape == (B, S, D)
    # Check that masked out elements remain zero
    assert torch.allclose(out[1, 12:], torch.zeros_like(out[1, 12:]))
