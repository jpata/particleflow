import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for fused HEPTv2")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_bucket_attention_outputs_and_hash_weight_gradients(dtype):
    # Both outputs enter HEPTv2's loss: LSE controls mixing across hashes.
    torch.manual_seed(42)
    tensors = [torch.randn(2, 2, 3, 128, 16, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    q, k, v = tensors
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(16)
    expected_lse = scores.logsumexp(-1, keepdim=True)
    expected_out = scores.softmax(-1) @ v.float()
    lse, out = qkv_res(q, k, v)
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.float(), expected_out, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(lse, expected_lse, atol=tolerance, rtol=tolerance)
    actual_loss = (out.float() * lse.softmax(0)).square().sum()
    expected_loss = (expected_out * expected_lse.softmax(0)).square().sum()
    actual_grads = torch.autograd.grad(actual_loss, tensors)
    expected_grads = torch.autograd.grad(expected_loss, tensors)
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for fused HEPTv2")
def test_gpu_never_falls_back_to_dense_attention(monkeypatch):
    import mlpf.model.heptv2 as heptv2

    tensors = [torch.randn(1, 2, 2, 8, 8, device="cuda") for _ in range(3)]
    monkeypatch.setattr(heptv2, "flex_attention", None)
    with pytest.raises(RuntimeError, match="dense fallback is disabled"):
        qkv_res(*tensors)

    def failing_kernel(*args, **kwargs):
        raise ValueError("kernel failure")

    monkeypatch.setattr(heptv2, "flex_attention", failing_kernel)
    with pytest.raises(RuntimeError, match="dense GPU fallback is disabled") as error:
        qkv_res(*tensors)
    assert isinstance(error.value.__cause__, ValueError)
