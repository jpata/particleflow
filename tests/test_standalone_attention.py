import torch
import pytest
from mlpf.standalone.train import (
    HEPTAttentionLayer,
    GlobalAttentionLayer,
    StandardAttentionLayer,
    FastformerAttentionLayer,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def device():
    return get_device()


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_layer_shapes(device, layer_class):
    embedding_dim = 64
    num_heads = 4
    width = 128
    batch_size = 2
    seq_len = 100

    # HEPT requires block_size to be a multiple of seq_len or pads it.
    # Standard HEPTAttentionLayer default block_size is 100.

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 50

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0  # dummy type

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = layer(x, mask, X)

    assert out.shape == (batch_size, seq_len, embedding_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_layer_gradients(device, layer_class):
    # StandardAttentionLayer with FLASH_ATTENTION might fail on CPU.
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 1
    seq_len = 64

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 32

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = layer(x, mask, X)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_layer_masking(device, layer_class):
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 1
    seq_len = 16

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 8

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)
    layer.eval()  # Disable dropout for deterministic masking test

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    # Mask out the second half of the sequence
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, seq_len // 2 :] = 0.0

    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    # Input should be masked if we want to ensure zeroed elements don't affect anything,
    # but the layer itself is supposed to handle the mask.
    # In train.py, the forward pass does:
    # if mask is not None: x = x * mask_

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = layer(x, mask, X)

    # Check that masked outputs are zero
    assert torch.allclose(out[:, seq_len // 2 :], torch.zeros_like(out[:, seq_len // 2 :]))

    # Check that changing masked inputs doesn't change unmasked outputs
    x2 = x.clone()
    x2[:, seq_len // 2 :] += 10.0
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out2 = layer(x2, mask, X)

    assert torch.allclose(out[:, : seq_len // 2], out2[:, : seq_len // 2], atol=1e-5)


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
@pytest.mark.parametrize("seq_len", [13, 128, 257])
def test_varying_sequence_lengths(device, layer_class, seq_len):
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 2

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 10

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = layer(x, mask, X)

    assert out.shape == (batch_size, seq_len, embedding_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_permutation_equivariance(device, layer_class):
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 1
    seq_len = 32

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 8

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)
    layer.eval()

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    # Original output
    with torch.no_grad():
        out_orig = layer(x, mask, X)

    # Permute
    perm = torch.randperm(seq_len)
    x_perm = x[:, perm]
    mask_perm = mask[:, perm]
    X_perm = X[:, perm]

    with torch.no_grad():
        out_perm = layer(x_perm, mask_perm, X_perm)

    # Un-permute output
    inv_perm = torch.argsort(perm)
    out_unperm = out_perm[:, inv_perm]

    # They should be equivalent
    # Using a slightly higher tolerance for HEPT due to LSH and sorting ties
    atol = 1e-4 if layer_class == HEPTAttentionLayer else 1e-5
    assert torch.allclose(out_orig, out_unperm, atol=atol)


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_attention_spike_sensitivity(device, layer_class):
    """
    Check if changing one element affects the output of another element.
    This confirms the attention mechanism is actually 'attending' across the sequence.
    """
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 1  # Use 1 head for clearer sensitivity
    width = 64
    batch_size = 1
    seq_len = 16

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 16

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)
    layer.eval()

    # Use random noise as baseline to ensure softmax doesn't collapse to a constant
    # We also vary the baseline query slightly to ensure non-uniformity
    x = torch.randn(batch_size, seq_len, embedding_dim, device=device) * 0.1
    # Add a unique 'address' to each element to help attention distinguish them
    x += torch.linspace(0, 1, seq_len, device=device).view(1, seq_len, 1)

    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    # Add a large spike at position 0
    x_spike = x.clone()
    x_spike[:, 0, :] += 100.0

    # Disable autocast for sensitivity test to avoid precision issues with bfloat16
    with torch.no_grad():
        out1 = layer(x, mask, X)
        out2 = layer(x_spike, mask, X)

    # Check if output at position 1 changed
    diff = (out1[:, 1:] - out2[:, 1:]).abs().max().item()
    print(f"\n{layer_class.__name__} diff: {diff}")

    if layer_class == HEPTAttentionLayer:
        # HEPT is extremely sparse (RBF kernel + buckets).
        # With default coords, elements might be in different buckets or far apart.
        # We'll just check if there is ANY change in ANY other position,
        # or if we need to make them spatially close.
        # Let's try making them spatially identical for the test.
        X_close = torch.zeros_like(X)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out1_close = layer(x, mask, X_close)
            out2_close = layer(x_spike, mask, X_close)
        diff = (out1_close[:, 1:] - out2_close[:, 1:]).abs().max().item()
        assert diff > 1e-6, f"HEPTAttentionLayer shows no sensitivity even when coordinates are identical (diff={diff})"
    else:
        # For Global/Fastformer, sensitivity is diluted by 1/N. 1e-9 should be safe.
        # Standard should definitely show it too.
        assert diff > 1e-9, f"Layer {layer_class.__name__} shows no sensitivity to input spikes at other positions (diff={diff})"


def test_standard_attention_mask_exclusion(device):
    """
    Test if StandardAttentionLayer correctly ignores masked elements in its softmax.
    If we change the VALUES of masked elements, the UNMASKED outputs should not change.
    (This is already covered by test_layer_masking, but let's be more explicit about 'attending').
    """
    if device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 1
    width = 64
    batch_size = 1
    seq_len = 4

    layer = StandardAttentionLayer(embedding_dim, num_heads, width).to(device)
    layer.eval()

    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    # To TRULY test if it attends to padding, we need to compare:
    # 1. Output with mask [1, 1, 0, 0]
    # 2. Output with mask [1, 1, 1, 1] but inputs at 2,3 are 0.
    # If they are different, it means the padding elements (even if 0) are part of the softmax.

    x_base = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    x_base[:, 2:] = 0.0  # elements 2 and 3 are zero

    mask_partial = torch.zeros(batch_size, seq_len, device=device)
    mask_partial[:, :2] = 1.0  # only elements 0 and 1 are 'real'

    mask_full = torch.ones(batch_size, seq_len, device=device)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out_partial = layer(x_base, mask_partial, X)
        out_full = layer(x_base, mask_full, X)

    # If they are different at pos 0 or 1, it means pos 0/1 are attending to the padding (2,3)
    diff = (out_partial[:, :2] - out_full[:, :2]).abs().max().item()
    # After the fix, this diff should be 0.0 (or very close)
    assert diff < 1e-6, f"StandardAttentionLayer still attends to masked elements (diff={diff:.6f})"


@pytest.mark.parametrize(
    "layer_class",
    [
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_attention_weights_normalization(device, layer_class):
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 2
    seq_len = 16

    layer = layer_class(embedding_dim, num_heads, width).to(device)
    layer.eval()

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    # Mask out the last 4 elements
    mask[:, -4:] = 0.0

    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out, attn = layer(x, mask, X, return_attn=True)

    if layer_class == StandardAttentionLayer:
        # attn is [B, num_heads, seq_len, seq_len]
        # Sum over the last dimension (keys) should be 1.0 for valid queries
        attn_sum = attn.sum(dim=-1)
        # Check unmasked queries
        assert torch.allclose(attn_sum[:, :, :-4], torch.ones_like(attn_sum[:, :, :-4]), atol=1e-5)

        # Masked queries (rows) might sum to 0 or 1 depending on implementation.
        # But for unmasked queries, the attention to masked keys should be exactly 0
        assert torch.allclose(attn[:, :, :-4, -4:], torch.zeros_like(attn[:, :, :-4, -4:]), atol=1e-5)

    elif layer_class == GlobalAttentionLayer:
        # attn is (alpha, beta), shapes: [B, seq_len, num_heads]
        alpha, beta = attn
        # alpha and beta are probability distributions over the sequence dimension (dim=1)
        alpha_sum = alpha.sum(dim=1)
        beta_sum = beta.sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
        assert torch.allclose(beta_sum, torch.ones_like(beta_sum), atol=1e-5)

        # Masked elements should have exactly 0 probability
        assert torch.allclose(alpha[:, -4:, :], torch.zeros_like(alpha[:, -4:, :]), atol=1e-5)
        assert torch.allclose(beta[:, -4:, :], torch.zeros_like(beta[:, -4:, :]), atol=1e-5)

    elif layer_class == FastformerAttentionLayer:
        # attn is (alpha, beta), shapes: [B, seq_len, decode_dim]
        alpha, beta = attn
        alpha_sum = alpha.sum(dim=1)
        beta_sum = beta.sum(dim=1)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5)
        assert torch.allclose(beta_sum, torch.ones_like(beta_sum), atol=1e-5)

        # Masked elements should have exactly 0 probability
        assert torch.allclose(alpha[:, -4:, :], torch.zeros_like(alpha[:, -4:, :]), atol=1e-5)
        assert torch.allclose(beta[:, -4:, :], torch.zeros_like(beta[:, -4:, :]), atol=1e-5)


@pytest.mark.parametrize(
    "layer_class",
    [
        HEPTAttentionLayer,
        GlobalAttentionLayer,
        StandardAttentionLayer,
        FastformerAttentionLayer,
    ],
)
def test_qkv_gradient_flow(device, layer_class):
    if layer_class == StandardAttentionLayer and device.type == "cpu":
        pytest.skip("StandardAttentionLayer with FLASH_ATTENTION requires CUDA")

    embedding_dim = 32
    num_heads = 2
    width = 64
    batch_size = 1
    seq_len = 16

    kwargs = {}
    if layer_class == HEPTAttentionLayer:
        kwargs["block_size"] = 8

    layer = layer_class(embedding_dim, num_heads, width, **kwargs).to(device)

    x = torch.randn(batch_size, seq_len, embedding_dim, device=device, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, device=device)
    X = torch.randn(batch_size, seq_len, 25, device=device)
    X[..., 0] = 1.0

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out = layer(x, mask, X)

    loss = out.sum()
    loss.backward()

    # Check gradients on projection layers
    if layer_class == HEPTAttentionLayer:
        assert layer.w_q.weight.grad is not None and (layer.w_q.weight.grad.abs() > 0).any()
        assert layer.w_k.weight.grad is not None and (layer.w_k.weight.grad.abs() > 0).any()
        assert layer.w_v.weight.grad is not None and (layer.w_v.weight.grad.abs() > 0).any()
    elif layer_class == GlobalAttentionLayer:
        assert layer.mha.q_proj.weight.grad is not None and (layer.mha.q_proj.weight.grad.abs() > 0).any()
        assert layer.mha.k_proj.weight.grad is not None and (layer.mha.k_proj.weight.grad.abs() > 0).any()
        assert layer.mha.v_proj.weight.grad is not None and (layer.mha.v_proj.weight.grad.abs() > 0).any()
    elif layer_class == StandardAttentionLayer:
        assert layer.q_proj.weight.grad is not None and (layer.q_proj.weight.grad.abs() > 0).any()
        assert layer.k_proj.weight.grad is not None and (layer.k_proj.weight.grad.abs() > 0).any()
        assert layer.v_proj.weight.grad is not None and (layer.v_proj.weight.grad.abs() > 0).any()
    elif layer_class == FastformerAttentionLayer:
        assert layer.attn.weight_q.weight.grad is not None and (layer.attn.weight_q.weight.grad.abs() > 0).any()
        assert layer.attn.weight_k.weight.grad is not None and (layer.attn.weight_k.weight.grad.abs() > 0).any()
        assert layer.attn.weight_v.weight.grad is not None and (layer.attn.weight_v.weight.grad.abs() > 0).any()


if __name__ == "__main__":
    pytest.main([__file__])
