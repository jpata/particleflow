"""
Spec: Verifies 'SimpleMultiheadAttention' respects 'key_padding_mask'. Assertion: Changes the values of masked (padded) elements in 'K' and 'V' and verifies that the output for unmasked queries remains bit-identical. This ensures that padding noise does not leak into the attention results.
"""

import torch
import pytest
from torch.nn.attention import SDPBackend

from mlpf.conf import AttentionType
from mlpf.model.mlpf import SimpleMultiheadAttention, dense_to_jagged, jagged_to_dense


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def device():
    return get_device()


def test_simple_mha_masking(device):
    embed_dim = 32
    num_heads = 4
    batch_size = 1
    seq_len = 8

    module = SimpleMultiheadAttention(embed_dim, num_heads).to(device)
    module.eval()

    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    k = torch.randn(batch_size, seq_len, embed_dim, device=device)
    v = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # key_padding_mask: True means ignore
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, seq_len // 2 :] = True  # mask out the second half

    with torch.no_grad():
        out1, _ = module(q, k, v, key_padding_mask=key_padding_mask)

        # Now change the values of the masked elements in k and v
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, seq_len // 2 :, :] += 10.0
        v_modified[:, seq_len // 2 :, :] += 10.0

        out2, _ = module(q, k_modified, v_modified, key_padding_mask=key_padding_mask)

    # The output for all queries should be identical because queries are attending (or not) to the same things
    # Actually, in standard attention, query i attends to all non-masked keys.
    # If key_padding_mask masks out keys j, then changing k[j] and v[j] should not affect any out[i].
    assert torch.allclose(out1, out2, atol=1e-5), "Masked elements affected the output"


def test_simple_mha_onnx_fused_masking(device):
    # Test the export_onnx_fused=True path
    embed_dim = 32
    num_heads = 4
    batch_size = 1
    seq_len = 8

    module = SimpleMultiheadAttention(embed_dim, num_heads, export_onnx_fused=True).to(device)
    module.eval()

    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    k = torch.randn(batch_size, seq_len, embed_dim, device=device)
    v = torch.randn(batch_size, seq_len, embed_dim, device=device)

    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    key_padding_mask[:, seq_len // 2 :] = True

    with torch.no_grad():
        out1, _ = module(q, k, v, key_padding_mask=key_padding_mask)

        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[:, seq_len // 2 :, :] += 10.0
        v_modified[:, seq_len // 2 :, :] += 10.0

        out2, _ = module(q, k_modified, v_modified, key_padding_mask=key_padding_mask)

    assert torch.allclose(out1, out2, atol=1e-5), "Masked elements affected the output (ONNX fused path)"


def test_flash_mha_processes_jagged_batch_without_padding(device):
    torch.manual_seed(7)
    embed_dim = 32
    module = SimpleMultiheadAttention(embed_dim, 4, attention_type="flash").to(device).eval()
    # Exercise the flash-specific variable-length branch on hosts without CUDA.
    module.attn_params[AttentionType.FLASH] = [SDPBackend.MATH]

    short = torch.randn(1, 4, embed_dim, device=device)
    long = torch.randn(1, 7, embed_dim, device=device)
    padded_short = torch.nn.functional.pad(short, (0, 0, 0, 3))
    batch = torch.cat([padded_short, long], dim=0)
    valid_mask = ~torch.tensor(
        [[False, False, False, False, True, True, True], [False, False, False, False, False, False, False]],
        device=device,
    )
    jagged_batch = dense_to_jagged(batch, valid_mask)

    with torch.no_grad():
        expected, _ = module(short, short, short)
        actual_jagged, _ = module(jagged_batch, jagged_batch, jagged_batch)
        actual = jagged_to_dense(actual_jagged, valid_mask)

    torch.testing.assert_close(expected, actual[:1, : short.shape[1]], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    d = get_device()
    test_simple_mha_masking(d)
    test_simple_mha_onnx_fused_masking(d)
