"""
Spec: Verifies 'SimpleMultiheadAttention' respects 'key_padding_mask'. Assertion: Changes the values of masked (padded) elements in 'K' and 'V' and verifies that the output for unmasked queries remains bit-identical. This ensures that padding noise does not leak into the attention results.
"""

import torch
import pytest
from torch.nn.attention import SDPBackend

import mlpf.model.mlpf as mlpf_module
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


def test_flash_attn_varlen_uses_packed_values_and_offsets(monkeypatch):
    calls = []

    def fake_flash_attn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, causal):
        calls.append((q.shape, k.shape, v.shape, cu_q.clone(), cu_k.clone(), max_q, max_k, dropout_p, causal))
        outputs = []
        for start, stop in zip(cu_q[:-1].tolist(), cu_q[1:].tolist()):
            q_event = q[start:stop].transpose(0, 1).unsqueeze(0)
            k_event = k[start:stop].transpose(0, 1).unsqueeze(0)
            v_event = v[start:stop].transpose(0, 1).unsqueeze(0)
            output = torch.nn.functional.scaled_dot_product_attention(q_event, k_event, v_event)
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(mlpf_module, "_flash_attn_varlen_func", fake_flash_attn_varlen)
    module = SimpleMultiheadAttention(32, 4, attention_type="flash", use_flash_attn_varlen=True).eval()
    reference = SimpleMultiheadAttention(32, 4, attention_type="math").eval()
    reference.load_state_dict(module.state_dict())

    batch = torch.randn(2, 7, 32)
    valid_mask = torch.tensor(
        [[True, True, True, True, False, False, False], [True, True, True, True, True, True, True]]
    )
    packed = dense_to_jagged(batch, valid_mask)

    with torch.no_grad():
        actual, _ = module(packed, packed, packed)
        expected = torch.cat(
            [reference(event, event, event)[0].squeeze(0) for event in (batch[:1, :4], batch[1:, :7])]
        )

    assert actual.values.shape == (11, 32)
    torch.testing.assert_close(actual.values, expected)
    assert len(calls) == 1
    q_shape, k_shape, v_shape, cu_q, cu_k, max_q, max_k, dropout_p, causal = calls[0]
    assert q_shape == k_shape == v_shape == (11, 4, 8)
    assert cu_q.dtype == cu_k.dtype == torch.int32
    torch.testing.assert_close(cu_q, torch.tensor([0, 4, 11], dtype=torch.int32))
    torch.testing.assert_close(cu_k, cu_q)
    assert max_q == max_k == 7
    assert dropout_p == 0.0
    assert not causal


if __name__ == "__main__":
    d = get_device()
    test_simple_mha_masking(d)
    test_simple_mha_onnx_fused_masking(d)
