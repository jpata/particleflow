import torch
import pytest
import time
from mlpf.model.mlpf import LinearAttention, SimpleMultiheadAttention
from mlpf.model.gnn_lsh import CombinedGraphLayer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def device():
    return get_device()


def benchmark_module(module, inputs, num_iters=10, autocast=False, device_type="cpu"):
    # Warmup
    for _ in range(3):
        if autocast and device_type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _ = module(*inputs)
        else:
            _ = module(*inputs)

    if device_type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    for _ in range(num_iters):
        if autocast and device_type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _ = module(*inputs)
        else:
            _ = module(*inputs)

    if device_type == "cuda":
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / num_iters  # in ms
    else:
        elapsed_time = (time.time() - start_time) * 1000.0 / num_iters  # in ms

    return elapsed_time


@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_linear_attention_stability(device, seq_len):
    embed_dim = 256
    num_heads = 16
    batch_size = 1

    module = LinearAttention(embed_dim, num_heads).to(device)
    module.eval()

    # Large inputs to test stability
    q = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0
    k = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0
    v = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0

    # Test FP32
    with torch.no_grad():
        out_32, _ = module(q, k, v)
        assert torch.isfinite(out_32).all(), "LinearAttention FP32 output is not finite"

        # Test FP16 Autocast
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out_16, _ = module(q, k, v)
            assert torch.isfinite(out_16).all(), "LinearAttention FP16 autocast output is not finite"

            # Check proximity (LinearAttention is now stabilized)
            mae = (out_32 - out_16.float()).abs().mean().item()
            assert mae < 1e-1, f"LinearAttention FP16 discrepancy too large: {mae}"

    t_ms = benchmark_module(module, (q, k, v), device_type=device.type)
    print(f"\nLinearAttention (seq_len={seq_len}) runtime: {t_ms:.2f} ms")


@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_simple_mha_stability(device, seq_len):
    embed_dim = 256
    num_heads = 16
    batch_size = 1

    module = SimpleMultiheadAttention(embed_dim, num_heads).to(device)
    module.eval()

    q = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0
    k = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0
    v = torch.randn(batch_size, seq_len, embed_dim, device=device) * 10.0

    with torch.no_grad():
        out_32, _ = module(q, k, v)
        assert torch.isfinite(out_32).all(), "SimpleMultiheadAttention FP32 output is not finite"

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out_16, _ = module(q, k, v)
            assert torch.isfinite(out_16).all(), "SimpleMultiheadAttention FP16 autocast output is not finite"

    t_ms = benchmark_module(module, (q, k, v), device_type=device.type)
    print(f"\nSimpleMultiheadAttention (seq_len={seq_len}) runtime: {t_ms:.2f} ms")


@pytest.mark.parametrize("seq_len", [1024, 4096])
def test_gnn_lsh_stability(device, seq_len):
    bin_size = 128
    # Ensure seq_len is divisible by bin_size
    if seq_len % bin_size != 0:
        seq_len = (seq_len // bin_size) * bin_size

    inout_dim = 256
    module = CombinedGraphLayer(
        inout_dim=inout_dim,
        max_num_bins=seq_len // bin_size + 2,
        bin_size=bin_size,
        distance_dim=64,
        layernorm=True,
        num_node_messages=2,
        dropout=0.0,
        ffn_dist_hidden_dim=128,
    ).to(device)
    module.eval()

    # GNN-LSH can be sensitive to large values in distance computation
    x = torch.randn(1, seq_len, inout_dim, device=device) * 10.0
    mask = torch.ones(1, seq_len, device=device)
    initial_embedding = x.clone()

    with torch.no_grad():
        out_32 = module(x, mask, initial_embedding)
        assert torch.isfinite(out_32).all(), "CombinedGraphLayer FP32 output is not finite"

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out_16 = module(x, mask, initial_embedding)
            assert torch.isfinite(out_16).all(), "CombinedGraphLayer FP16 autocast output is not finite"

    t_ms = benchmark_module(module, (x, mask, initial_embedding), device_type=device.type)
    print(f"\nCombinedGraphLayer (seq_len={seq_len}) runtime: {t_ms:.2f} ms")


if __name__ == "__main__":
    # This allows running the file directly with python and seeing prints
    d = get_device()
    for sl in [1024, 4096]:
        test_linear_attention_stability(d, sl)
        test_simple_mha_stability(d, sl)
        test_gnn_lsh_stability(d, sl)
