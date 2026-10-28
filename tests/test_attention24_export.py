"""Padding and batch invariance through the complete model and real ONNX export."""

import json
import os

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.conf import MLPFConfig, AttentionType  # noqa: E402
from mlpf.model.mlpf import MLPF, SimpleMultiheadAttention  # noqa: E402
from mlpf.model.onnx_attention import export_mlpf, validate_padding_mask  # noqa: E402


def make_model(fused=True):
    torch.manual_seed(19)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {"type": "attention", "attention": {"num_heads": 2, "head_dim": 8, "attention_type": "math", "export_onnx_fused": fused}},
        }
    )
    return MLPF(config).eval(), config


def make_batch(config, lengths, padded_length):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(len(lengths), padded_length, config.input_dim, generator=generator)
    x[..., 0] = config.elemtypes_nonzero[0]
    x[..., 1] = x[..., 1].exp()
    x[..., 5] = x[..., 5].exp()
    mask = torch.arange(padded_length)[None, :] < torch.tensor(lengths)[:, None]
    x[~mask] = 0
    return x, mask.float()


def create_session(path, provider="CPUExecutionProvider", options=None):
    options = options or ort.SessionOptions()
    options.intra_op_num_threads = 1
    providers = [(provider, {"sdpa_kernel": "1", "use_tf32": "0"})] if provider == "CUDAExecutionProvider" else [provider]
    session = ort.InferenceSession(str(path), options, providers=providers)
    assert provider in session.get_providers()
    return session


@pytest.mark.parametrize("dtype,attention_dtype", [(torch.float32, None), (torch.float32, torch.float16), (torch.float16, None)])
def test_full_model_attention24_dynamic_padding(tmp_path, dtype, attention_dtype):
    torch.set_num_threads(2)
    model, config = make_model()
    model = model.to(dtype)
    x, mask = make_batch(config, [7, 3], 9)
    path = tmp_path / "model.onnx"
    export_mlpf(model, x.to(dtype), mask.to(dtype), path, attention_dtype=attention_dtype)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    attention = [n for n in graph.graph.node if n.op_type == "Attention"]
    assert attention
    assert all(n.domain == "" and len(n.input) == 7 and n.input[6] and not any(n.input[3:6]) for n in attention)
    assert not any(n.op_type in ("ScatterND", "MultiHeadAttention", "SDPA", "Elu") for n in graph.graph.node)
    assert all(m.onnx_attention_dtype is None for m in model.modules() if isinstance(m, SimpleMultiheadAttention))
    runtime = create_session(path)
    # Deliberately differ from the export's batch size AND sequence length.
    for lengths, size in [([5], 5), ([11, 6, 1], 15)]:
        x, mask = make_batch(config, lengths, size)
        x, mask = x.to(dtype), mask.to(dtype)
        feeds = {"Xfeat_normed": x.numpy(), "mask": mask.numpy()}
        actual = runtime.run(None, feeds)
        with torch.no_grad():
            expected = model(x, mask)
        valid = mask.bool().numpy()
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a[valid], b.float().numpy()[valid], atol=0.01 if dtype == torch.float16 else 0.003, rtol=0.02)
            assert np.isfinite(a).all()
        for i, n in enumerate(lengths):
            single = runtime.run(None, {"Xfeat_normed": x[i : i + 1, :n].numpy(), "mask": mask[i : i + 1, :n].numpy()})
            for a, b in zip(actual, single):
                np.testing.assert_allclose(a[i, :n], b[0], atol=0.002, rtol=0.003)
        # Finite junk in padding must not leak into valid tokens.
        poisoned = x.clone()
        poisoned[~mask.bool()] = 10
        poisoned[..., 0][~mask.bool()] = 0
        second = runtime.run(None, {"Xfeat_normed": poisoned.numpy(), "mask": mask.numpy()})
        for a, b in zip(actual, second):
            np.testing.assert_allclose(a[valid], b[valid], atol=1e-6, rtol=1e-6)


def test_dense_pytorch_padding_and_gradients():
    model, config = make_model(fused=False)
    x, mask = make_batch(config, [7, 3], 11)
    x.requires_grad_(True)
    actual = model(x, mask)
    for i, n in enumerate([7, 3]):
        expected = model(x[i : i + 1, :n], mask[i : i + 1, :n])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a[i, :n], b[0], atol=1e-5, rtol=1e-5)
    # Classifier loss isolates attention from deliberately singular padded energy features.
    actual[1][mask.bool()].square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[~mask.bool()]) == 0


@pytest.mark.parametrize("mask", [torch.tensor([[1, 0, 1]]), torch.tensor([[0, 0, 0]]), torch.tensor([[1.0, 0.5, 0.0]])])
def test_deployment_rejects_invalid_masks(mask):
    with pytest.raises(ValueError):
        validate_padding_mask(mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to verify FlashAttention dispatch")
def test_cuda_flash_full_model_export(tmp_path, capfd):
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("Use envs/ort-gpu for CUDA ONNX Runtime")
    pytest.importorskip("flash_attn")
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    ort.preload_dlls()
    model, config = make_model()
    model = model.half()
    x, mask = make_batch(config, [13, 7, 1], 17)
    x, mask = x.half(), mask.half()
    path = tmp_path / "flash.onnx"
    export_mlpf(model, x, mask, path)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "profile")
    session = create_session(path, "CUDAExecutionProvider", options)
    actual = session.run(None, {"Xfeat_normed": x.numpy(), "mask": mask.numpy()})
    output = capfd.readouterr()
    assert "SdpaKernel=FLASH_ATTENTION" in output.out + output.err
    profile = json.loads(open(session.end_profiling()).read())
    assert any(e.get("args", {}).get("op_name") == "Attention" and e["args"].get("provider") == "CUDAExecutionProvider" for e in profile)
    assert not any(e.get("args", {}).get("op_name") == "Elu" and e["args"].get("provider") == "CPUExecutionProvider" for e in profile)
    config.model.attention.export_onnx_fused = False
    config.model.attention.attention_type = AttentionType.FLASH
    config.model.attention.use_jagged_attention = True
    config.model.attention.use_flash_attn_varlen = True
    flash_model = MLPF(config).eval().half().cuda()
    flash_model.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        expected = flash_model(x.cuda(), mask.cuda())
    assert any("flash_attn_varlen" in event.key for event in profiler.key_averages())
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a[mask.bool()], b.float().cpu().numpy()[mask.bool()], atol=0.005, rtol=0.02)
