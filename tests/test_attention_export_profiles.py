"""CPU CI coverage and opt-in tests on actual CUDA/MIGraphX devices.

MLPF_TEST_PROVIDER=cuda|migraphx selects a device test. Missing requested
providers fail; they never become CPU-only passes. Use --basetemp to retain
models, profiles and numerical reports. No flash-attn package is required.
"""

import json
import platform
import os
import subprocess
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.model.onnx_attention import export_attention, export_mlpf  # noqa: E402
from mlpf.model.onnx_graph import _lower_nonfinite, _remove_dead_nodes  # noqa: E402
from tests.test_attention24_export import make_batch, make_model  # noqa: E402


OUTPUTS = ["bid", "id", "momentum", "pu"]


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def make_device_model():
    from mlpf.conf import MLPFConfig
    from mlpf.model.mlpf import MLPF

    torch.manual_seed(19)
    config = MLPFConfig.model_validate(
        {
            "dataset": "cms",
            "data_dir": "/tmp",
            "conv_type": "attention",
            "model": {"type": "attention", "attention": {"num_heads": 2, "head_dim": 16, "num_convs": 1, "attention_type": "math"}},
        }
    )
    return MLPF(config).eval(), config


def assert_portable_graph(path, attention_format):
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    nodes = graph.graph.node
    assert not any(n.op_type in {"IsNaN", "IsInf", "ScatterND", "NonZero", "SDPA", "MultiHeadAttention"} for n in nodes)
    assert not any(n.domain == "mlpf" for n in nodes)
    assert [o.name for o in graph.graph.output] == OUTPUTS
    assert all(node.op_type == "Where" for node in nodes[-4:])
    assert [node.output[0] for node in nodes[-4:]] == OUTPUTS
    assert dict((p.key, p.value) for p in graph.metadata_props)["attention_format"] == attention_format
    if attention_format == "attention24":
        attention = [n for n in nodes if n.op_type == "Attention"]
        assert attention
        assert all(len(n.input) == 7 and n.input[6] and not any(n.input[3:6]) for n in attention)
        assert all(next(a.i for a in n.attribute if a.name == "softmax_precision") == onnx.TensorProto.FLOAT for n in attention)
    else:
        assert not any(n.op_type == "Attention" for n in nodes)
        inferred = onnx.shape_inference.infer_shapes(graph)
        types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
        assert any(n.op_type == "Softmax" for n in nodes)
        assert all(types[n.input[0]] == onnx.TensorProto.FLOAT for n in nodes if n.op_type == "Softmax")
    for value in (*graph.graph.input, *graph.graph.output):
        assert all(d.dim_param for d in value.type.tensor_type.shape.dim[:2])


def exercise_batches(session, model, config):
    report = []
    for lengths, size in [([5], 9), ([11, 6], 17), ([19, 13, 7, 1], 23)]:
        x, mask = make_batch(config, lengths, size)
        feeds = {"Xfeat_normed": x.numpy(), "mask": mask.numpy()}
        actual = session.run(OUTPUTS, feeds)
        with torch.no_grad():
            expected = model(x, mask)
        valid = mask.bool().numpy()
        differences = []
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a[valid], b.numpy()[valid], atol=0.01, rtol=0.02)
            assert np.isfinite(a).all()
            assert np.all(a[~valid] == 0)
        for i, length in enumerate(lengths):
            single = session.run(OUTPUTS, {"Xfeat_normed": x[i : i + 1, :length].numpy(), "mask": mask[i : i + 1, :length].numpy()})
            for a, b in zip(actual, single):
                differences.append(float(np.max(abs(a[i, :length] - b[0]))))
                np.testing.assert_allclose(a[i, :length], b[0], atol=0.002, rtol=0.003)
        poisoned = x.clone()
        poisoned[~mask.bool()] = 100
        second = session.run(OUTPUTS, {"Xfeat_normed": poisoned.numpy(), "mask": mask.numpy()})
        for a, b in zip(actual, second):
            np.testing.assert_array_equal(a, b)
        report.append({"lengths": lengths, "padded_length": size, "max_batch_difference": max(differences)})
    return report


@pytest.mark.parametrize("attention_format", ["attention24", "decomposed"])
@pytest.mark.parametrize("attention_dtype", [torch.float32, torch.float16])
def test_cpu_export_profiles(tmp_path, attention_format, attention_dtype):
    torch.set_num_threads(1)
    model, config = make_model(fused=False)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    x, mask = make_batch(config, [7, 3], 11)
    path = tmp_path / "model.onnx"
    export_mlpf(model, x, mask, path, attention_format=attention_format, attention_dtype=attention_dtype)
    assert_portable_graph(path, attention_format)
    exercise_batches(cpu_session(path), model, config)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert all(getattr(m, "onnx_attention_format", None) is None for m in model.modules())
    assert all(getattr(m, "onnx_attention_dtype", None) is None for m in model.modules())


@pytest.mark.parametrize("attention_format", ["attention24", "decomposed"])
def test_export_does_not_require_eager_flash_device(tmp_path, attention_format):
    from mlpf.conf import AttentionType
    from mlpf.model.mlpf import MLPF

    _, config = make_model(fused=False)
    config.model.attention.attention_type = AttentionType.FLASH
    model = MLPF(config).eval()
    assert model.use_jagged_attention
    x, mask = make_batch(config, [7, 3], 11)
    path = tmp_path / "native-flash.onnx"
    export_mlpf(model, x, mask, path, attention_format=attention_format, attention_dtype=torch.float16)
    assert model.use_jagged_attention
    assert_portable_graph(path, attention_format)
    assert all(np.isfinite(y).all() for y in cpu_session(path).run(OUTPUTS, {"Xfeat_normed": x.numpy(), "mask": mask.numpy()}))


def test_export_restores_settings_on_failure(tmp_path, monkeypatch):
    model, config = make_model(fused=False)
    x, mask = make_batch(config, [7, 3], 11)
    model.use_jagged_attention = True

    def fail(*args, **kwargs):
        assert not model.use_jagged_attention
        raise RuntimeError("deliberate export failure")

    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        export_mlpf(model, x, mask, tmp_path / "fail.onnx", attention_format="decomposed", attention_dtype=torch.float16)
    assert model.use_jagged_attention
    assert all(getattr(m, "onnx_attention_format", None) is None for m in model.modules())
    assert all(getattr(m, "onnx_attention_dtype", None) is None for m in model.modules())


@pytest.mark.parametrize("attention_format", ["attention24", "decomposed"])
@pytest.mark.parametrize("bad_mask", [[[0, 0, 0]], [[1, 0, 1]], [[1, 0.5, 0]], [1, 1, 1]])
def test_both_formats_reject_invalid_masks(tmp_path, attention_format, bad_mask):
    model, config = make_model(fused=False)
    with pytest.raises(ValueError, match="mask|valid|padding"):
        export_mlpf(model, torch.ones(1, 3, config.input_dim), torch.tensor(bad_mask), tmp_path / "invalid.onnx", attention_format=attention_format)


class OverflowProbe(torch.nn.Module):
    def forward(self, features, mask):
        q = features.half().unsqueeze(1)
        value = (1 + 99 * (~mask.bool())[:, None, :, None]).expand_as(q).to(q.dtype)
        out = export_attention(q, q, value, mask.bool(), attention_format="decomposed", head_dim=16).squeeze(1).float()
        return out, out, out, out


def test_exported_decomposition_overflow_and_key_mask(tmp_path):
    path = tmp_path / "overflow.onnx"
    x = torch.full((1, 2, 16), 200.0)
    mask = torch.tensor([[1.0, 0.0]])
    export_mlpf(OverflowProbe().eval(), x, mask, path, attention_format="decomposed")
    for out in cpu_session(path).run(OUTPUTS, {"Xfeat_normed": x.numpy(), "mask": mask.numpy()}):
        np.testing.assert_array_equal(out[0, 0], np.ones(16))
        np.testing.assert_array_equal(out[0, 1], np.zeros(16))


@pytest.mark.parametrize("head_dim", [3, 8, 16, 80])
def test_decomposition_scale_uses_actual_head_dimension(head_dim):
    torch.manual_seed(3)
    q, k, v = [torch.randn(2, 2, 7, head_dim) for _ in range(3)]
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    actual = export_attention(q, k, v, valid, attention_format="decomposed", head_dim=head_dim)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_dead_node_cleanup_keeps_control_flow_captures():
    h = onnx.helper
    branch = h.make_graph(
        [h.make_node("Identity", ["captured"], ["result"])], "branch", [], [h.make_tensor_value_info("result", onnx.TensorProto.FLOAT, [1])]
    )
    nodes = [
        h.make_node("Constant", [], ["captured"], value=h.make_tensor("c", onnx.TensorProto.FLOAT, [1], [7])),
        h.make_node("If", ["cond"], ["out"], then_branch=branch, else_branch=branch),
    ]
    graph = h.make_model(
        h.make_graph(
            nodes,
            "captures",
            [h.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
            [h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
        ),
        opset_imports=[h.make_opsetid("", 20)],
        ir_version=9,
    )
    _remove_dead_nodes(graph)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"cond": np.array(True)})[0], [7])


@pytest.mark.parametrize("dtype", [onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE])
@pytest.mark.parametrize("positive,negative", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_nonfinite_rewrite_preserves_ieee_semantics(dtype, positive, negative):
    h = onnx.helper
    inputs = [h.make_tensor_value_info("x", dtype, [5])]
    outputs = [h.make_tensor_value_info(name, onnx.TensorProto.BOOL, [5]) for name in ["nan", "inf"]]
    nodes = [h.make_node("IsNaN", ["x"], ["nan"]), h.make_node("IsInf", ["x"], ["inf"], detect_positive=positive, detect_negative=negative)]
    graph = h.make_model(h.make_graph(nodes, "nonfinite", inputs, outputs), opset_imports=[h.make_opsetid("", 20)], ir_version=9)
    _lower_nonfinite(graph)
    session = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.array([np.nan, np.inf, -np.inf, 0, -1], dtype=h.tensor_dtype_to_np_dtype(dtype))
    nan, inf = session.run(None, {"x": x})
    np.testing.assert_array_equal(nan, np.isnan(x))
    np.testing.assert_array_equal(inf, (np.isposinf(x) & bool(positive)) | (np.isneginf(x) & bool(negative)))


def device_session(path, provider, tmp_path):
    assert tuple(map(int, ort.__version__.split(".")[:2])) >= (1, 30), "Device qualification targets ORT >= 1.30"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path / "device")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if provider == "cuda":
        ort.preload_dlls()
        session = ort.InferenceSession(str(path), options, providers=[("CUDAExecutionProvider", {"use_tf32": "0", "sdpa_kernel": "1"})])
    elif "MIGraphXExecutionProvider" in ort.get_available_providers() and not os.environ.get("MLPF_TEST_MIGRAPHX_LIBRARY"):
        session = ort.InferenceSession(str(path), options, providers=["MIGraphXExecutionProvider"])
    else:
        library = os.environ.get("MLPF_TEST_MIGRAPHX_LIBRARY")
        if library:
            ort.register_execution_provider_library("MLPF_MIGraphX", library)
        else:
            import onnxruntime_ep_migraphx as plugin

            for name, library in zip(plugin.get_ep_names(), plugin.get_library_paths(), strict=True):
                ort.register_execution_provider_library(name, library)
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "MIGraphXExecutionProvider"]
        assert devices, "Requested MIGraphX GPU is unavailable"
        options.add_provider_for_devices([devices[0]], {})
        session = ort.InferenceSession(str(path), options)
    expected = "CUDAExecutionProvider" if provider == "cuda" else "MIGraphXExecutionProvider"
    assert expected in session.get_providers()
    session.disable_fallback()
    return session, expected


def device_provenance(provider):
    packages = {}
    for name in [
        "torch",
        "onnx",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-ep-migraphx",
        "migraphx-libs",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    command = (
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]
        if provider == "cuda"
        else ["rocm-smi", "--showproductname", "--showdriverversion"]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        driver = {"command": command, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        driver = {"command": command, "error": str(error)}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "driver": driver,
        "torch_cuda_build": torch.version.cuda,
        "torch_hip_build": torch.version.hip,
        "ort_build": ort.get_build_info(),
        "migraphx_library": os.environ.get("MLPF_TEST_MIGRAPHX_LIBRARY"),
    }


def final_attention_kernels(trace):
    """Inspect the last compiled program, excluding earlier tuning/pass dumps."""
    for block in reversed(trace.split("Pass:")):
        kernels = [line for line in block.splitlines() if "gpu::code_object" in line]
        if kernels:
            return [k for k in kernels if "reduce_max" in k and "reduce_sum" in k and k.count("dot") >= 2]
    return []


def test_fusion_evidence_ignores_tuning_kernels():
    attention = "gpu::code_object[symbol_name=mlir_dot_reduce_max_reduce_sum_dot]"
    separate = "gpu::code_object[symbol_name=mlir_dot]\ngpu::code_object[symbol_name=softmax]"
    assert final_attention_kernels("Pass: tuning\n" + attention + "\nPass: final\n" + separate) == []
    assert final_attention_kernels("Pass: final\n" + attention + "\nPass: constant\n@return") == [attention]


@pytest.mark.parametrize("provider,attention_format", [("cuda", "attention24"), ("migraphx", "decomposed")])
def test_real_device_export(tmp_path, monkeypatch, capfd, provider, attention_format):
    if os.environ.get("MLPF_TEST_PROVIDER", "cpu") != provider:
        pytest.skip(f"Set MLPF_TEST_PROVIDER={provider} to require this GPU test")
    monkeypatch.setenv("ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO", "1")
    if provider == "migraphx":
        monkeypatch.setenv("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "attention")
    require_migraphx_fusion = provider == "migraphx" and os.environ.get("MLPF_TEST_REQUIRE_FUSION") == "1"
    if require_migraphx_fusion:
        monkeypatch.setenv("MIGRAPHX_TRACE_COMPILE", "1")
    torch.set_num_threads(1)
    model, config = make_device_model()
    x, mask = make_batch(config, [13, 7], 17)
    path = tmp_path / "device.onnx"
    export_mlpf(model, x, mask, path, attention_format=attention_format, attention_dtype=torch.float16)
    assert_portable_graph(path, attention_format)
    session, expected = device_session(path, provider, tmp_path)
    report = {"ort": ort.__version__, "provider": expected, "format": attention_format, "provenance": device_provenance(provider)}
    try:
        report["batches"] = exercise_batches(session, model, config)
    except AssertionError as error:
        report["numerical_failure"] = str(error)
        raise
    finally:
        profile = json.loads(Path(session.end_profiling()).read_text())
        report["compute_providers"] = dict(Counter(e["args"]["provider"] for e in profile if "provider" in e.get("args", {})))
        output = capfd.readouterr()
        kernel_log = output.out + output.err
        report["kernel_log"] = kernel_log[-64000:]
        report["migraphx_fusion_check_requested"] = require_migraphx_fusion
        report["migraphx_attention_kernels"] = final_attention_kernels(kernel_log) if require_migraphx_fusion else []
        if require_migraphx_fusion:
            (tmp_path / "compile.log").write_text(kernel_log)
        maps = Path("/proc/self/maps")
        report["loaded_libraries"] = (
            sorted({line.split()[-1] for line in maps.read_text().splitlines() if "/" in line and ".so" in line}) if maps.exists() else []
        )
        (tmp_path / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["compute_providers"] and set(report["compute_providers"]) == {expected}
    if provider == "cuda":
        assert "SdpaKernel=FLASH_ATTENTION" in report["kernel_log"]
    if require_migraphx_fusion:
        expected_count = sum(n.op_type == "Softmax" for n in onnx.load(path).graph.node)
        assert len(report["migraphx_attention_kernels"]) >= expected_count > 0, "Actual MIGraphX attention fusion not found"


def test_migraphx_offline_compile(tmp_path):
    driver = os.environ.get("MLPF_TEST_MIGRAPHX_DRIVER")
    if not driver:
        pytest.skip("Set MLPF_TEST_MIGRAPHX_DRIVER to a MIGraphX 2.18+ driver for hardware-free compilation")
    model, config = make_device_model()
    x, mask = make_batch(config, [113, 71], 128)
    path = tmp_path / "offline.onnx"
    export_mlpf(model, x, mask, path, attention_format="decomposed", attention_dtype=torch.float16)
    unsupported = {"MIGRAPHX_DISABLE_MLIR", "MIGRAPHX_ENABLE_CK", "MIGRAPHX_SET_GEMM_PROVIDER", "MIGRAPHX_ENABLE_MIOPEN_POOLING"}
    env = {k: v for k, v in os.environ.items() if k not in unsupported}
    env["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = "attention"
    command = [
        driver,
        "compile",
        str(path),
        "--gpu-arch",
        "gfx1100",
        "--input-dim",
        "@Xfeat_normed",
        "2",
        "128",
        str(config.input_dim),
        "--input-dim",
        "@mask",
        "2",
        "128",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    (tmp_path / "compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-4000:]
    assert final_attention_kernels(result.stdout + result.stderr), "Compiled attention fusion not found"
