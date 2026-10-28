"""Provider-selection and placement contracts used by the dataset validator."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.model.onnx_runtime import create_session, finish_profile  # noqa: E402


def test_cpu_session_retains_real_placement_evidence(tmp_path):
    h = onnx.helper
    graph = h.make_model(
        h.make_graph(
            [h.make_node("Identity", ["x"], ["y"])],
            "identity",
            [h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        ),
        opset_imports=[h.make_opsetid("", 20)],
        ir_version=9,
    )
    path = tmp_path / "model.onnx"
    onnx.save(graph, path)
    session = create_session(path, "cpu", profile_prefix=tmp_path / "cpu")
    np.testing.assert_array_equal(session.run(None, {"x": np.array([1, 2], dtype=np.float32)})[0], [1, 2])
    report = finish_profile(session, "cpu")
    assert report["full_provider_placement"]
    assert set(report["compute_providers"]) == {"CPUExecutionProvider"}


def test_requested_unavailable_provider_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider.*unavailable"):
        create_session(tmp_path / "unused.onnx", "cuda")


def test_cpu_cannot_be_qualified_as_gpu(tmp_path):
    with pytest.raises(ValueError, match="GPU"):
        create_session(tmp_path / "unused.onnx", "cpu", require_full_gpu=True)


def test_migraphx_plugin_registration_is_reused_between_sessions(tmp_path, monkeypatch):
    import mlpf.model.onnx_runtime as runtime

    devices = []
    registrations = []
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"] + (["MIGraphXExecutionProvider"] if devices else []))
    monkeypatch.setattr(ort, "get_ep_devices", lambda: devices, raising=False)

    def register(name, library):
        registrations.append((name, library))
        devices.append(SimpleNamespace(ep_name="MIGraphXExecutionProvider"))

    monkeypatch.setattr(ort, "register_execution_provider_library", register, raising=False)
    plugin = SimpleNamespace(get_ep_names=lambda: ["MIGraphXExecutionProvider"], get_library_paths=lambda: ["plugin.so"])
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: plugin)
    monkeypatch.setattr(ort, "SessionOptions", lambda: SimpleNamespace(add_provider_for_devices=lambda devices, settings: None))
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda *args, **kwargs: SimpleNamespace(get_providers=lambda: ["MIGraphXExecutionProvider"], disable_fallback=lambda: None),
    )
    for _ in range(2):
        create_session(tmp_path / "model.onnx", "migraphx")
    assert registrations == [("MIGraphXExecutionProvider", "plugin.so")]


@pytest.mark.parametrize("providers", [[], ["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]])
def test_full_gpu_requires_profile_evidence_without_cpu_nodes(tmp_path, providers):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([{"cat": "Node", "args": {"provider": p}} for p in providers]))

    class Session:
        def end_profiling(self):
            return str(path)

    with pytest.raises(RuntimeError, match="placement"):
        finish_profile(Session(), "cuda", require_full_gpu=True)
