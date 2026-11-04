"""Portable log-sum-exp must handle trained-score ranges and infinities."""

import os

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from mlpf.model.onnx_graph import _lower_logsumexp  # noqa: E402


def session_for(model):
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider"}[
        os.environ.get("MLPF_TEST_PROVIDER", "cpu")
    ]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if provider != "CPUExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(model.SerializeToString(), options, providers=[provider])
    session.disable_fallback()
    return session


@pytest.mark.parametrize("opset", [17, 20])
@pytest.mark.parametrize("keepdims", [0, 1])
def test_logsumexp_extreme_scores_and_nonfinite_values(opset, keepdims):
    x = np.array([[100, 101], [-1000, -1001], [np.inf, 1], [-np.inf, -np.inf], [np.nan, 1], [1, 2]], dtype=np.float32)
    expected = np.array([101 + np.log1p(np.exp(-1)), -1000 + np.log1p(np.exp(-1)), np.inf, -np.inf, np.nan, 2 + np.log1p(np.exp(-1))])
    if keepdims:
        expected = expected[:, None]
    axes = onnx.numpy_helper.from_array(np.array([-1], np.int64), "axes")
    inputs = ["x", "axes"] if opset >= 18 else ["x"]
    attrs = {"keepdims": keepdims, **({"axes": [-1]} if opset < 18 else {})}
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("ReduceLogSumExp", inputs, ["y"], **attrs)],
            "logsumexp",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(x.shape))],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, list(expected.shape))],
            initializer=[axes] if opset >= 18 else [],
        ),
        opset_imports=[onnx.helper.make_opsetid("", opset)],
        ir_version=9,
    )
    _lower_logsumexp(model)
    onnx.checker.check_model(model, full_check=True)
    session = session_for(model)
    actual = session.run(None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-6, equal_nan=True)


@pytest.mark.parametrize("noop", [0, 1])
def test_logsumexp_empty_axes(noop):
    x = np.array([[100, 101], [-1000, -1001]], dtype=np.float32)
    expected = x if noop else np.array([[101 + np.log1p(np.exp(-1))]], np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("ReduceLogSumExp", ["x", "axes"], ["y"], keepdims=1, noop_with_empty_axes=noop)],
            "empty_axes",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(x.shape))],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, list(expected.shape))],
            initializer=[onnx.numpy_helper.from_array(np.array([], np.int64), "axes")],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 20)],
        ir_version=9,
    )
    _lower_logsumexp(model)
    onnx.checker.check_model(model, full_check=True)
    session = session_for(model)
    np.testing.assert_allclose(session.run(None, {"x": x})[0], expected, atol=1e-4, rtol=1e-6)
