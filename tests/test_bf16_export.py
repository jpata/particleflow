"""BF16 exports retain FP32 external I/O and restore model precision settings."""

import pytest
import torch

onnx = pytest.importorskip("onnx")
from mlpf.model.onnx_attention import export_mlpf  # noqa: E402
from mlpf.model.mlpf import SimpleMultiheadAttention  # noqa: E402
from mlpf.model.heptv2 import HEPTv2Attention  # noqa: E402
from tests.test_attention24_export import make_model, make_batch  # noqa: E402
from tests.test_heptv2_export import make_heptv2, batch  # noqa: E402


@pytest.mark.parametrize("kind", ["attention", "heptv2"])
@pytest.mark.parametrize("format", ["attention24", "decomposed"])
def test_bf16_amp_export(tmp_path, kind, format):
    torch.set_num_threads(1)
    if kind == "attention":
        model, config = make_model(fused=False)
        x, mask = make_batch(config, [7, 3], 8)
    else:
        model, config = make_heptv2()
        x, mask = batch(config, [7, 3], 8)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    path = tmp_path / "bf16.onnx"
    export_mlpf(model, x, mask, path, attention_format=format, attention_dtype=torch.bfloat16, compute_dtype=torch.bfloat16)
    graph = onnx.load(path)
    onnx.checker.check_model(graph, full_check=True)
    assert all(v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for v in [*graph.graph.input, *graph.graph.output])
    assert any(
        n.op_type == "Cast" and any(a.name == "to" and a.i == onnx.TensorProto.BFLOAT16 for a in n.attribute) for n in graph.graph.node
    ) or any(v.data_type == onnx.TensorProto.BFLOAT16 for v in graph.graph.initializer)
    assert sum(n.op_type == "Attention" for n in graph.graph.node) == (
        sum(isinstance(m, (SimpleMultiheadAttention, HEPTv2Attention)) for m in model.modules()) if format == "attention24" else 0
    )
    assert not any(n.op_type == "ReduceLogSumExp" for n in graph.graph.node)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(getattr(m, "onnx_attention_format", None) is None for m in model.modules())
    assert all(getattr(m, "onnx_attention_dtype", None) is None for m in model.modules())
    assert all(not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize("operator", ["Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual"])
@pytest.mark.parametrize("initializer", [False, True])
def test_bf16_predicates_widen_exactly(operator, initializer):
    import numpy as np
    import ml_dtypes
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    from onnx.reference import ReferenceEvaluator
    from mlpf.model.onnx_graph import _widen_bf16_comparisons

    values = np.array([-np.inf, -1, -0.0, 0.0, 1, np.inf, np.nan], dtype=ml_dtypes.bfloat16)
    graph = h.make_model(
        h.make_graph(
            [
                h.make_node("Constant", [], ["bound"], value=nh.from_array(values, "bound")),
                h.make_node(operator, ["x", "bound"], ["y"]),
            ],
            "predicates",
            [h.make_tensor_value_info("x", T.BFLOAT16, [7])],
            [h.make_tensor_value_info("y", T.BOOL, [7])],
        ),
        opset_imports=[h.make_opsetid("", 20)],
    )
    if initializer:
        graph.graph.initializer.append(graph.graph.node[0].attribute[0].t)
        del graph.graph.node[0]
    _widen_bf16_comparisons(graph)
    onnx.checker.check_model(graph, full_check=True)
    candidate = ReferenceEvaluator(graph).run(None, {"x": values})[0]
    operation = {"Equal": np.equal, "Less": np.less, "Greater": np.greater, "LessOrEqual": np.less_equal, "GreaterOrEqual": np.greater_equal}[
        operator
    ]
    np.testing.assert_array_equal(candidate, operation(values.astype(np.float32), values.astype(np.float32)))
    assert sum(n.op_type == "Cast" for n in graph.graph.node) == 2


def test_bf16_literal_raw_encoding_preserves_bits():
    import numpy as np
    from onnx import helper as h, TensorProto as T, numpy_helper as nh
    from mlpf.model.onnx_graph import _raw_bf16_literals

    tensor = h.make_tensor("literal", T.BFLOAT16, [5], [-np.inf, -1, 0, 1, np.inf])
    graph = h.make_model(
        h.make_graph([h.make_node("Constant", [], ["y"], value=tensor)], "literal", [], [h.make_tensor_value_info("y", T.BFLOAT16, [5])])
    )
    before = nh.to_array(tensor).view(np.uint16).copy()
    _raw_bf16_literals(graph)
    after = graph.graph.node[0].attribute[0].t
    assert after.raw_data and not after.int32_data
    np.testing.assert_array_equal(nh.to_array(after).view(np.uint16), before)
