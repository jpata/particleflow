"""Common portability rewrites for elementwise MLPF ONNX exports."""

import json

import onnx
from onnx import TensorProto, helper


def _constant(name, dtype, value, dims=()):
    return helper.make_node("Constant", [], [name], value=helper.make_tensor(name, dtype, list(dims), value))


def _input_padding(graph):
    dtype = graph.graph.input[0].type.tensor_type.elem_type
    nodes = [
        helper.make_node("Cast", ["mask"], ["mlpf_export_valid"], to=TensorProto.BOOL),
        _constant("mlpf_export_mask_axis", TensorProto.INT64, [2], [1]),
        helper.make_node("Unsqueeze", ["mlpf_export_valid", "mlpf_export_mask_axis"], ["mlpf_export_query_mask"]),
        _constant("mlpf_export_zero", dtype, [0]),
        helper.make_node("Where", ["mlpf_export_query_mask", "Xfeat_normed", "mlpf_export_zero"], ["mlpf_export_features"]),
    ]
    for node in graph.graph.node:
        for i, value in enumerate(node.input):
            if value == "Xfeat_normed":
                node.input[i] = "mlpf_export_features"
    return nodes


def _output_padding(graph, output_widths):
    outputs = graph.graph.output
    widths = output_widths
    dtypes = {value.type.tensor_type.elem_type for value in outputs}
    if len(outputs) != 4 or not all(widths) or len(dtypes) != 1:
        raise ValueError("Export needs four elementwise outputs with known widths and a common dtype")
    for value, width in zip(outputs, widths, strict=True):
        value.type.tensor_type.shape.dim[-1].dim_value = width
    names = {value.name: f"mlpf_export_prediction_{i}" for i, value in enumerate(outputs)}
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = names.get(name, name)
        for i, name in enumerate(node.output):
            node.output[i] = names.get(name, name)
    return [
        _constant("mlpf_export_output_zero", next(iter(dtypes)), [0]),
        # A final Split exposes strided views of a shared output allocation.
        # MIGraphX EP can bind those views to smaller contiguous ORT buffers,
        # corrupting adjacent outputs. Materialize each masked head instead.
        *[helper.make_node("Where", ["mlpf_export_query_mask", prediction, "mlpf_export_output_zero"], [name]) for name, prediction in names.items()],
    ]


def _isinf_nodes(node, types, producers, index):
    value = node.input[0]
    producer = producers.get(value)
    # Torch promotes to double solely for this test. Widening cannot create inf.
    if (
        producer is not None
        and producer.op_type == "Cast"
        and types.get(producer.input[0]) in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16)
    ):
        if types.get(value) == TensorProto.DOUBLE:
            value = producer.input[0]
    dtype = types[value]
    prefix = f"mlpf_export_inf_{index}"
    flags = {attr.name: attr.i for attr in node.attribute}
    positive, negative = flags.get("detect_positive", 1), flags.get("detect_negative", 1)
    if positive and negative:
        return [
            _constant(prefix + "_constant", dtype, [float("inf")]),
            helper.make_node("Abs", [value], [prefix + "_abs"]),
            helper.make_node("Equal", [prefix + "_abs", prefix + "_constant"], list(node.output)),
        ]
    if positive or negative:
        return [
            _constant(prefix + "_constant", dtype, [float("inf") if positive else -float("inf")]),
            helper.make_node("Equal", [value, prefix + "_constant"], list(node.output)),
        ]
    return [
        _constant(prefix + "_constant", dtype, [float("inf")]),
        helper.make_node("Equal", [value, prefix + "_constant"], [prefix + "_positive"]),
        helper.make_node("Not", [prefix + "_positive"], [prefix + "_not_positive"]),
        helper.make_node("And", [prefix + "_positive", prefix + "_not_positive"], list(node.output)),
    ]


def _lower_logsumexp(graph):
    """Keep exp inputs bounded when a provider lowers LogSumExp naively.

    MIGraphX 2.18 parses ReduceLogSumExp as log(sum(exp(x))). HEPTv2's
    trained attention scores can exceed the FP32 exponential range. Shift by
    the maximum explicitly; retain +/-inf semantics by using a zero shift
    for infinite maxima. This changes export only, not the eager reference.
    """
    if not any(n.domain == "" and n.op_type == "ReduceLogSumExp" for n in graph.graph.node):
        return
    inferred = onnx.shape_inference.infer_shapes(graph)
    types = {v.name: v.type.tensor_type.elem_type for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    version = next(v.version for v in graph.opset_import if v.domain == "")
    nodes = []
    for index, node in enumerate(graph.graph.node):
        if node.domain != "" or node.op_type != "ReduceLogSumExp":
            nodes.append(node)
            continue
        prefix = f"mlpf_export_lse_{index}"
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        keepdims = attrs.get("keepdims", 1)
        reduction_attrs = {"keepdims": 1}
        if "noop_with_empty_axes" in attrs:
            reduction_attrs["noop_with_empty_axes"] = attrs["noop_with_empty_axes"]
        axes = list(node.input[1:])
        if "axes" in attrs:
            axes = [prefix + "_axes"]
            nodes.append(_constant(axes[0], TensorProto.INT64, attrs["axes"], [len(attrs["axes"])]))
        maximum_inputs = [node.input[0], *axes]
        if version < 18:
            maximum_inputs = [node.input[0]]
            if "axes" in attrs:
                reduction_attrs["axes"] = attrs["axes"]
        dtype = types[node.input[0]]
        nodes += [
            helper.make_node("ReduceMax", maximum_inputs, [prefix + "_max"], **reduction_attrs),
            _constant(prefix + "_inf", dtype, [float("inf")]),
            _constant(prefix + "_zero", dtype, [0]),
            helper.make_node("Abs", [prefix + "_max"], [prefix + "_abs"]),
            helper.make_node("Equal", [prefix + "_abs", prefix + "_inf"], [prefix + "_infinite"]),
            helper.make_node("Where", [prefix + "_infinite", prefix + "_zero", prefix + "_max"], [prefix + "_shift"]),
            helper.make_node("Sub", [node.input[0], prefix + "_shift"], [prefix + "_centered"]),
            helper.make_node("Exp", [prefix + "_centered"], [prefix + "_exp"]),
            helper.make_node(
                "ReduceSum",
                [prefix + "_exp", *axes],
                [prefix + "_sum"],
                keepdims=keepdims,
                **({"noop_with_empty_axes": attrs["noop_with_empty_axes"]} if "noop_with_empty_axes" in attrs else {}),
            ),
            helper.make_node("Log", [prefix + "_sum"], [prefix + "_log"]),
            helper.make_node("Shape", [prefix + "_sum"], [prefix + "_shape"]),
            helper.make_node("Reshape", [prefix + "_shift", prefix + "_shape"], [prefix + "_offset"]),
            helper.make_node("Add", [prefix + "_offset", prefix + "_log"], list(node.output)),
        ]
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)


def _lower_nonfinite(graph):
    inferred = onnx.shape_inference.infer_shapes(graph)
    types = {value.name: value.type.tensor_type.elem_type for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    producers = {value: node for node in graph.graph.node for value in node.output}
    nodes = []
    for index, node in enumerate(graph.graph.node):
        if node.domain == "" and node.op_type == "IsNaN":
            name = f"mlpf_export_not_nan_{index}"
            nodes += [helper.make_node("Equal", [node.input[0], node.input[0]], [name]), helper.make_node("Not", [name], list(node.output))]
        elif node.domain == "" and node.op_type == "IsInf":
            nodes.extend(_isinf_nodes(node, types, producers, index))
        else:
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)


def _widen_bf16_comparisons(graph):
    """BF16-to-FP32 widening is exact, including nonfinite classification.

    ORT CUDA 1.29 does not register BF16 comparison kernels. Keep arithmetic
    BF16 where requested while computing these predicates in FP32.
    """
    inferred = onnx.shape_inference.infer_shapes(graph)
    types = {v.name: v.type.tensor_type.elem_type for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    types.update({v.name: v.data_type for v in inferred.graph.initializer})
    nodes, widened = [], {}
    for index, node in enumerate(graph.graph.node):
        if node.domain == "" and node.op_type in {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"}:
            for i, name in enumerate(node.input):
                if types.get(name) == TensorProto.BFLOAT16:
                    if name not in widened:
                        widened[name] = f"mlpf_bf16_predicate_{index}_{i}"
                        nodes.append(helper.make_node("Cast", [name], [widened[name]], to=TensorProto.FLOAT))
                    node.input[i] = widened[name]
        nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)


def _node_inputs(node):
    used = set(node.input)
    for attribute in node.attribute:
        subgraphs = [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else attribute.graphs
        for subgraph in subgraphs:
            local = {v.name for v in (*subgraph.input, *subgraph.initializer)} | {v for n in subgraph.node for v in n.output}
            used.update(set().union(*(_node_inputs(n) for n in subgraph.node)) - local)
    return used


def _raw_bf16_literals(graph):
    """Use exact raw BF16 bits; MIGraphX cannot parse BF16 int32_data literals."""
    from onnx import numpy_helper

    def tensor(value):
        if value.data_type == TensorProto.BFLOAT16 and not value.raw_data:
            value.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(value), name=value.name))

    def visit(body):
        for value in body.initializer:
            tensor(value)
        for node in body.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    tensor(attr.t)
                elif attr.type == onnx.AttributeProto.GRAPH:
                    visit(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for child in attr.graphs:
                        visit(child)

    visit(graph.graph)


def _remove_dead_nodes(graph):
    used = {value.name for value in graph.graph.output}
    nodes = []
    for node in reversed(graph.graph.node):
        if any(value in used for value in node.output):
            used.update(_node_inputs(node))
            nodes.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(reversed(nodes))
    # Intermediate value_info may refer to removed casts or renamed outputs.
    del graph.graph.value_info[:]


def finalize_export(graph, attention_format, attention_dtype, *, output_widths):
    import torch

    prefix = _input_padding(graph)
    suffix = _output_padding(graph, output_widths)
    nodes = prefix + list(graph.graph.node) + suffix
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    _lower_logsumexp(graph)
    _lower_nonfinite(graph)
    _widen_bf16_comparisons(graph)
    _remove_dead_nodes(graph)
    _raw_bf16_literals(graph)
    metadata = {prop.key: prop.value for prop in graph.metadata_props}
    metadata.update(
        attention_format=attention_format,
        attention_dtype=str(attention_dtype) if attention_dtype is not None else "model dtype",
        mask_contract="binary [B,L]; nonempty valid prefix then right padding; finite features; padded outputs zero",
        target_onnxruntime="1.30.0",
        export_versions=json.dumps({"torch": torch.__version__, "onnx": onnx.__version__}),
        qualification="Graph validity only; GPU placement, fusion and numerical compatibility require device tests",
    )
    helper.set_model_props(graph, metadata)
