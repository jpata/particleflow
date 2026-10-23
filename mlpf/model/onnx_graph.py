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
    if producer is not None and producer.op_type == "Cast" and types.get(producer.input[0]) in (TensorProto.FLOAT, TensorProto.FLOAT16):
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


def _node_inputs(node):
    used = set(node.input)
    for attribute in node.attribute:
        subgraphs = [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else attribute.graphs
        for subgraph in subgraphs:
            local = {v.name for v in (*subgraph.input, *subgraph.initializer)} | {v for n in subgraph.node for v in n.output}
            used.update(set().union(*(_node_inputs(n) for n in subgraph.node)) - local)
    return used


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
    _lower_nonfinite(graph)
    _remove_dead_nodes(graph)
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
