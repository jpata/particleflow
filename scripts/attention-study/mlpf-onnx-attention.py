"""Length-aware ONNX Attention-24 export, without global SDPA overrides."""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def validate_padding_mask(mask):
    """The deployment contract is nonempty events with binary right padding."""
    valid = mask.bool()
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Attention export requires a binary validity mask")
    if not torch.all(valid.any(dim=-1)):
        raise ValueError("Attention export requires at least one valid element per event")
    if torch.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError("Attention export requires right padding (no holes or left padding)")


class _Attention24(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, lengths):
        valid = torch.arange(key.shape[2], device=key.device)[None, :] < lengths[:, None]
        with sdpa_kernel(SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=valid[:, None, None, :])

    @staticmethod
    def symbolic(g, query, key, value, lengths):
        # Attention-24: Q,K,V,attn_mask,past_key,past_value,nonpad_kv_seqlen.
        # Lengths preserve FlashAttention eligibility; an explicit mask does not.
        from torch.onnx import symbolic_helper

        empty = symbolic_helper._optional_input_placeholder_tensor(g)
        # The legacy PyTorch exporter cannot infer all model ops at opset 24.
        # A typed placeholder survives opset-20 tracing and is lowered below.
        return g.op("mlpf::Attention24", query, key, value, empty, empty, empty, lengths, is_causal_i=0).setType(query.type())


def export_attention(query, key, value, valid):
    if not torch.onnx.is_in_onnx_export():
        validate_padding_mask(valid)
    return _Attention24.apply(query, key, value, valid.to(torch.int64).sum(dim=-1))


def _lower_elu_for_cuda(graph):
    """Avoid ORT CUDA's CPU placement of Elu-22 after the opset-24 upgrade."""
    import onnx
    from onnx import TensorProto, helper

    inferred = onnx.shape_inference.infer_shapes(graph)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    constants = {}
    nodes = []

    def constant(dtype, value):
        key = (dtype, value)
        if key not in constants:
            name = f"mlpf_elu_constant_{len(constants)}"
            nodes.append(helper.make_node("Constant", [], [name], value=helper.make_tensor(name, dtype, [], [value])))
            constants[key] = name
        return constants[key]

    for index, node in enumerate(graph.graph.node):
        if node.domain or node.op_type != "Elu":
            nodes.append(node)
            continue
        dtype = types.get(node.input[0])
        if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
            raise ValueError(f"Cannot lower Elu input {node.input[0]!r} with ONNX type {dtype}")
        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)
        zero = constant(dtype, 0.0)
        one = constant(dtype, 1.0)
        prefix = f"mlpf_elu_{index}"
        is_negative = f"{prefix}_negative"
        exponent = f"{prefix}_exp"
        negative_value = f"{prefix}_value"
        nodes.append(helper.make_node("Less", [node.input[0], zero], [is_negative]))
        nodes.append(helper.make_node("Exp", [node.input[0]], [exponent]))
        nodes.append(helper.make_node("Sub", [exponent, one], [negative_value]))
        if alpha != 1.0:
            scaled = f"{prefix}_scaled"
            nodes.append(helper.make_node("Mul", [negative_value, constant(dtype, alpha)], [scaled]))
            negative_value = scaled
        nodes.append(helper.make_node("Where", [is_negative, negative_value, node.input[0]], node.output, name=node.name))
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)


def export_mlpf(model, features, mask, path, *, attention_dtype=None):
    """Export an eval model; fused=True modules emit Attention-24.

    Callers must enforce validate_padding_mask for deployment inputs as well.
    The graph itself cannot raise for malformed masks or empty events.
    """
    from mlpf.model.mlpf import SimpleMultiheadAttention
    import io

    import onnx

    modules = [m for m in model.modules() if isinstance(m, SimpleMultiheadAttention) and m.export_onnx_fused]
    if model.training:
        raise ValueError("Export requires model.eval()")
    if modules:
        validate_padding_mask(mask)
    old_dtypes = [m.onnx_attention_dtype for m in modules]
    try:
        for m in modules:
            m.onnx_attention_dtype = attention_dtype
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            (features, mask),
            buffer,
            opset_version=20,
            custom_opsets={"mlpf": 1} if modules else {},
            input_names=["Xfeat_normed", "mask"],
            output_names=["bid", "id", "momentum", "pu"],
            dynamic_axes={name: {0: "num_batch", 1: "num_elements"} for name in ("Xfeat_normed", "mask", "bid", "id", "momentum", "pu")},
            dynamo=False,
        )
        graph = onnx.load_model_from_string(buffer.getvalue())
        if modules:
            # Upgrade standard operators with ONNX's adapters, rather than just
            # relabeling their opset. No custom operator remains in the artifact.
            graph = onnx.version_converter.convert_version(graph, 24)
            for node in graph.graph.node:
                if node.domain == "mlpf" and node.op_type == "Attention24":
                    node.domain = ""
                    node.op_type = "Attention"
            _lower_elu_for_cuda(graph)
            imports = [opset for opset in graph.opset_import if opset.domain != "mlpf"]
            del graph.opset_import[:]
            graph.opset_import.extend(imports)
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
    finally:
        for m, dtype in zip(modules, old_dtypes):
            m.onnx_attention_dtype = dtype
