"""Explicit ONNX attention representations, independent of eager kernel selection."""

import math
from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def validate_padding_mask(mask):
    """The deployment contract is nonempty events with binary right padding."""
    if mask.ndim != 2:
        raise ValueError("Attention export requires a mask of shape [batch, sequence]")
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
        from onnx import TensorProto

        empty = symbolic_helper._optional_input_placeholder_tensor(g)
        # The legacy PyTorch exporter cannot infer all model ops at opset 24.
        # A typed placeholder survives opset-20 tracing and is lowered below.
        return g.op(
            "mlpf::Attention24", query, key, value, empty, empty, empty, lengths, is_causal_i=0, softmax_precision_i=TensorProto.FLOAT
        ).setType(query.type())


def decomposed_attention(query, key, value, valid, head_dim):
    """Materializable attention with FP32 normalization and conservative Q scaling.

    Scaling Q by a power of two reduces FP16 dot-product overflow risk; it does
    not guarantee safety for unbounded inputs. Nonempty key sets are required.
    """
    q_scale = 2.0 ** -math.ceil(math.log2(head_dim))
    scores = torch.matmul(query * q_scale, key.transpose(-2, -1)).float()
    scores = scores * (head_dim**-0.5 / q_scale)
    scores = scores.masked_fill(~valid.bool()[:, None, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.matmul(probabilities, value)


def export_attention(query, key, value, valid, *, attention_format="attention24", head_dim=None):
    if not torch.onnx.is_in_onnx_export():
        validate_padding_mask(valid)
    if attention_format == "decomposed":
        return decomposed_attention(query, key, value, valid, head_dim if head_dim is not None else int(query.shape[-1]))
    if attention_format != "attention24":
        raise ValueError(f"Unknown attention format: {attention_format}")
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


@contextmanager
def _output_widths(model):
    widths = []

    def record(module, args, outputs):
        if len(outputs) != 4 or any(output.ndim != 3 for output in outputs):
            raise ValueError("Export requires four [B,L,F] elementwise predictions")
        # Feature widths are architecture constants; only B and L are dynamic.
        widths[:] = [int(output.shape[-1]) for output in outputs]

    hook = model.register_forward_hook(record)
    try:
        yield widths
    finally:
        hook.remove()


@contextmanager
def _export_settings(model, attention_format, attention_dtype):
    from mlpf.model.mlpf import SimpleMultiheadAttention

    settings = []
    for module in model.modules():
        values = {}
        if isinstance(module, SimpleMultiheadAttention):
            values.update(onnx_attention_format=attention_format, onnx_attention_dtype=attention_dtype, use_flash_attn_varlen=False)
        if hasattr(module, "use_jagged_attention"):
            values["use_jagged_attention"] = False
        for name, value in values.items():
            settings.append((module, name, getattr(module, name)))
            setattr(module, name, value)
    try:
        yield
    finally:
        for module, name, value in reversed(settings):
            setattr(module, name, value)


def export_mlpf(model, features, mask, path, *, attention_format=None, attention_dtype=None):
    """Export elementwise predictions with explicit attention and padding semantics.

    Callers must enforce validate_padding_mask for deployment inputs as well.
    The graph itself cannot raise for malformed masks or empty events. If format
    is omitted, preserve the old export_onnx_fused choice (otherwise decompose).
    Export success establishes graph validity, not GPU fusion or equivalence.
    """
    from mlpf.model.mlpf import SimpleMultiheadAttention
    import io

    import onnx

    from mlpf.model.onnx_graph import finalize_export

    modules = [m for m in model.modules() if isinstance(m, SimpleMultiheadAttention)]
    attention_format = attention_format or ("attention24" if any(m.export_onnx_fused for m in modules) else "decomposed")
    if attention_format not in ("attention24", "decomposed"):
        raise ValueError(f"Unknown attention format: {attention_format}")
    if attention_dtype not in (None, torch.float16, torch.float32):
        raise ValueError("Attention export supports float16 or float32")
    if model.training:
        raise ValueError("Export requires model.eval()")
    if getattr(getattr(model, "output_mode", None), "value", None) == "set":
        raise ValueError("This export helper requires elementwise outputs; set predictions need a separate output contract")
    validate_padding_mask(mask)
    if features.ndim != 3 or features.shape[:2] != mask.shape or not torch.isfinite(features).all():
        raise ValueError("Export requires finite features[B,L,F] matching mask[B,L]")
    with _export_settings(model, attention_format, attention_dtype), _output_widths(model) as widths:
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            (features, mask),
            buffer,
            opset_version=20,
            custom_opsets={"mlpf": 1} if modules and attention_format == "attention24" else {},
            input_names=["Xfeat_normed", "mask"],
            output_names=["bid", "id", "momentum", "pu"],
            dynamic_axes={name: {0: "num_batch", 1: "num_elements"} for name in ("Xfeat_normed", "mask", "bid", "id", "momentum", "pu")},
            dynamo=False,
        )
        graph = onnx.load_model_from_string(buffer.getvalue())
        if modules and attention_format == "attention24":
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
        finalize_export(graph, attention_format, attention_dtype, output_widths=widths)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(path))
