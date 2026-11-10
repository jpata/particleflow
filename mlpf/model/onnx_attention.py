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
        if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE):
            raise ValueError(f"Cannot lower Elu input {node.input[0]!r} with ONNX type {dtype}")
        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)
        prefix = f"mlpf_elu_{index}"
        value = node.input[0]
        if dtype == TensorProto.BFLOAT16:
            # CUDA EP has no BF16 comparison kernel; use FP32 opmath, as
            # PyTorch's BF16 activation does, then round the result once.
            value = prefix + "_fp32"
            nodes.append(helper.make_node("Cast", [node.input[0]], [value], to=TensorProto.FLOAT))
            dtype = TensorProto.FLOAT
        zero = constant(dtype, 0.0)
        one = constant(dtype, 1.0)
        is_negative = f"{prefix}_negative"
        exponent = f"{prefix}_exp"
        negative_value = f"{prefix}_value"
        nodes.append(helper.make_node("Less", [value, zero], [is_negative]))
        nodes.append(helper.make_node("Exp", [value], [exponent]))
        nodes.append(helper.make_node("Sub", [exponent, one], [negative_value]))
        if alpha != 1.0:
            scaled = f"{prefix}_scaled"
            nodes.append(helper.make_node("Mul", [negative_value, constant(dtype, alpha)], [scaled]))
            negative_value = scaled
        result = [prefix + "_fp32_output"] if value != node.input[0] else node.output
        nodes.append(helper.make_node("Where", [is_negative, negative_value, value], result, name=node.name))
        if value != node.input[0]:
            nodes.append(helper.make_node("Cast", result, node.output, to=TensorProto.BFLOAT16))
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
    from mlpf.model.heptv2 import HEPTv2Attention

    settings = []
    for module in model.modules():
        values = {}
        if isinstance(module, SimpleMultiheadAttention):
            values.update(onnx_attention_format=attention_format, onnx_attention_dtype=attention_dtype, use_flash_attn_varlen=False)
        if isinstance(module, HEPTv2Attention) and (attention_dtype is not None or attention_format == "attention24"):
            values.update(onnx_attention_format=attention_format, onnx_attention_dtype=attention_dtype)
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


@contextmanager
def fp32_layer_norms(model):
    """Make the normalization precision policy explicit for AMP/export tests."""
    hooks = [
        module.register_forward_pre_hook(lambda module, args: (args[0].float(), *args[1:]))
        for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


class _FloatOutputs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, mask, source_id=None):
        kwargs = {} if source_id is None else {"source_id": source_id}
        return tuple(value.float() for value in self.model(features, mask, **kwargs))


def export_mlpf(model, features, mask, path, *, source_id=None, attention_format=None, attention_dtype=None, compute_dtype=None):
    """Export elementwise predictions with explicit attention and padding semantics.

    Callers must enforce validate_padding_mask for deployment inputs as well.
    The graph itself cannot raise for malformed masks or empty events. If format
    is omitted, preserve the old export_onnx_fused choice (otherwise decompose).
    Export success establishes graph validity, not GPU fusion or equivalence.
    Normalized models require source_id: int64[B], exported as a dynamic input.
    Features remain raw; normalization tables are embedded in the graph.
    """
    from mlpf.model.mlpf import SimpleMultiheadAttention
    from mlpf.model.heptv2 import HEPTv2Attention
    from contextlib import nullcontext
    import io

    import onnx

    from mlpf.model.onnx_graph import finalize_export

    modules = [m for m in model.modules() if isinstance(m, (SimpleMultiheadAttention, HEPTv2Attention))]
    attention_format = attention_format or ("attention24" if any(getattr(m, "export_onnx_fused", False) for m in modules) else "decomposed")
    if attention_format not in ("attention24", "decomposed"):
        raise ValueError(f"Unknown attention format: {attention_format}")
    if attention_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Attention export supports float16, bfloat16 or float32")
    if compute_dtype not in (None, torch.bfloat16):
        raise ValueError("Full-model AMP export currently supports bfloat16")
    if model.training:
        raise ValueError("Export requires model.eval()")
    if getattr(getattr(model, "output_mode", None), "value", None) == "set":
        raise ValueError("This export helper requires elementwise outputs; set predictions need a separate output contract")
    validate_padding_mask(mask)
    if features.ndim != 3 or features.shape[:2] != mask.shape or not torch.isfinite(features).all():
        raise ValueError("Export requires finite features[B,L,F] matching mask[B,L]")
    uses_normalization = getattr(model, "input_normalization", None) is not None
    inputs = (features, mask)
    input_names = ["Xfeat_normed", "mask"]
    dynamic_axes = {name: {0: "num_batch", 1: "num_elements"} for name in (*input_names, "bid", "id", "momentum", "pu")}
    if uses_normalization:
        from mlpf.conf import SOURCE_IDS

        if source_id is None or source_id.dtype != torch.int64 or source_id.shape != features.shape[:1]:
            raise ValueError("Normalized export requires source_id: int64[B], one dataset source per event")
        versions = model.input_normalization.versions
        allowed = source_id.new_tensor([SOURCE_IDS[name] for name in versions])
        if not torch.isin(source_id, allowed).all():
            raise ValueError("Export source_id must select a dataset with configured normalization statistics")
        source_id = source_id.to(features.device)
        inputs += (source_id,)
        input_names.append("source_id")
        dynamic_axes["source_id"] = {0: "num_batch"}
    elif source_id is not None:
        raise ValueError("source_id is only an ONNX input for models with input normalization")
    normalization = fp32_layer_norms(model) if compute_dtype is not None else nullcontext()
    exported_model = _FloatOutputs(model).eval() if compute_dtype is not None else model
    with (
        _export_settings(model, attention_format, attention_dtype),
        _output_widths(model) as widths,
        normalization,
        torch.autocast(features.device.type, dtype=compute_dtype, enabled=compute_dtype is not None, cache_enabled=False),
    ):
        buffer = io.BytesIO()
        torch.onnx.export(
            exported_model,
            inputs,
            buffer,
            opset_version=20,
            custom_opsets={"mlpf": 1} if modules and attention_format == "attention24" else {},
            input_names=input_names,
            output_names=["bid", "id", "momentum", "pu"],
            dynamic_axes=dynamic_axes,
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
        elif compute_dtype is not None:
            _lower_elu_for_cuda(graph)
        finalize_export(graph, attention_format, attention_dtype, output_widths=widths)
        if uses_normalization:
            import json

            metadata = {prop.key: prop.value for prop in graph.metadata_props}
            metadata.update(
                input_normalization=json.dumps(
                    {SOURCE_IDS[name]: {"dataset": name, "version": version} for name, version in versions.items()}, sort_keys=True
                ),
                source_id_contract="int64[B]; one source per event, from the input_normalization metadata; no external feature normalization",
            )
            onnx.helper.set_model_props(graph, metadata)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(path))
