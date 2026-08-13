import time
from typing import Union, List, Optional, NamedTuple

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from mlpf.logger import _logger
from mlpf.model.gnnlsh import CombinedGraphLayer

try:
    from mlpf.model.litept import LitePTLayer
except ImportError:
    LitePTLayer = None

from mlpf.model.hept import HEPTLayer, trunc_normal_
from mlpf.model.heptv2 import HEPTv2Layer

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except (ImportError, OSError):
    _flash_attn_varlen_func = None

from mlpf.conf import (
    MLPFConfig,
    Activation,
    AttentionType,
    BackboneMode,
    ModelType,
    InputEncoding,
    LearnedRepresentationMode,
    RegressionMode,
    KernelType,
    Dataset,
)


@torch.compiler.disable
def _flash_attn_varlen_eager(*args, **kwargs):
    # The ROCm FlashAttention custom op can return an auxiliary softmax_lse
    # layout that disagrees with its fake registration under torch.compile.
    return _flash_attn_varlen_func(*args, **kwargs)


def get_activation(activation: Activation):
    activation = Activation(activation)
    if activation == Activation.ELU:
        act = nn.ELU
    elif activation == Activation.RELU:
        act = nn.ReLU
    elif activation == Activation.RELU6:
        act = nn.ReLU6
    elif activation == Activation.LEAKYRELU:
        act = nn.LeakyReLU
    elif activation == Activation.GELU:
        act = nn.GELU
    else:
        raise ValueError(f"Unknown activation {activation}")
    return act


def count_parameters(model):
    if isinstance(model, torch.nn.Parameter):
        return model.numel() if model.requires_grad else 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_num_heads(embed_dim: int, preferred_heads: Optional[int] = None) -> int:
    candidates = []
    if preferred_heads is not None:
        candidates.append(preferred_heads)
    candidates.extend([8, 4, 2, 1])
    for num_heads in candidates:
        if num_heads and embed_dim % num_heads == 0:
            return num_heads
    return 1


class PackedJaggedTensor(NamedTuple):
    values: torch.Tensor
    offsets: torch.Tensor
    cu_seqlens: torch.Tensor
    batch_size: int
    max_seqlen: int

    def with_values(self, values):
        return PackedJaggedTensor(values, self.offsets, self.cu_seqlens, self.batch_size, self.max_seqlen)


def is_jagged_tensor(tensor):
    return isinstance(tensor, PackedJaggedTensor)


def dense_to_jagged(tensor, mask):
    valid = mask.bool()
    valid_lengths = valid.sum(dim=-1)
    offsets = torch.nn.functional.pad(valid_lengths.cumsum(dim=0), (1, 0))
    return PackedJaggedTensor(tensor[valid], offsets, offsets.to(torch.int32), tensor.shape[0], tensor.shape[1])


def jagged_to_dense(tensor, mask):
    output = tensor.values.new_zeros((*mask.shape, tensor.values.shape[-1]))
    return output.masked_scatter(mask.bool().unsqueeze(-1), tensor.values)


class SimpleMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        device=None,
        dtype=None,
        export_onnx_fused=False,
        attention_type: AttentionType = AttentionType.SIMPLE,
        use_flash_attn_varlen=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        bias = True
        batch_first = True
        super().__init__(embed_dim, num_heads, dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.head_dim = int(embed_dim // num_heads)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.export_onnx_fused = export_onnx_fused
        self.attention_type = AttentionType(attention_type)
        self.use_flash_attn_varlen = use_flash_attn_varlen
        if self.use_flash_attn_varlen and self.attention_type != AttentionType.FLASH:
            raise ValueError("flash_attn_varlen requires attention_type='flash'")
        if self.use_flash_attn_varlen and (self.head_dim < 8 or self.head_dim > 256 or self.head_dim % 8 != 0):
            raise ValueError("flash_attn_varlen requires head_dim to be a multiple of 8 between 8 and 256")
        if self.use_flash_attn_varlen and _flash_attn_varlen_func is None:
            raise ImportError("use_flash_attn_varlen=True requires a compatible flash-attn installation " "that provides flash_attn_varlen_func")
        self.attn_params = {
            AttentionType.SIMPLE: [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
            AttentionType.MATH: [SDPBackend.MATH],
            AttentionType.FLASH: [SDPBackend.FLASH_ATTENTION],
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, need_weights=False, key_padding_mask=None) -> torch.Tensor:
        # q, k, v: 3D tensors (batch_size, seq_len, embed_dim), embed_dim = num_heads*head_dim
        jagged_input = is_jagged_tensor(q)
        if jagged_input != is_jagged_tensor(k) or jagged_input != is_jagged_tensor(v):
            raise ValueError("q, k, and v must all use the same dense or jagged layout")
        if jagged_input and (key_padding_mask is not None or self.export_onnx_fused):
            raise ValueError("Jagged attention does not use padding masks and cannot use the ONNX-fused path")
        if self.use_flash_attn_varlen and not jagged_input:
            raise ValueError("flash_attn_varlen_func requires packed jagged q, k, and v")

        if jagged_input:
            bs = q.batch_size
            q_values, k_values, v_values = q.values, k.values, v.values
        else:
            bs = q.size(0)
            k_len = k.size(1)
            q_values, k_values, v_values = q, k, v
        head_dim = self.head_dim
        num_heads = self.num_heads

        # split stacked in_proj_weight, in_proj_bias to q, k, v matrices
        wq, wk, wv = torch.split(self.in_proj_weight, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)
        bq, bk, bv = torch.split(self.in_proj_bias, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)

        q_values = torch.matmul(q_values, wq.T)
        k_values = torch.matmul(k_values, wk.T)
        v_values = torch.matmul(v_values, wv.T)
        # ROCm Inductor 2.7 can fuse autocast matmul+bias into addmm without
        # casting the FP32 bias. Cast to the actual matmul output dtype first.
        q_values = q_values + bq.to(q_values.dtype)
        k_values = k_values + bk.to(k_values.dtype)
        v_values = v_values + bv.to(v_values.dtype)

        # for pytorch internal scaled dot product attention, we need (bs, num_heads, seq_len, head_dim)
        if jagged_input and self.use_flash_attn_varlen:
            q_attn = q_values.reshape(-1, num_heads, head_dim)
            k_attn = k_values.reshape(-1, num_heads, head_dim)
            v_attn = v_values.reshape(-1, num_heads, head_dim)
        elif jagged_input:
            nested_qkv = []
            for tensor in (q_values, k_values, v_values):
                nested = torch.nested.nested_tensor_from_jagged(
                    tensor.reshape(-1, num_heads, head_dim),
                    q.offsets,
                    min_seqlen=1,
                    max_seqlen=q.max_seqlen,
                )
                nested_qkv.append(nested.transpose(1, 2))
            q_attn, k_attn, v_attn = nested_qkv
        elif not self.export_onnx_fused:
            q_attn = q_values.reshape(bs, -1, num_heads, head_dim).transpose(1, 2)
            k_attn = k_values.reshape(bs, -1, num_heads, head_dim).transpose(1, 2)
            v_attn = v_values.reshape(bs, -1, num_heads, head_dim).transpose(1, 2)
        else:
            q_attn, k_attn, v_attn = q_values, k_values, v_values

        # this function will have different shape signatures in native pytorch sdpa and in ONNX com.microsoft.MultiHeadAttention
        # in pytorch: (bs, num_heads, seq_len, head_dim)
        # in ONNX: (bs, seq_len, num_heads*head_dim)

        # Dense flash SDPA does not accept a non-null mask. Callers that need
        # correct variable-length flash attention must pass jagged q/k/v.
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [bs, seq_len], True where elements are to be ignored
            # scaled_dot_product_attention expects attn_mask: [bs, 1, 1, seq_len], False where elements are to be ignored
            if self.export_onnx_fused:
                attn_mask = ~key_padding_mask.unsqueeze(1)
            elif self.attention_type == AttentionType.FLASH:
                raise ValueError("Dense flash attention cannot use key_padding_mask; enable jagged attention")
            else:
                attn_mask = ~key_padding_mask.view(bs, 1, 1, k_len)

        if self.use_flash_attn_varlen:
            flash_attn_varlen = _flash_attn_varlen_eager if torch.version.hip is not None else _flash_attn_varlen_func
            attn_output = flash_attn_varlen(
                q_attn,
                k_attn,
                v_attn,
                q.cu_seqlens,
                k.cu_seqlens,
                q.max_seqlen,
                k.max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )
        elif self.export_onnx_fused:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q_attn, k_attn, v_attn, attn_mask=attn_mask, dropout_p=self.dropout)
        else:
            with sdpa_kernel(self.attn_params[self.attention_type]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(q_attn, k_attn, v_attn, attn_mask=attn_mask, dropout_p=self.dropout)

        # Keep residual paths type-stable for half-precision ONNX export.
        attn_output = attn_output.to(q_values.dtype)

        # in case running with pytorch internal scaled dot product attention, reshape back to the original shape
        if jagged_input:
            if self.use_flash_attn_varlen:
                attn_output = attn_output.reshape(-1, num_heads * head_dim)
            else:
                attn_output = attn_output.transpose(1, 2).values().reshape(-1, num_heads * head_dim)
        elif not self.export_onnx_fused:
            attn_output = attn_output.transpose(1, 2).reshape(bs, -1, num_heads * head_dim)

        # assert list(attn_output.size()) == [bs, seq_len, num_heads * head_dim]
        attn_output_projected = torch.matmul(attn_output, self.out_proj.weight.T)
        attn_output = attn_output_projected + self.out_proj.bias.to(attn_output_projected.dtype)
        if jagged_input:
            attn_output = q.with_values(attn_output)
        return attn_output, None


class PreLnSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        name="",
        activation: Activation = Activation.ELU,
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
        attention_type: AttentionType = AttentionType.SIMPLE,
        learnable_queries=False,
        elems_as_queries=False,
        export_onnx_fused=False,
        save_attention=False,
        use_flash_attn_varlen=False,
    ):
        super(PreLnSelfAttentionLayer, self).__init__()
        self.name = name

        self.attention_type = AttentionType(attention_type)
        self.act = get_activation(activation)

        _logger.info("layer {} using attention_type={} (SimpleMultiheadAttention)".format(self.name, self.attention_type))
        self.mha = SimpleMultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout_mha,
            export_onnx_fused=export_onnx_fused,
            attention_type=self.attention_type,
            use_flash_attn_varlen=use_flash_attn_varlen,
        )

        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act())
        self.dropout = torch.nn.Dropout(dropout_ff)

        self.learnable_queries = learnable_queries
        self.elems_as_queries = elems_as_queries
        if self.learnable_queries:
            self.queries = nn.Parameter(torch.zeros(1, 1, embedding_dim), requires_grad=True)
            trunc_normal_(self.queries, std=0.02)

        # options for saving the attention matrix
        self.save_attention = save_attention
        self.att_mat_idx = 0
        self.outdir = "."

        self.mha_res_norm = None
        self.ffn_res_norm = None
        self.input_norm = None
        self.seq_len = 0

    def forward(self, x, mask, initial_embedding):
        jagged_input = is_jagged_tensor(x)
        self.input_norm = (x.values if jagged_input else x).norm().detach()
        self.seq_len = mask.shape[1] if jagged_input else x.shape[1]

        if mask is not None and not jagged_input:
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = x.with_values(self.norm0(x.values)) if jagged_input else self.norm0(x)

        q = x_norm

        if self.learnable_queries:
            if jagged_input:
                raise ValueError("Learnable per-element queries are not supported with jagged attention")
            q = self.queries.expand(*x.shape)
        elif self.elems_as_queries:
            q = initial_embedding
        if mask is not None and not jagged_input:
            q = q * mask_

        mha_out = self.mha(q, x_norm, x_norm, need_weights=False)[0]

        self.mha_res_norm = (mha_out.values if jagged_input else mha_out).norm().detach()

        x = residual.with_values(residual.values + mha_out.values) if jagged_input else residual + mha_out
        residual = x
        x_norm = x.with_values(self.norm1(x.values)) if jagged_input else self.norm1(x)
        if jagged_input:
            ffn_out = x.with_values(self.dropout(self.seq(x_norm.values)))
        else:
            ffn_out = self.dropout(self.seq(x_norm))

        self.ffn_res_norm = (ffn_out.values if jagged_input else ffn_out).norm().detach()

        x = residual.with_values(residual.values + ffn_out.values) if jagged_input else residual + ffn_out
        if mask is not None and not jagged_input:
            x = x * mask_
        return x


class TaskQueryAttentionReadout(nn.Module):
    def __init__(
        self,
        name="",
        activation: Activation = Activation.ELU,
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
        attention_type: AttentionType = AttentionType.SIMPLE,
        export_onnx_fused=False,
    ):
        super(TaskQueryAttentionReadout, self).__init__()
        self.name = name
        self.attention_type = AttentionType(attention_type)
        self.act = get_activation(activation)

        _logger.info("readout {} using attention_type={} (SimpleMultiheadAttention)".format(self.name, self.attention_type))
        self.mha = SimpleMultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout_mha,
            export_onnx_fused=export_onnx_fused,
            attention_type=self.attention_type,
        )

        self.query_norm = torch.nn.LayerNorm(embedding_dim)
        self.key_norm = torch.nn.LayerNorm(embedding_dim)
        self.post_norm = torch.nn.LayerNorm(embedding_dim)
        self.ffn = torch.nn.Sequential(nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act())
        self.dropout = torch.nn.Dropout(dropout_ff)

    def forward(self, x, mask, query):
        residual = x
        x_norm = self.key_norm(x)
        q = self.query_norm(query.expand(x.shape[0], 1, -1))

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else mask <= 0

        attn_out = self.mha(q, x_norm, x_norm, need_weights=False, key_padding_mask=key_padding_mask)[0]
        attn_out = attn_out.expand(-1, x.shape[1], -1)

        x = residual + attn_out
        residual = x
        x_norm = self.post_norm(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout(ffn_out)
        x = residual + ffn_out

        if mask is not None:
            x = x * (mask.unsqueeze(-1).to(x.dtype) if mask.dtype == torch.bool else mask.unsqueeze(-1))
        return x


def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        act(),
        torch.nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class HitFeatureEngineering(nn.Module):
    """Append inexpensive geometry features to the raw EDM4HEP hit inputs.

    The original feature order is preserved because the regression heads use the
    ET, eta, phi, and energy columns as skip-connection reference values. All
    calculations are elementwise and run on the same device as the model input.
    """

    POSITION_SCALE_MM = 3000.0
    CONFORMAL_SCALE_MM = 1000.0
    TIME_SCALE_NS = 10.0
    SPEED_OF_LIGHT_MM_PER_NS = 299.792458
    NUM_OUTPUT_FEATURES = 15
    OUTPUT_FEATURE_NAMES = (
        "position_x_norm",
        "position_y_norm",
        "position_z_norm",
        "rho_norm",
        "radius_norm",
        "sin_theta",
        "cos_theta",
        "barrel_fraction",
        "conformal_u",
        "conformal_v",
        "time_residual_norm",
        "is_ecal",
        "is_hcal",
        "is_other",
        "is_tracker",
    )

    def forward(self, X_features, mask=None):
        # Geometry is evaluated in float32 to avoid overflow and excessive
        # quantization when training the rest of the model in bfloat16.
        geometry = X_features.to(torch.float32)
        x = geometry[..., 6:7]
        y = geometry[..., 7:8]
        z = geometry[..., 8:9]
        time = geometry[..., 9:10]
        subdetector = geometry[..., 10:11]
        elemtype = geometry[..., 0:1]

        rho_sq = x.square() + y.square()
        rho = torch.sqrt(rho_sq)
        radius = torch.sqrt(rho_sq + z.square())
        safe_radius = radius.clamp_min(1.0e-6)

        position_features = torch.cat(
            [
                x / self.POSITION_SCALE_MM,
                y / self.POSITION_SCALE_MM,
                z / self.POSITION_SCALE_MM,
                rho / self.POSITION_SCALE_MM,
                radius / self.POSITION_SCALE_MM,
                rho / safe_radius,
                z / safe_radius,
                rho / (rho + z.abs()).clamp_min(1.0e-6),
            ],
            dim=-1,
        ).clamp(min=-4.0, max=4.0)

        # Prompt circular tracks become approximately straight in conformal
        # coordinates. Calorimeter conformal coordinates are intentionally zero.
        is_tracker = elemtype == 1
        inverse_rho_sq = torch.where(rho_sq > 0, self.CONFORMAL_SCALE_MM / rho_sq, torch.zeros_like(rho_sq))
        conformal_features = torch.cat([x * inverse_rho_sq, y * inverse_rho_sq], dim=-1)
        conformal_features = torch.where(is_tracker.expand_as(conformal_features), conformal_features, 0.0).clamp(min=-4.0, max=4.0)

        time_residual = (time - radius / self.SPEED_OF_LIGHT_MM_PER_NS) / self.TIME_SCALE_NS
        time_residual = time_residual.clamp(min=-10.0, max=10.0)
        subdetector_features = torch.cat([subdetector == index for index in range(4)], dim=-1).to(torch.float32)

        engineered = torch.cat([position_features, conformal_features, time_residual, subdetector_features], dim=-1)
        engineered = torch.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            engineered = engineered * mask.unsqueeze(-1).to(engineered.dtype)

        return torch.cat([X_features, engineered.to(X_features.dtype)], dim=-1)


class RegressionOutput(nn.Module):
    def __init__(self, mode: RegressionMode, embed_dim, width, act, dropout, elemtypes):
        super(RegressionOutput, self).__init__()
        self.mode = RegressionMode(mode)
        self.elemtypes = elemtypes

        # single output
        if self.mode == RegressionMode.DIRECT or self.mode == RegressionMode.ADDITIVE or self.mode == RegressionMode.MULTIPLICATIVE:
            self.nn = ffn(embed_dim, 1, width, act, dropout)
        elif self.mode == RegressionMode.DIRECT_ELEMTYPE:
            self.nn = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
        elif self.mode == RegressionMode.DIRECT_ELEMTYPE_SPLIT:
            self.nn = nn.ModuleList()
            for elem in range(len(self.elemtypes)):
                self.nn.append(ffn(embed_dim, 1, width, act, dropout))
        # two outputs
        elif self.mode == RegressionMode.LINEAR:
            self.nn = ffn(embed_dim, 2, width, act, dropout)
        elif self.mode == RegressionMode.LINEAR_ELEMTYPE:
            self.nn1 = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
            self.nn2 = ffn(embed_dim, len(self.elemtypes), width, act, dropout)

    def forward(self, elems, x, orig_value):
        if self.mode == RegressionMode.DIRECT:
            nn_out = self.nn(x)
            return nn_out
        elif self.mode == RegressionMode.DIRECT_ELEMTYPE:
            nn_out = self.nn(x)
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            nn_out = torch.sum(elemtype_mask * nn_out, axis=-1, keepdims=True)
            return nn_out
        elif self.mode == RegressionMode.DIRECT_ELEMTYPE_SPLIT:
            elem_outs = []
            for elem in range(len(self.elemtypes)):
                elem_outs.append(self.nn[elem](x))
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            elem_outs = torch.cat(elem_outs, axis=-1)
            return torch.sum(elem_outs * elemtype_mask, axis=-1, keepdims=True)
        elif self.mode == RegressionMode.ADDITIVE:
            nn_out = self.nn(x)
            return orig_value + nn_out
        elif self.mode == RegressionMode.MULTIPLICATIVE:
            nn_out = self.nn(x)
            return orig_value * nn_out
        elif self.mode == RegressionMode.LINEAR:
            nn_out = self.nn(x)
            return orig_value * nn_out[..., 0:1] + nn_out[..., 1:2]
        elif self.mode == RegressionMode.LINEAR_ELEMTYPE:
            nn_out1 = self.nn1(x)
            nn_out2 = self.nn2(x)
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            a = torch.sum(elemtype_mask * nn_out1, axis=-1, keepdims=True)
            b = torch.sum(elemtype_mask * nn_out2, axis=-1, keepdims=True)
            return orig_value * a + b


class MLPF(nn.Module):
    def __init__(
        self,
        config: MLPFConfig,
    ):
        super(MLPF, self).__init__()

        self.config = config.model
        self.raw_input_dim = config.input_dim
        self.feature_engineering = HitFeatureEngineering() if config.dataset in (Dataset.CLD_HITS, Dataset.CLIC_HITS) else nn.Identity()
        self.input_dim = self.raw_input_dim + (HitFeatureEngineering.NUM_OUTPUT_FEATURES if self.uses_hit_feature_engineering else 0)
        self.num_classes = config.num_classes
        self.elemtypes_nonzero = config.elemtypes_nonzero

        # Determine architecture parameters based on the chosen type
        self.conv_type = ModelType(self.config.type)
        sub_config = getattr(self.config, self.conv_type.value)
        if sub_config is None:
            raise ValueError(f"Missing model.{self.conv_type.value} configuration for model type {self.conv_type.value}")

        # Extract architecture-level parameters
        self.input_encoding = InputEncoding(self.config.input_encoding)
        self.learned_representation_mode = LearnedRepresentationMode(self.config.learned_representation_mode)
        self.task_queries = bool(self.config.task_queries)
        self.backbone_mode = BackboneMode(self.config.backbone.mode)
        self.use_split_backbone = self.backbone_mode == BackboneMode.SPLIT
        self.use_jagged_attention = False
        self.use_flash_attn_varlen = False
        regression_selector_values = self.elemtypes_nonzero
        pt_mode = RegressionMode(self.config.pt_mode)
        eta_mode = RegressionMode(self.config.eta_mode)
        sin_phi_mode = RegressionMode(self.config.sin_phi_mode)
        cos_phi_mode = RegressionMode(self.config.cos_phi_mode)
        energy_mode = RegressionMode(self.config.energy_mode)

        backbone_config = self.config.backbone
        backbone_num_convs = backbone_config.num_convs

        # Extract parameters from the sub-config per model type
        self.num_convs = backbone_num_convs
        activation = Activation(sub_config.activation)
        self.act = get_activation(activation)
        head_dropout_ff = sub_config.dropout_ff
        backbone_dropout_ff = sub_config.dropout_ff
        backbone_dropout_mha = 0.0
        export_onnx_fused = False

        if self.conv_type == ModelType.ATTENTION:
            num_heads = sub_config.num_heads
            head_dim = sub_config.head_dim
            attention_type = sub_config.attention_type
            backbone_dropout_mha = sub_config.dropout_conv_id_mha
            backbone_dropout_ff = sub_config.dropout_conv_id_ff
            self.use_pre_layernorm = sub_config.use_pre_layernorm
            self.use_jagged_attention = sub_config.use_jagged_attention
            self.use_flash_attn_varlen = sub_config.use_flash_attn_varlen
            export_onnx_fused = sub_config.export_onnx_fused
            save_attention = sub_config.save_attention
            if self.use_jagged_attention and export_onnx_fused:
                raise ValueError("Jagged attention is not supported by the ONNX-fused path")
            if self.use_flash_attn_varlen and not self.use_jagged_attention:
                raise ValueError("flash_attn_varlen requires use_jagged_attention=True to pack the backbone input")
            if self.use_flash_attn_varlen and attention_type != AttentionType.FLASH:
                raise ValueError("flash_attn_varlen requires attention_type='flash'")
            if self.use_jagged_attention and attention_type == AttentionType.FLASH and (head_dim < 8 or head_dim > 256 or head_dim % 8 != 0):
                raise ValueError("Jagged flash attention requires head_dim to be a multiple of 8 between 8 and 256")

            embedding_dim = num_heads * head_dim
            width = num_heads * head_dim
        elif self.conv_type == ModelType.GNNLSH:
            embedding_dim = sub_config.embedding_dim
            width = sub_config.width
            self.bin_size = sub_config.bin_size
            max_num_bins = sub_config.max_num_bins
            distance_dim = sub_config.distance_dim
            layernorm = sub_config.layernorm
            num_node_messages = sub_config.num_node_messages
            ffn_dist_hidden_dim = sub_config.ffn_dist_hidden_dim
            ffn_dist_num_layers = sub_config.ffn_dist_num_layers
            kernel_type = sub_config.kernel_type
            use_interbin_attention = sub_config.use_interbin_attention
            num_interbin_heads = sub_config.num_interbin_heads
            num_attention_heads = sub_config.num_attention_heads
            # attention_head_dim = sub_config.attention_head_dim
            distance_dim = sub_config.distance_dim
            num_or_hashes = sub_config.num_or_hashes
            num_and_hashes = sub_config.num_and_hashes
            if kernel_type == KernelType.ATTENTION:
                if distance_dim % num_attention_heads != 0:
                    raise ValueError(f"distance_dim ({distance_dim}) must be divisible by num_attention_heads ({num_attention_heads})")
            self.use_pre_layernorm = False
        elif self.conv_type == ModelType.LITEPT:
            embedding_dim = sub_config.embedding_dim
            width = sub_config.width
            self.use_pre_layernorm = False
        elif self.conv_type == ModelType.HEPT:
            embedding_dim = sub_config.embedding_dim
            width = sub_config.width
            num_heads = sub_config.num_heads
            pos = sub_config.pos
            self.use_pre_layernorm = False
        elif self.conv_type == ModelType.HEPTV2:
            embedding_dim = sub_config.embedding_dim
            width = sub_config.width
            num_heads = sub_config.num_heads
            self.use_pre_layernorm = False

        _logger.info(
            f"MLPF __init__ conv_type={self.conv_type} num_convs={self.num_convs} "
            f"input_encoding={self.input_encoding} backbone_mode={self.backbone_mode} "
            f"raw_input_dim={self.raw_input_dim} engineered_input_dim={self.input_dim}"
        )

        # Input encoding and backbone
        t0 = time.time()
        self.embedding_dim = embedding_dim
        if self.input_encoding == InputEncoding.JOINT:
            _logger.info("Initializing joint input encoding")
            if self.use_split_backbone:
                self._nn0_id = ffn(self.input_dim, embedding_dim, width, self.act, head_dropout_ff)
                self._nn0_reg = ffn(self.input_dim, embedding_dim, width, self.act, head_dropout_ff)
            else:
                self.nn0 = ffn(self.input_dim, embedding_dim, width, self.act, head_dropout_ff)
        elif self.input_encoding == InputEncoding.SPLIT:
            _logger.info("Initializing split input encoding, elemtypes_nonzero={}".format(self.elemtypes_nonzero))
            num_types = len(self.elemtypes_nonzero)
            if self.use_split_backbone:
                self._nn0_id = ffn(self.input_dim, num_types * embedding_dim, width, self.act, head_dropout_ff)
                self._nn0_reg = ffn(self.input_dim, num_types * embedding_dim, width, self.act, head_dropout_ff)
            else:
                self.nn0 = ffn(self.input_dim, num_types * embedding_dim, width, self.act, head_dropout_ff)
        _logger.info("Input encoding initialization took {:.2f}s".format(time.time() - t0))
        if self.use_split_backbone:
            _logger.info("nn0_id parameters: {}".format(count_parameters(self._nn0_id)))
            _logger.info("nn0_reg parameters: {}".format(count_parameters(self._nn0_reg)))
        else:
            _logger.info("nn0 parameters: {}".format(count_parameters(self.nn0)))

        t0 = time.time()
        self.backbone = nn.ModuleList()
        if self.use_split_backbone:
            self._backbone_id = nn.ModuleList()
            self._backbone_reg = nn.ModuleList()
        if self.num_convs != 0:
            backbone_kwargs = {
                "num_layers": self.num_convs,
                "activation": activation,
                "embedding_dim": embedding_dim,
                "width": width,
                "dropout_ff": backbone_dropout_ff,
                "dropout_mha": backbone_dropout_mha,
                "num_heads": num_heads if self.conv_type in [ModelType.ATTENTION, ModelType.HEPT, ModelType.HEPTV2] else None,
                "attention_type": attention_type if self.conv_type == ModelType.ATTENTION else None,
                "export_onnx_fused": export_onnx_fused if self.conv_type == ModelType.ATTENTION else False,
                "use_flash_attn_varlen": self.use_flash_attn_varlen if self.conv_type == ModelType.ATTENTION else False,
                "save_attention": save_attention if self.conv_type == ModelType.ATTENTION else False,
                "pos": pos if self.conv_type == ModelType.HEPT else False,
                "layer_params": {
                    "max_num_bins": max_num_bins if self.conv_type == ModelType.GNNLSH else None,
                    "distance_dim": distance_dim if self.conv_type == ModelType.GNNLSH else None,
                    "layernorm": layernorm if self.conv_type == ModelType.GNNLSH else None,
                    "num_node_messages": num_node_messages if self.conv_type == ModelType.GNNLSH else None,
                    "ffn_dist_hidden_dim": ffn_dist_hidden_dim if self.conv_type == ModelType.GNNLSH else None,
                    "ffn_dist_num_layers": ffn_dist_num_layers if self.conv_type == ModelType.GNNLSH else None,
                    "kernel_type": kernel_type if self.conv_type == ModelType.GNNLSH else None,
                    "use_interbin_attention": use_interbin_attention if self.conv_type == ModelType.GNNLSH else None,
                    "num_interbin_heads": num_interbin_heads if self.conv_type == ModelType.GNNLSH else None,
                    "num_attention_heads": num_attention_heads if self.conv_type == ModelType.GNNLSH else None,
                    "num_or_hashes": num_or_hashes if self.conv_type == ModelType.GNNLSH else None,
                    "num_and_hashes": num_and_hashes if self.conv_type == ModelType.GNNLSH else None,
                },
            }
            if self.use_split_backbone:
                _logger.info("Initializing split id/reg backbone layers, num_convs={}".format(self.num_convs))
                self._backbone_id = self._build_backbone_layers(name_prefix="backbone_id", **backbone_kwargs)
                self._backbone_reg = self._build_backbone_layers(name_prefix="backbone_reg", **backbone_kwargs)
            else:
                _logger.info("Initializing shared backbone layers, num_convs={}".format(self.num_convs))
                self.backbone = self._build_backbone_layers(name_prefix="backbone", **backbone_kwargs)
        _logger.info("Backbone initialization took {:.2f}s".format(time.time() - t0))
        if self.use_split_backbone:
            _logger.info("backbone_id parameters: {}".format(count_parameters(self._backbone_id)))
            _logger.info("backbone_reg parameters: {}".format(count_parameters(self._backbone_reg)))
        else:
            _logger.info("backbone parameters: {}".format(count_parameters(self.backbone)))

        if self.learned_representation_mode == LearnedRepresentationMode.CONCAT:
            decoding_dim = max(self.num_convs, 1) * embedding_dim
        elif self.learned_representation_mode == LearnedRepresentationMode.LAST:
            decoding_dim = embedding_dim

        _logger.info("Initializing output DNNs")
        t0 = time.time()
        self.classification_norm = torch.nn.LayerNorm(decoding_dim) if self.use_pre_layernorm else None
        self.regression_norm = torch.nn.LayerNorm(decoding_dim) if self.use_pre_layernorm else None

        if self.task_queries and not self.use_split_backbone:
            self.classification_query = nn.Parameter(torch.zeros(1, 1, decoding_dim), requires_grad=True)
            self.regression_query = nn.Parameter(torch.zeros(1, 1, decoding_dim), requires_grad=True)
            trunc_normal_(self.classification_query, std=0.02)
            trunc_normal_(self.regression_query, std=0.02)
            readout_heads = infer_num_heads(
                decoding_dim, num_heads if self.conv_type in [ModelType.ATTENTION, ModelType.HEPT, ModelType.HEPTV2] else None
            )
            readout_attention_type = AttentionType.MATH
            readout_export_onnx_fused = bool(export_onnx_fused)
            self.classification_readout = TaskQueryAttentionReadout(
                name="classification_readout",
                activation=activation,
                embedding_dim=decoding_dim,
                num_heads=readout_heads,
                width=width,
                dropout_mha=backbone_dropout_mha,
                dropout_ff=head_dropout_ff,
                attention_type=readout_attention_type,
                export_onnx_fused=readout_export_onnx_fused,
            )
            self.regression_readout = TaskQueryAttentionReadout(
                name="regression_readout",
                activation=activation,
                embedding_dim=decoding_dim,
                num_heads=readout_heads,
                width=width,
                dropout_mha=backbone_dropout_mha,
                dropout_ff=head_dropout_ff,
                attention_type=readout_attention_type,
                export_onnx_fused=readout_export_onnx_fused,
            )
        else:
            self.classification_query = None
            self.regression_query = None
            self.classification_readout = None
            self.regression_readout = None

        self.nn_binary_particle = ffn(decoding_dim, 2, width, self.act, head_dropout_ff)
        self.nn_pid = ffn(decoding_dim, self.num_classes, width, self.act, head_dropout_ff)

        self.nn_pt = RegressionOutput(pt_mode, decoding_dim, width, self.act, head_dropout_ff, regression_selector_values)
        self.nn_eta = RegressionOutput(eta_mode, decoding_dim, width, self.act, head_dropout_ff, regression_selector_values)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, decoding_dim, width, self.act, head_dropout_ff, regression_selector_values)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, decoding_dim, width, self.act, head_dropout_ff, regression_selector_values)
        self.nn_energy = RegressionOutput(energy_mode, decoding_dim, width, self.act, head_dropout_ff, regression_selector_values)
        _logger.info("Output DNNs initialization took {:.2f}s".format(time.time() - t0))

        _logger.info("backbone_mode={}".format(self.backbone_mode))
        _logger.info("jagged_attention enabled={}".format(self.use_jagged_attention))
        _logger.info("flash_attn_varlen enabled={}".format(self.use_flash_attn_varlen))
        _logger.info("task_queries enabled={}".format(self.task_queries and not self.use_split_backbone))
        _logger.info(
            "classification_norm parameters: {}".format(count_parameters(self.classification_norm) if self.classification_norm is not None else 0)
        )
        _logger.info("regression_norm parameters: {}".format(count_parameters(self.regression_norm) if self.regression_norm is not None else 0))
        _logger.info(
            "classification_query parameters: {}".format(count_parameters(self.classification_query) if self.classification_query is not None else 0)
        )
        _logger.info("regression_query parameters: {}".format(count_parameters(self.regression_query) if self.regression_query is not None else 0))
        _logger.info(
            "classification_readout parameters: {}".format(
                count_parameters(self.classification_readout) if self.classification_readout is not None else 0
            )
        )
        _logger.info(
            "regression_readout parameters: {}".format(count_parameters(self.regression_readout) if self.regression_readout is not None else 0)
        )
        _logger.info("nn_binary_particle parameters: {}".format(count_parameters(self.nn_binary_particle)))
        _logger.info("nn_pid parameters: {}".format(count_parameters(self.nn_pid)))
        _logger.info("nn_pt parameters: {}".format(count_parameters(self.nn_pt)))
        _logger.info("nn_eta parameters: {}".format(count_parameters(self.nn_eta)))
        _logger.info("nn_sin_phi parameters: {}".format(count_parameters(self.nn_sin_phi)))
        _logger.info("nn_cos_phi parameters: {}".format(count_parameters(self.nn_cos_phi)))
        _logger.info("nn_energy parameters: {}".format(count_parameters(self.nn_energy)))
        _logger.info("Total MLPF parameters: {}".format(count_parameters(self)))
        _logger.info("MLPF __init__ done")

    def _build_backbone_layers(
        self,
        num_layers,
        activation,
        embedding_dim,
        width,
        dropout_ff,
        name_prefix="backbone",
        dropout_mha=0.0,
        num_heads=None,
        attention_type=None,
        export_onnx_fused=False,
        use_flash_attn_varlen=False,
        save_attention=False,
        pos=False,
        layer_params=None,
    ):
        layers = nn.ModuleList()
        layer_params = layer_params or {}
        for i in range(num_layers):
            _logger.info(f"Initializing backbone layer {i}")
            layers.append(
                self._build_backbone_layer(
                    name=f"{name_prefix}_{i}",
                    is_last=i == num_layers - 1,
                    activation=activation,
                    embedding_dim=embedding_dim,
                    width=width,
                    dropout_ff=dropout_ff,
                    dropout_mha=dropout_mha,
                    num_heads=num_heads,
                    attention_type=attention_type,
                    export_onnx_fused=export_onnx_fused,
                    use_flash_attn_varlen=use_flash_attn_varlen,
                    save_attention=save_attention,
                    pos=pos,
                    layer_params=layer_params,
                )
            )
        return layers

    def _build_backbone_layer(
        self,
        name,
        is_last,
        activation,
        embedding_dim,
        width,
        dropout_ff,
        dropout_mha=0.0,
        num_heads=None,
        attention_type=None,
        export_onnx_fused=False,
        use_flash_attn_varlen=False,
        save_attention=False,
        pos=False,
        layer_params=None,
    ):
        layer_params = layer_params or {}
        if self.conv_type == ModelType.ATTENTION:
            return PreLnSelfAttentionLayer(
                name=name,
                activation=activation,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                width=width,
                dropout_mha=dropout_mha,
                dropout_ff=layer_params.get("dropout_ff", dropout_ff),
                attention_type=attention_type,
                elems_as_queries=is_last,
                export_onnx_fused=export_onnx_fused,
                use_flash_attn_varlen=use_flash_attn_varlen,
                save_attention=save_attention,
            )
        if self.conv_type == ModelType.GNNLSH:
            gnn_conf = {
                "inout_dim": embedding_dim,
                "bin_size": self.bin_size,
                "max_num_bins": layer_params["max_num_bins"],
                "distance_dim": layer_params["distance_dim"],
                "layernorm": layer_params["layernorm"],
                "num_node_messages": layer_params["num_node_messages"],
                "dropout": dropout_ff,
                "ffn_dist_hidden_dim": layer_params["ffn_dist_hidden_dim"],
                "ffn_dist_num_layers": layer_params["ffn_dist_num_layers"],
                "kernel_type": layer_params["kernel_type"],
                "use_interbin_attention": layer_params["use_interbin_attention"],
                "num_interbin_heads": layer_params["num_interbin_heads"],
                "num_attention_heads": layer_params["num_attention_heads"],
                "num_or_hashes": layer_params["num_or_hashes"],
                "num_and_hashes": layer_params["num_and_hashes"],
            }
            return CombinedGraphLayer(**gnn_conf)
        if self.conv_type == ModelType.LITEPT:
            if LitePTLayer is None:
                raise ImportError("LitePTLayer is not available. Please check the LitePT installation.")
            litept_conf = self.config.litept.model_dump()
            return LitePTLayer(name=name, litept_config=litept_conf, embedding_dim=embedding_dim)
        if self.conv_type == ModelType.HEPT:
            hept_conf = self.config.hept.model_dump()
            for key in ["num_convs", "conv_type", "embedding_dim", "width", "activation", "dropout_ff", "num_heads", "pos"]:
                if key in hept_conf:
                    hept_conf.pop(key)
            return HEPTLayer(
                name=name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                width=width,
                dropout=dropout_ff,
                pos=pos,
                **hept_conf,
            )
        if self.conv_type == ModelType.HEPTV2:
            heptv2_conf = self.config.heptv2.model_dump()
            for key in ["num_convs", "conv_type", "embedding_dim", "width", "activation", "dropout_ff", "num_heads"]:
                if key in heptv2_conf:
                    heptv2_conf.pop(key)
            return HEPTv2Layer(
                name=name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                width=width,
                dropout=dropout_ff,
                **heptv2_conf,
            )
        raise ValueError(f"Unsupported conv type {self.conv_type}")

    def _encode_inputs(self, X_features, mask=None, encoder=None):
        encoder = self.nn0 if encoder is None else encoder
        if self.input_encoding == InputEncoding.JOINT:
            encoded = encoder(X_features)
        if self.input_encoding == InputEncoding.SPLIT:
            B, S, _ = X_features.shape
            num_types = len(self.elemtypes_nonzero)
            all_embeddings = encoder(X_features).view(B, S, num_types, -1)
            elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)
            encoded = torch.sum(all_embeddings * elemtype_mask.unsqueeze(-1), axis=2)
        elif self.input_encoding not in [InputEncoding.JOINT, InputEncoding.SPLIT]:
            raise ValueError(f"Unsupported input encoding {self.input_encoding}")
        if mask is not None:
            encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        return encoded

    @property
    def uses_hit_feature_engineering(self):
        return isinstance(self.feature_engineering, HitFeatureEngineering)

    def _engineer_input_features(self, X_features, mask):
        if self.uses_hit_feature_engineering:
            return self.feature_engineering(X_features, mask)
        return X_features

    def _run_backbone(self, x, mask, initial_embedding, X_features, backbone=None):
        backbone = self.backbone if backbone is None else backbone
        embeddings = []
        if len(backbone) == 0:
            embeddings.append(x)
            return embeddings

        for conv in backbone:
            if self.conv_type in [ModelType.LITEPT, ModelType.HEPT, ModelType.HEPTV2]:
                x = conv(x, mask, X_features)
            else:
                x = conv(x, mask, initial_embedding)
            embeddings.append(x)
        return embeddings

    def _prepare_backbone_input(self, x, mask):
        if self.use_jagged_attention:
            return dense_to_jagged(x, mask)
        return x

    def _restore_backbone_output(self, x, mask):
        if self.use_jagged_attention:
            return jagged_to_dense(x, mask)
        return x

    def _collect_representation(self, embeddings):
        if self.learned_representation_mode == LearnedRepresentationMode.CONCAT:
            if self.use_jagged_attention:
                return embeddings[0].with_values(torch.cat([embedding.values for embedding in embeddings], dim=-1))
            return torch.cat(embeddings, axis=-1)
        if self.learned_representation_mode == LearnedRepresentationMode.LAST:
            return embeddings[-1]
        raise ValueError(f"Unsupported learned representation mode {self.learned_representation_mode}")

    def encode_backbone(self, X_features, mask):
        X_features = self._engineer_input_features(X_features, mask)
        if self.use_split_backbone:
            x = self._encode_inputs(X_features, mask=mask, encoder=self._nn0_id)
            x = self._prepare_backbone_input(x, mask)
            embeddings = self._run_backbone(x, mask, x, X_features, backbone=self._backbone_id)
        else:
            x = self._encode_inputs(X_features, mask=mask)
            x = self._prepare_backbone_input(x, mask)
            embeddings = self._run_backbone(x, mask, x, X_features)
        return self._restore_backbone_output(self._collect_representation(embeddings), mask)

    @property
    def nn0_id(self):
        return self._nn0_id if self.use_split_backbone else self.nn0

    @property
    def nn0_reg(self):
        return self._nn0_reg if self.use_split_backbone else self.nn0

    @property
    def conv_id(self):
        return self._backbone_id if self.use_split_backbone else self.backbone

    @property
    def conv_reg(self):
        return self._backbone_reg if self.use_split_backbone else self.backbone

    @property
    def final_norm_id(self):
        return self.classification_norm

    @property
    def final_norm_reg(self):
        return self.regression_norm

    # @torch.compile
    def forward(self, X_features, mask):
        X_features = self._engineer_input_features(X_features, mask)
        if self.use_split_backbone:
            x_id = self._encode_inputs(X_features, mask=mask, encoder=self._nn0_id)
            x_reg = self._encode_inputs(X_features, mask=mask, encoder=self._nn0_reg)
            x_id = self._prepare_backbone_input(x_id, mask)
            x_reg = self._prepare_backbone_input(x_reg, mask)
            embeddings_id = self._run_backbone(x_id, mask, x_id, X_features, backbone=self._backbone_id)
            embeddings_reg = self._run_backbone(x_reg, mask, x_reg, X_features, backbone=self._backbone_reg)
            final_embedding_cls = self._restore_backbone_output(self._collect_representation(embeddings_id), mask)
            final_embedding_reg = self._restore_backbone_output(self._collect_representation(embeddings_reg), mask)
        else:
            x = self._encode_inputs(X_features, mask=mask)
            x = self._prepare_backbone_input(x, mask)
            backbone_embeddings = self._run_backbone(x, mask, x, X_features)
            final_embedding = self._restore_backbone_output(self._collect_representation(backbone_embeddings), mask)

            if self.task_queries:
                final_embedding_cls = self.classification_readout(final_embedding, mask, self.classification_query)
                final_embedding_reg = self.regression_readout(final_embedding, mask, self.regression_query)
            else:
                final_embedding_cls = final_embedding
                final_embedding_reg = final_embedding

        if self.classification_norm is not None:
            final_embedding_cls = self.classification_norm(final_embedding_cls)
        if self.regression_norm is not None:
            final_embedding_reg = self.regression_norm(final_embedding_reg)

        preds_binary_particle = self.nn_binary_particle(final_embedding_cls)
        preds_pid = self.nn_pid(final_embedding_cls)
        preds_pu = torch.zeros_like(preds_binary_particle)

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(X_features, final_embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(X_features, final_embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X_features, final_embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X_features, final_embedding_reg, X_features[..., 4:5])

        # ensure created particle has positive mass^2 by computing energy from pt and adding a positive-only correction
        pt_real = torch.exp(preds_pt.detach()) * X_features[..., 1:2]
        pz_real = pt_real * (torch.exp(preds_eta.detach()) - torch.exp(-preds_eta.detach())) / 2.0
        e_real = torch.log(torch.sqrt(pt_real**2 + pz_real**2) / X_features[..., 5:6])
        if mask is not None:
            e_real = e_real * mask.unsqueeze(-1)
        e_real = torch.nan_to_num(e_real, nan=0.0, posinf=0.0, neginf=0.0)
        preds_energy = e_real + torch.nn.functional.relu(self.nn_energy(X_features, final_embedding_reg, X_features[..., 5:6]))
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)

        preds_binary_particle = torch.nan_to_num(preds_binary_particle, nan=0.0, posinf=0.0, neginf=0.0)
        preds_pid = torch.nan_to_num(preds_pid, nan=0.0, posinf=0.0, neginf=0.0)
        preds_momentum = torch.nan_to_num(preds_momentum, nan=0.0, posinf=0.0, neginf=0.0)

        return preds_binary_particle, preds_pid, preds_momentum, preds_pu

    def predict_particles(self, X_features, mask):
        from mlpf.model.utils import unpack_predictions

        ypred_raw = self.forward(X_features, mask)
        ypred_raw = tuple([y.to(torch.float32) for y in ypred_raw])

        # transform log (pt/elempt) -> pt
        ypred_raw[2][..., 0] = torch.exp(ypred_raw[2][..., 0]) * X_features[..., 1]
        # transform log (E/elemE) -> E
        ypred_raw[2][..., 4] = torch.exp(ypred_raw[2][..., 4]) * X_features[..., 5]

        ypred = unpack_predictions(ypred_raw)
        ypred["ispu"] = torch.softmax(ypred["ispu"], axis=-1)[:, :, -1]

        # By default, use standard argmax
        pred_cls = torch.argmax(ypred_raw[0], axis=-1)

        # Zero out predictions for non-particles
        ypred["cls_id"][pred_cls == 0] = 0
        ypred["pt"][pred_cls == 0] = 0
        ypred["energy"][pred_cls == 0] = 0
        ypred["eta"][pred_cls == 0] = 0
        ypred["sin_phi"][pred_cls == 0] = 0
        ypred["cos_phi"][pred_cls == 0] = 0

        return ypred


def configure_model_trainable(model: MLPF, trainable: Union[str, List[str]], is_training: bool):
    """Set only the given layers as trainable in the model"""

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise Exception("configure trainability before distributing the model")
    if is_training:
        model.train()
        if trainable != "all":
            model.eval()

            # first set all parameters as non-trainable
            for param in model.parameters():
                param.requires_grad = False

            # now explicitly enable specific layers
            for layer in trainable:
                layer = getattr(model, layer)
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        model.eval()
