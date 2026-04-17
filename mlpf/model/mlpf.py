import time
from typing import Union, List

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

from mlpf.conf import (
    MLPFConfig,
    Activation,
    AttentionType,
    ModelType,
    InputEncoding,
    LearnedRepresentationMode,
    RegressionMode,
)


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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        bias = True
        batch_first = True
        super().__init__(embed_dim, num_heads, dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.head_dim = int(embed_dim // num_heads)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        self.export_onnx_fused = export_onnx_fused
        self.attention_type = AttentionType(attention_type)
        self.attn_params = {
            AttentionType.SIMPLE: [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
            AttentionType.MATH: [SDPBackend.MATH],
            AttentionType.FLASH: [SDPBackend.FLASH_ATTENTION],
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, need_weights=False, key_padding_mask=None) -> torch.Tensor:
        # q, k, v: 3D tensors (batch_size, seq_len, embed_dim), embed_dim = num_heads*head_dim
        bs, seq_len, embed_dim = q.size()
        head_dim = self.head_dim
        num_heads = self.num_heads

        # split stacked in_proj_weight, in_proj_bias to q, k, v matrices
        wq, wk, wv = torch.split(self.in_proj_weight, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)
        bq, bk, bv = torch.split(self.in_proj_bias, [self.embed_dim, self.embed_dim, self.embed_dim], dim=0)

        q = torch.matmul(q, wq.T) + bq
        k = torch.matmul(k, wk.T) + bk
        v = torch.matmul(v, wv.T) + bv

        # for pytorch internal scaled dot product attention, we need (bs, num_heads, seq_len, head_dim)
        if not self.export_onnx_fused:
            q = q.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(bs, seq_len, num_heads, head_dim).transpose(1, 2)

        # this function will have different shape signatures in native pytorch sdpa and in ONNX com.microsoft.MultiHeadAttention
        # in pytorch: (bs, num_heads, seq_len, head_dim)
        # in ONNX: (bs, seq_len, num_heads*head_dim)
        if self.export_onnx_fused:
            attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        else:
            with sdpa_kernel(self.attn_params[self.attention_type]):
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)

        # in case running with pytorch internal scaled dot product attention, reshape back to the original shape
        if not self.export_onnx_fused:
            attn_output = attn_output.transpose(1, 2).reshape(bs, seq_len, num_heads * head_dim)

        # assert list(attn_output.size()) == [bs, seq_len, num_heads * head_dim]
        attn_output = self.out_proj(attn_output)
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
        self.input_norm = x.norm().detach()
        self.seq_len = x.shape[1]

        if mask is not None:
            mask_ = mask.unsqueeze(-1)

        residual = x
        x_norm = self.norm0(x)

        q = x_norm

        if self.learnable_queries:
            q = self.queries.expand(*x.shape)
        elif self.elems_as_queries:
            q = initial_embedding
        if mask is not None:
            q = q * mask_

        mha_out = self.mha(q, x_norm, x_norm, need_weights=False)[0]

        self.mha_res_norm = mha_out.norm().detach()

        x = residual + mha_out
        residual = x
        x_norm = self.norm1(x)
        ffn_out = self.seq(x_norm)
        ffn_out = self.dropout(ffn_out)

        self.ffn_res_norm = ffn_out.norm().detach()

        x = residual + ffn_out
        if mask is not None:
            x = x * mask_
        return x


def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        act(),
        torch.nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


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
        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.elemtypes_nonzero = config.elemtypes_nonzero

        # Determine architecture parameters based on the chosen type
        self.conv_type = ModelType(self.config.type)
        sub_config = getattr(self.config, self.conv_type.value)

        # Ensure sub_config is initialized if it's None (can happen if not in spec or args)
        if sub_config is None:
            from mlpf.conf import AttentionConfig, GNNLSHConfig, LitePTConfig, HEPTConfig

            if self.conv_type == ModelType.ATTENTION:
                sub_config = AttentionConfig()
            elif self.conv_type == ModelType.GNNLSH:
                sub_config = GNNLSHConfig()
            elif self.conv_type == ModelType.LITEPT:
                sub_config = LitePTConfig()
            elif self.conv_type == ModelType.HEPT:
                sub_config = HEPTConfig()
            setattr(self.config, self.conv_type.value, sub_config)

        # Extract architecture-level parameters
        self.input_encoding = InputEncoding(self.config.input_encoding)
        self.learned_representation_mode = LearnedRepresentationMode(self.config.learned_representation_mode)
        pt_mode = RegressionMode(self.config.pt_mode)
        eta_mode = RegressionMode(self.config.eta_mode)
        sin_phi_mode = RegressionMode(self.config.sin_phi_mode)
        cos_phi_mode = RegressionMode(self.config.cos_phi_mode)
        energy_mode = RegressionMode(self.config.energy_mode)

        # Extract parameters from the sub-config per model type
        self.num_convs = sub_config.num_convs
        activation = Activation(sub_config.activation)
        self.act = get_activation(activation)
        dropout_ff = sub_config.dropout_ff

        if self.conv_type == ModelType.ATTENTION:
            num_heads = sub_config.num_heads
            head_dim = sub_config.head_dim
            attention_type = sub_config.attention_type
            dropout_conv_reg_mha = sub_config.dropout_conv_reg_mha
            dropout_conv_reg_ff = sub_config.dropout_conv_reg_ff
            dropout_conv_id_mha = sub_config.dropout_conv_id_mha
            dropout_conv_id_ff = sub_config.dropout_conv_id_ff
            self.use_pre_layernorm = sub_config.use_pre_layernorm
            export_onnx_fused = sub_config.export_onnx_fused
            save_attention = sub_config.save_attention

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

        _logger.info(f"MLPF __init__ conv_type={self.conv_type} num_convs={self.num_convs} input_encoding={self.input_encoding}")

        # embedding of the inputs
        t0 = time.time()
        if self.input_encoding == InputEncoding.JOINT:
            _logger.info("Initializing joint input encoding")
            self.nn0_id = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)
            self.nn0_reg = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)
        elif self.input_encoding == InputEncoding.SPLIT:
            _logger.info("Initializing split input encoding, elemtypes_nonzero={}".format(self.elemtypes_nonzero))
            # Wide MLP approach: one large Linear layer to produce all embeddings at once
            num_types = len(self.elemtypes_nonzero)
            self.nn0_id = ffn(self.input_dim, num_types * embedding_dim, width, self.act, dropout_ff)
            self.nn0_reg = ffn(self.input_dim, num_types * embedding_dim, width, self.act, dropout_ff)
        _logger.info("Input encoding initialization took {:.2f}s".format(time.time() - t0))
        _logger.info("nn0_id parameters: {}".format(count_parameters(self.nn0_id)))
        _logger.info("nn0_reg parameters: {}".format(count_parameters(self.nn0_reg)))

        if self.num_convs != 0:
            t0 = time.time()
            if self.conv_type == ModelType.ATTENTION:
                _logger.info("Initializing attention convolution layers, num_convs={}".format(self.num_convs))
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()

                for i in range(self.num_convs):
                    _logger.info(f"Initializing attention layer {i}")
                    lastlayer = i == self.num_convs - 1

                    self.conv_id.append(
                        PreLnSelfAttentionLayer(
                            name="conv_id_{}".format(i),
                            activation=activation,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            width=width,
                            dropout_mha=dropout_conv_id_mha,
                            dropout_ff=dropout_conv_id_ff,
                            attention_type=attention_type,
                            elems_as_queries=lastlayer,
                            # learnable_queries=lastlayer,
                            export_onnx_fused=export_onnx_fused,
                            save_attention=save_attention,
                        )
                    )
                    self.conv_reg.append(
                        PreLnSelfAttentionLayer(
                            name="conv_reg_{}".format(i),
                            activation=activation,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            width=width,
                            dropout_mha=dropout_conv_reg_mha,
                            dropout_ff=dropout_conv_reg_ff,
                            attention_type=attention_type,
                            elems_as_queries=lastlayer,
                            # learnable_queries=lastlayer,
                            export_onnx_fused=export_onnx_fused,
                            save_attention=save_attention,
                        )
                    )

            elif self.conv_type == ModelType.GNNLSH:
                _logger.info("Initializing GNNLSH convolution layers")
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(self.num_convs):
                    _logger.info(f"Initializing GNN-LSH layer {i}")
                    gnn_conf = {
                        "inout_dim": embedding_dim,
                        "bin_size": self.bin_size,
                        "max_num_bins": max_num_bins,
                        "distance_dim": distance_dim,
                        "layernorm": layernorm,
                        "num_node_messages": num_node_messages,
                        "dropout": dropout_ff,
                        "ffn_dist_hidden_dim": ffn_dist_hidden_dim,
                        "ffn_dist_num_layers": ffn_dist_num_layers,
                    }
                    self.conv_id.append(CombinedGraphLayer(**gnn_conf))
                    self.conv_reg.append(CombinedGraphLayer(**gnn_conf))
            elif self.conv_type == ModelType.LITEPT:
                if LitePTLayer is None:
                    raise ImportError("LitePTLayer is not available. Please check the LitePT installation.")
                _logger.info("Initializing LitePT convolution layers")
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                litept_conf = self.config.litept.model_dump()
                for i in range(self.num_convs):
                    _logger.info(f"Initializing LitePT layer {i}")
                    self.conv_id.append(LitePTLayer(name=f"litept_id_{i}", litept_config=litept_conf, embedding_dim=embedding_dim))
                    self.conv_reg.append(LitePTLayer(name=f"litept_reg_{i}", litept_config=litept_conf, embedding_dim=embedding_dim))
            elif self.conv_type == ModelType.HEPT:
                _logger.info("Initializing HEPT convolution layers")
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                hept_conf = self.config.hept.model_dump()
                # Remove keys that HEPTLayer constructor handles via explicit arguments
                for key in ["num_convs", "conv_type", "embedding_dim", "width", "activation", "dropout_ff", "num_heads", "pos"]:
                    if key in hept_conf:
                        hept_conf.pop(key)
                for i in range(self.num_convs):
                    _logger.info(f"Initializing HEPT layer {i}")
                    self.conv_id.append(
                        HEPTLayer(
                            name=f"hept_id_{i}",
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            width=width,
                            dropout=dropout_ff,
                            pos=pos,
                            **hept_conf,
                        )
                    )
                    self.conv_reg.append(
                        HEPTLayer(
                            name=f"hept_reg_{i}",
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            width=width,
                            dropout=dropout_ff,
                            pos=pos,
                            **hept_conf,
                        )
                    )
            _logger.info("Convolution layers initialization took {:.2f}s".format(time.time() - t0))
            _logger.info("conv_id parameters: {}".format(count_parameters(self.conv_id)))
            _logger.info("conv_reg parameters: {}".format(count_parameters(self.conv_reg)))

        if self.learned_representation_mode == LearnedRepresentationMode.CONCAT:
            decoding_dim = self.num_convs * embedding_dim
        elif self.learned_representation_mode == LearnedRepresentationMode.LAST:
            decoding_dim = embedding_dim

        _logger.info("Initializing output DNNs")
        t0 = time.time()
        # DNN that acts on the node level to predict the PID
        self.nn_binary_particle = ffn(decoding_dim, 2, width, self.act, dropout_ff)
        self.nn_pid = ffn(decoding_dim, self.num_classes, width, self.act, dropout_ff)
        # self.nn_pu = ffn(decoding_dim, 2, width, self.act, dropout_ff)

        # elementwise DNN for node momentum regression
        embed_dim = decoding_dim
        self.nn_pt = RegressionOutput(pt_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_eta = RegressionOutput(eta_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_energy = RegressionOutput(energy_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        _logger.info("Output DNNs initialization took {:.2f}s".format(time.time() - t0))

        _logger.info("nn_binary_particle parameters: {}".format(count_parameters(self.nn_binary_particle)))
        _logger.info("nn_pid parameters: {}".format(count_parameters(self.nn_pid)))
        _logger.info("nn_pt parameters: {}".format(count_parameters(self.nn_pt)))
        _logger.info("nn_eta parameters: {}".format(count_parameters(self.nn_eta)))
        _logger.info("nn_sin_phi parameters: {}".format(count_parameters(self.nn_sin_phi)))
        _logger.info("nn_cos_phi parameters: {}".format(count_parameters(self.nn_cos_phi)))
        _logger.info("nn_energy parameters: {}".format(count_parameters(self.nn_energy)))

        if self.use_pre_layernorm:  # add final norm after last attention block as per https://arxiv.org/abs/2002.04745
            _logger.info("Initializing final normalization layers")
            self.final_norm_id = torch.nn.LayerNorm(decoding_dim)
            self.final_norm_reg = torch.nn.LayerNorm(embed_dim)
            _logger.info("final_norm_id parameters: {}".format(count_parameters(self.final_norm_id)))
            _logger.info("final_norm_reg parameters: {}".format(count_parameters(self.final_norm_reg)))
        _logger.info("Total MLPF parameters: {}".format(count_parameters(self)))
        _logger.info("MLPF __init__ done")

    # @torch.compile
    def forward(self, X_features, mask):
        Xfeat_normed = X_features

        embeddings_id, embeddings_reg = [], []
        if self.input_encoding == InputEncoding.JOINT:
            embedding_id = self.nn0_id(Xfeat_normed)
            embedding_reg = self.nn0_reg(Xfeat_normed)
        elif self.input_encoding == InputEncoding.SPLIT:
            # embedding_id = torch.stack([nn0(Xfeat_normed) for nn0 in self.nn0_id], axis=-1)
            # elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)
            # embedding_id = torch.sum(embedding_id * elemtype_mask.unsqueeze(-2), axis=-1)

            # embedding_reg = torch.stack([nn0(Xfeat_normed) for nn0 in self.nn0_reg], axis=-1)
            # elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)
            # embedding_reg = torch.sum(embedding_reg * elemtype_mask.unsqueeze(-2), axis=-1)

            B, S, _ = Xfeat_normed.shape
            num_types = len(self.elemtypes_nonzero)

            # Wide MLP approach: compute all at once and reshape to [B, S, num_types, embedding_dim]
            all_id = self.nn0_id(Xfeat_normed).view(B, S, num_types, -1)
            all_reg = self.nn0_reg(Xfeat_normed).view(B, S, num_types, -1)

            # Create mask [B, S, num_types]
            elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)

            # Select relevant embedding: [B, S, num_types, D] * [B, S, num_types, 1] -> [B, S, num_types, D] -> sum over num_types -> [B, S, D]
            embedding_id = torch.sum(all_id * elemtype_mask.unsqueeze(-1), axis=2)
            embedding_reg = torch.sum(all_reg * elemtype_mask.unsqueeze(-1), axis=2)
        if self.num_convs != 0:
            for num, conv in enumerate(self.conv_id):
                conv_input = embedding_id if num == 0 else embeddings_id[-1]
                if self.conv_type == ModelType.LITEPT or self.conv_type == ModelType.HEPT:
                    out_padded = conv(conv_input, mask, X_features)
                else:
                    out_padded = conv(conv_input, mask, embedding_id)
                embeddings_id.append(out_padded)

            for num, conv in enumerate(self.conv_reg):
                conv_input = embedding_reg if num == 0 else embeddings_reg[-1]
                if self.conv_type == ModelType.LITEPT or self.conv_type == ModelType.HEPT:
                    out_padded = conv(conv_input, mask, X_features)
                else:
                    out_padded = conv(conv_input, mask, embedding_reg)
                embeddings_reg.append(out_padded)
        else:
            embeddings_id.append(embedding_id)
            embeddings_reg.append(embedding_reg)

        # id input
        if self.learned_representation_mode == LearnedRepresentationMode.CONCAT:
            final_embedding_id = torch.cat(embeddings_id, axis=-1)
        elif self.learned_representation_mode == LearnedRepresentationMode.LAST:
            final_embedding_id = torch.cat([embeddings_id[-1]], axis=-1)

        if self.use_pre_layernorm:
            final_embedding_id = self.final_norm_id(final_embedding_id)

        preds_binary_particle = self.nn_binary_particle(final_embedding_id)
        preds_pid = self.nn_pid(final_embedding_id)
        # preds_pu = self.nn_pu(final_embedding_id)
        preds_pu = torch.zeros_like(preds_binary_particle)

        # pred_charge = self.nn_charge(final_embedding_id)

        # regression input
        if self.learned_representation_mode == LearnedRepresentationMode.CONCAT:
            final_embedding_reg = torch.cat(embeddings_reg, axis=-1)
        elif self.learned_representation_mode == LearnedRepresentationMode.LAST:
            final_embedding_reg = torch.cat([embeddings_reg[-1]], axis=-1)

        if self.use_pre_layernorm:
            final_embedding_reg = self.final_norm_reg(final_embedding_reg)

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(X_features, final_embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(X_features, final_embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X_features, final_embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X_features, final_embedding_reg, X_features[..., 4:5])

        # ensure created particle has positive mass^2 by computing energy from pt and adding a positive-only correction
        pt_real = torch.exp(preds_pt.detach()) * X_features[..., 1:2]
        # sinh does not exist on opset13, required for CMSSW_12_3_0_pre6
        # pz_real = pt_real * torch.sinh(preds_eta.detach())
        pz_real = pt_real * (torch.exp(preds_eta.detach()) - torch.exp(-preds_eta.detach())) / 2.0
        e_real = torch.log(torch.sqrt(pt_real**2 + pz_real**2) / X_features[..., 5:6])
        if mask is not None:
            e_real = e_real * mask.unsqueeze(-1)
        e_real[torch.isinf(e_real)] = 0
        e_real[torch.isnan(e_real)] = 0
        preds_energy = e_real + torch.nn.functional.relu(self.nn_energy(X_features, final_embedding_reg, X_features[..., 5:6]))
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)

        # Guard against nan/inf to prevent segfaults in downstream libraries like fastjet
        preds_binary_particle = torch.nan_to_num(preds_binary_particle, nan=0.0, posinf=0.0, neginf=0.0)
        preds_pid = torch.nan_to_num(preds_pid, nan=0.0, posinf=0.0, neginf=0.0)
        preds_momentum = torch.nan_to_num(preds_momentum, nan=0.0, posinf=0.0, neginf=0.0)

        return preds_binary_particle, preds_pid, preds_momentum, preds_pu


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
