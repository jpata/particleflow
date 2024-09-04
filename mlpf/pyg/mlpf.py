import torch
import torch.nn as nn

from .gnn_lsh import CombinedGraphLayer

from torch.nn.attention import SDPBackend, sdpa_kernel
from pyg.logger import _logger


def get_activation(activation):
    if activation == "elu":
        act = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "relu6":
        act = nn.ReLU6
    elif activation == "leakyrelu":
        act = nn.LeakyReLU
    elif activation == "gelu":
        act = nn.GELU
    return act


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        activation="elu",
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
        attention_type="efficient",
    ):
        super(SelfAttentionLayer, self).__init__()

        # to enable manual override for ONNX export
        self.enable_ctx_manager = True

        self.attention_type = attention_type
        self.act = get_activation(activation)
        if self.attention_type == "flash_external":
            from flash_attn.modules.mha import MHA

            self.mha = MHA(embedding_dim, num_heads, dropout=dropout_mha)
        else:
            self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_mha, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act())
        self.dropout = torch.nn.Dropout(dropout_ff)
        _logger.info("using attention_type={}".format(attention_type))
        # params for torch sdp_kernel
        self.attn_params = {
            "math": [SDPBackend.MATH],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "flash": [SDPBackend.FLASH_ATTENTION],
        }

    def forward(self, x, mask):
        # explicitly call the desired attention mechanism
        if self.attention_type == "flash_external":
            mha_out = self.mha(x)
        else:
            if self.enable_ctx_manager:
                with sdpa_kernel(self.attn_params[self.attention_type]):
                    mha_out = self.mha(x, x, x, need_weights=False)[0]
            else:
                mha_out = self.mha(x, x, x, need_weights=False)[0]

        x = x + mha_out
        x = self.norm0(x)
        x = x + self.seq(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = x * mask.unsqueeze(-1)
        return x


class PreLnSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        activation="elu",
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
        attention_type="efficient",
    ):
        super(PreLnSelfAttentionLayer, self).__init__()

        # to enable manual override for ONNX export
        self.enable_ctx_manager = True

        self.attention_type = attention_type
        self.act = get_activation(activation)
        if self.attention_type == "flash_external":
            from flash_attn.modules.mha import MHA

            self.mha = MHA(embedding_dim, num_heads, dropout=dropout_mha)
        else:
            self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_mha, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act())
        self.dropout = torch.nn.Dropout(dropout_ff)
        _logger.info("using attention_type={}".format(attention_type))
        # params for torch sdp_kernel
        self.attn_params = {
            "math": [SDPBackend.MATH],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "flash": [SDPBackend.FLASH_ATTENTION],
        }

    def forward(self, x, mask):
        x = self.norm0(x)

        # explicitly call the desired attention mechanism
        if self.attention_type == "flash_external":
            mha_out = self.mha(x)
        else:
            if self.enable_ctx_manager:
                with sdpa_kernel(self.attn_params[self.attention_type]):
                    mha_out = self.mha(x, x, x, need_weights=False)[0]
            else:
                mha_out = self.mha(x, x, x, need_weights=False)[0]

        mha_out = x + mha_out
        x = self.norm1(mha_out)
        x = mha_out + self.seq(x)
        x = self.dropout(x)
        x = x * mask.unsqueeze(-1)
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
    def __init__(self, mode, embed_dim, width, act, dropout, elemtypes):
        super(RegressionOutput, self).__init__()
        self.mode = mode
        self.elemtypes = elemtypes

        # single output
        if self.mode == "direct" or self.mode == "additive" or self.mode == "multiplicative":
            self.nn = ffn(embed_dim, 1, width, act, dropout)
        # two outputs
        elif self.mode == "linear":
            self.nn = ffn(embed_dim, 2, width, act, dropout)
        elif self.mode == "linear-elemtype":
            self.nn1 = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
            self.nn2 = ffn(embed_dim, len(self.elemtypes), width, act, dropout)

    def forward(self, elems, x, orig_value):

        if self.mode == "direct":
            nn_out = self.nn(x)
            return nn_out
        elif self.mode == "additive":
            nn_out = self.nn(x)
            return orig_value + nn_out
        elif self.mode == "multiplicative":
            nn_out = self.nn(x)
            return orig_value * nn_out
        elif self.mode == "linear":
            nn_out = self.nn(x)
            return orig_value * nn_out[..., 0:1] + nn_out[..., 1:2]
        elif self.mode == "linear-elemtype":
            nn_out1 = self.nn1(x)
            nn_out2 = self.nn2(x)
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            a = torch.sum(elemtype_mask * nn_out1, axis=-1, keepdims=True)
            b = torch.sum(elemtype_mask * nn_out2, axis=-1, keepdims=True)
            return orig_value * a + b


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        num_classes=8,
        embedding_dim=128,
        width=128,
        num_convs_common=2,
        num_convs_id=2,
        num_convs_reg=2,
        dropout_ff=0.0,
        activation="elu",
        layernorm=True,
        conv_type="attention",
        input_encoding="joint",
        pt_mode="additive-elemtype",
        eta_mode="additive-elemtype",
        sin_phi_mode="additive-elemtype",
        cos_phi_mode="additive-elemtype",
        energy_mode="additive-elemtype",
        # element types which actually exist in the dataset
        elemtypes_nonzero=[1, 4, 5, 6, 8, 9, 10, 11],
        # gnn-lsh specific parameters
        bin_size=640,
        max_num_bins=200,
        distance_dim=128,
        num_node_messages=2,
        ffn_dist_hidden_dim=128,
        ffn_dist_num_layers=2,
        # self-attention specific parameters
        num_heads=16,
        head_dim=16,
        attention_type="flash",
        dropout_conv_common_mha=0.0,
        dropout_conv_common_ff=0.0,
        dropout_conv_reg_mha=0.0,
        dropout_conv_reg_ff=0.0,
        dropout_conv_id_mha=0.0,
        dropout_conv_id_ff=0.0,
        use_pre_layernorm=False,
    ):
        super(MLPF, self).__init__()

        self.conv_type = conv_type

        self.act = get_activation(activation)

        self.input_encoding = input_encoding

        self.input_dim = input_dim

        self.num_convs_common = num_convs_common
        self.num_convs_id = num_convs_id
        self.num_convs_reg = num_convs_reg

        self.bin_size = bin_size
        self.elemtypes_nonzero = elemtypes_nonzero

        self.use_pre_layernorm = use_pre_layernorm

        if self.conv_type == "attention":
            embedding_dim = num_heads * head_dim
            width = num_heads * head_dim

        # embedding of the inputs
        self.ffn_embedding = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)

        # transformer or GNN layers
        self.conv_common = nn.ModuleList()
        self.conv_id = nn.ModuleList()
        self.conv_reg = nn.ModuleList()

        if self.conv_type == "attention":
            attention_layer = PreLnSelfAttentionLayer if self.use_pre_layernorm else SelfAttentionLayer
            for i in range(self.num_convs_common):
                self.conv_common.append(
                    attention_layer(
                        activation=activation,
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        width=width,
                        dropout_mha=dropout_conv_common_mha,
                        dropout_ff=dropout_conv_common_ff,
                        attention_type=attention_type,
                    )
                )

            for i in range(self.num_convs_id):
                self.conv_id.append(
                    attention_layer(
                        activation=activation,
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        width=width,
                        dropout_mha=dropout_conv_id_mha,
                        dropout_ff=dropout_conv_id_ff,
                        attention_type=attention_type,
                    )
                )
            for i in range(self.num_convs_reg):
                self.conv_reg.append(
                    attention_layer(
                        activation=activation,
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        width=width,
                        dropout_mha=dropout_conv_reg_mha,
                        dropout_ff=dropout_conv_reg_ff,
                        attention_type=attention_type,
                    )
                )
        elif self.conv_type == "gnn_lsh":
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
            for i in range(self.num_convs_common):
                self.conv_common.append(CombinedGraphLayer(**gnn_conf))

            for i in range(self.num_convs_id):
                self.conv_id.append(CombinedGraphLayer(**gnn_conf))

            for i in range(self.num_convs_reg):
                self.conv_reg.append(CombinedGraphLayer(**gnn_conf))

        decoding_dim = self.input_dim + embedding_dim

        # DNN that acts on the node level to predict the PID
        self.nn_binary_particle = ffn(decoding_dim, 2, width, self.act, dropout_ff)
        self.nn_pid = ffn(decoding_dim, num_classes, width, self.act, dropout_ff)

        # elementwise DNN for node momentum regression
        embed_dim = decoding_dim + 2 + num_classes
        self.nn_pt = RegressionOutput(pt_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_eta = RegressionOutput(eta_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)
        self.nn_energy = RegressionOutput(energy_mode, embed_dim, width, self.act, dropout_ff, self.elemtypes_nonzero)

        if self.use_pre_layernorm:  # add final norm after last attention block as per https://arxiv.org/abs/2002.04745
            self.final_norm_id = torch.nn.LayerNorm(decoding_dim)
            self.final_norm_reg = torch.nn.LayerNorm(embed_dim)

    # @torch.compile
    def forward(self, X_features, mask):
        Xfeat_normed = X_features

        embedding = self.ffn_embedding(Xfeat_normed)
        embeddings = []
        for num, conv in enumerate(self.conv_common):
            conv_input = embedding if num == 0 else embeddings[-1]
            out_padded = conv(conv_input, mask)
            embeddings.append(out_padded)

        embeddings_id = []
        for num, conv in enumerate(self.conv_id):
            conv_input = embeddings[-1] if num == 0 else embeddings_id[-1]
            out_padded = conv(conv_input, mask)
            embeddings_id.append(out_padded)
        embeddings_reg = []
        for num, conv in enumerate(self.conv_reg):
            conv_input = embeddings[-1] if num == 0 else embeddings_reg[-1]
            out_padded = conv(conv_input, mask)
            embeddings_reg.append(out_padded)

        # ID
        final_embedding_id = torch.cat([Xfeat_normed] + [embeddings_id[-1]], axis=-1)
        if self.use_pre_layernorm:
            final_embedding_id = self.final_norm_id(final_embedding_id)
        preds_binary_particle = self.nn_binary_particle(final_embedding_id)
        preds_pid = self.nn_pid(final_embedding_id)

        # regression
        final_embedding_reg = torch.cat([Xfeat_normed] + [embeddings_reg[-1]] + [preds_binary_particle.detach(), preds_pid.detach()], axis=-1)

        if self.use_pre_layernorm:
            final_embedding_reg = self.final_norm_reg(final_embedding_reg)

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(X_features, final_embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(X_features, final_embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X_features, final_embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X_features, final_embedding_reg, X_features[..., 4:5])
        preds_energy = self.nn_energy(X_features, final_embedding_reg, X_features[..., 5:6])
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)

        return preds_binary_particle, preds_pid, preds_momentum
