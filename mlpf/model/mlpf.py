import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from mlpf.model.logger import _logger
from mlpf.model.gnn_lsh import CombinedGraphLayer


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # From https://github.com/rwightman/pytorch-image-models/blob/
    #        18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lo = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2lo-1, 2up-1].
        tensor.uniform_(2 * lo - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


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


class PreLnSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        name="",
        activation="elu",
        embedding_dim=128,
        num_heads=2,
        width=128,
        dropout_mha=0.1,
        dropout_ff=0.1,
        attention_type="efficient",
        learnable_queries=False,
        elems_as_queries=False,
    ):
        super(PreLnSelfAttentionLayer, self).__init__()
        self.name = name

        # set to False to enable manual override for ONNX export
        self.enable_ctx_manager = True

        self.attention_type = attention_type
        self.act = get_activation(activation)
        self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_mha, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act())
        self.dropout = torch.nn.Dropout(dropout_ff)
        _logger.info("using attention_type={}".format(attention_type))
        # params for torch sdp_kernel
        if self.enable_ctx_manager:
            self.attn_params = {
                "math": [SDPBackend.MATH],
                "efficient": [SDPBackend.EFFICIENT_ATTENTION],
                "flash": [SDPBackend.FLASH_ATTENTION],
            }

        self.learnable_queries = learnable_queries
        self.elems_as_queries = elems_as_queries
        if self.learnable_queries:
            self.queries = nn.Parameter(torch.zeros(1, 1, embedding_dim), requires_grad=True)
            trunc_normal_(self.queries, std=0.02)

        self.save_attention = False
        self.outdir = ""

    def forward(self, x, mask, initial_embedding):
        mask_ = mask.unsqueeze(-1)
        x = self.norm0(x * mask_)

        q = x
        if self.learnable_queries:
            q = self.queries.expand(*x.shape) * mask_
        elif self.elems_as_queries:
            q = initial_embedding * mask_

        key_padding_mask = None
        if self.attention_type == "math":
            key_padding_mask = ~mask

        # default path, for FlashAttn/Math backend
        if self.enable_ctx_manager:
            with sdpa_kernel(self.attn_params[self.attention_type]):
                mha_out = self.mha(q, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

                if self.save_attention:
                    att_mat = self.mha(q, x, x, need_weights=True, key_padding_mask=key_padding_mask)[1]
                    att_mat = att_mat.detach().cpu().numpy()
                    np.savez(
                        open("{}/attn_{}.npz".format(self.outdir, self.name), "wb"),
                        att=att_mat,
                        in_proj_weight=self.mha.in_proj_weight.detach().cpu().numpy(),
                    )

        # path for ONNX export
        else:
            mha_out = self.mha(q, x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

        mha_out = mha_out * mask_

        mha_out = x + mha_out
        x = self.norm1(mha_out)
        x = mha_out + self.seq(x)
        x = self.dropout(x)
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
    def __init__(self, mode, embed_dim, width, act, dropout, elemtypes):
        super(RegressionOutput, self).__init__()
        self.mode = mode
        self.elemtypes = elemtypes

        # single output
        if self.mode == "direct" or self.mode == "additive" or self.mode == "multiplicative":
            self.nn = ffn(embed_dim, 1, width, act, dropout)
        elif self.mode == "direct-elemtype":
            self.nn = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
        elif self.mode == "direct-elemtype-split":
            self.nn = nn.ModuleList()
            for elem in range(len(self.elemtypes)):
                self.nn.append(ffn(embed_dim, 1, width, act, dropout))
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
        elif self.mode == "direct-elemtype":
            nn_out = self.nn(x)
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            nn_out = torch.sum(elemtype_mask * nn_out, axis=-1, keepdims=True)
            return nn_out
        elif self.mode == "direct-elemtype-split":
            elem_outs = []
            for elem in range(len(self.elemtypes)):
                elem_outs.append(self.nn[elem](x))
            elemtype_mask = torch.cat([elems[..., 0:1] == elemtype for elemtype in self.elemtypes], axis=-1)
            elem_outs = torch.cat(elem_outs, axis=-1)
            return torch.sum(elem_outs * elemtype_mask, axis=-1, keepdims=True)
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
        num_convs=2,
        dropout_ff=0.0,
        activation="elu",
        layernorm=True,
        conv_type="attention",
        input_encoding="joint",
        pt_mode="linear",
        eta_mode="linear",
        sin_phi_mode="linear",
        cos_phi_mode="linear",
        energy_mode="linear",
        # element types which actually exist in the dataset
        elemtypes_nonzero=[1, 4, 5, 6, 8, 9, 10, 11],
        # should the conv layer outputs be concatted (concat) or take the last (last)
        learned_representation_mode="last",
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
        dropout_conv_reg_mha=0.0,
        dropout_conv_reg_ff=0.0,
        dropout_conv_id_mha=0.0,
        dropout_conv_id_ff=0.0,
        use_pre_layernorm=False,
    ):
        super(MLPF, self).__init__()

        self.conv_type = conv_type

        self.act = get_activation(activation)

        self.learned_representation_mode = learned_representation_mode

        self.input_encoding = input_encoding

        self.input_dim = input_dim
        self.num_convs = num_convs

        self.bin_size = bin_size
        self.elemtypes_nonzero = elemtypes_nonzero

        self.use_pre_layernorm = use_pre_layernorm

        if self.conv_type == "attention":
            embedding_dim = num_heads * head_dim
            width = num_heads * head_dim

        # embedding of the inputs
        if self.num_convs != 0:
            if self.input_encoding == "joint":
                self.nn0_id = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)
                self.nn0_reg = ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff)
            elif self.input_encoding == "split":
                self.nn0_id = nn.ModuleList()
                for ielem in range(len(self.elemtypes_nonzero)):
                    self.nn0_id.append(ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff))
                self.nn0_reg = nn.ModuleList()
                for ielem in range(len(self.elemtypes_nonzero)):
                    self.nn0_reg.append(ffn(self.input_dim, embedding_dim, width, self.act, dropout_ff))

            if self.conv_type == "attention":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()

                for i in range(self.num_convs):
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
                        )
                    )
            elif self.conv_type == "gnn_lsh":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(self.num_convs):
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

        if self.learned_representation_mode == "concat":
            decoding_dim = self.num_convs * embedding_dim
        elif self.learned_representation_mode == "last":
            decoding_dim = embedding_dim

        # DNN that acts on the node level to predict the PID
        self.nn_binary_particle = ffn(decoding_dim, 2, width, self.act, dropout_ff)
        self.nn_pid = ffn(decoding_dim, num_classes, width, self.act, dropout_ff)

        # elementwise DNN for node momentum regression
        embed_dim = decoding_dim
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

        embeddings_id, embeddings_reg = [], []
        if self.num_convs != 0:
            if self.input_encoding == "joint":
                embedding_id = self.nn0_id(Xfeat_normed)
                embedding_reg = self.nn0_reg(Xfeat_normed)
            elif self.input_encoding == "split":
                embedding_id = torch.stack([nn0(Xfeat_normed) for nn0 in self.nn0_id], axis=-1)
                elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)
                embedding_id = torch.sum(embedding_id * elemtype_mask.unsqueeze(-2), axis=-1)

                embedding_reg = torch.stack([nn0(Xfeat_normed) for nn0 in self.nn0_reg], axis=-1)
                elemtype_mask = torch.cat([X_features[..., 0:1] == elemtype for elemtype in self.elemtypes_nonzero], axis=-1)
                embedding_reg = torch.sum(embedding_reg * elemtype_mask.unsqueeze(-2), axis=-1)

            for num, conv in enumerate(self.conv_id):
                conv_input = embedding_id if num == 0 else embeddings_id[-1]
                out_padded = conv(conv_input, mask, embedding_id)
                embeddings_id.append(out_padded)
            for num, conv in enumerate(self.conv_reg):
                conv_input = embedding_reg if num == 0 else embeddings_reg[-1]
                out_padded = conv(conv_input, mask, embedding_reg)
                embeddings_reg.append(out_padded)

        # id input
        if self.learned_representation_mode == "concat":
            final_embedding_id = torch.cat(embeddings_id, axis=-1)
        elif self.learned_representation_mode == "last":
            final_embedding_id = torch.cat([embeddings_id[-1]], axis=-1)

        if self.use_pre_layernorm:
            final_embedding_id = self.final_norm_id(final_embedding_id)

        preds_binary_particle = self.nn_binary_particle(final_embedding_id)
        preds_pid = self.nn_pid(final_embedding_id)

        # pred_charge = self.nn_charge(final_embedding_id)

        # regression input
        if self.learned_representation_mode == "concat":
            final_embedding_reg = torch.cat(embeddings_reg, axis=-1)
        elif self.learned_representation_mode == "last":
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
        pz_real = pt_real * torch.sinh(preds_eta.detach())
        e_real = torch.log(torch.sqrt(pt_real**2 + pz_real**2) / X_features[..., 5:6])
        e_real[~mask] = 0
        e_real[torch.isinf(e_real)] = 0
        e_real[torch.isnan(e_real)] = 0
        preds_energy = e_real + torch.nn.functional.relu(self.nn_energy(X_features, final_embedding_reg, X_features[..., 5:6]))
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)
        return preds_binary_particle, preds_pid, preds_momentum


def set_save_attention(model, outdir, save_attention):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if model.conv_type == "attention":
        for iconv in range(model.num_convs):
            model.conv_id[iconv].outdir = outdir
            model.conv_reg[iconv].outdir = outdir
            model.conv_id[iconv].save_attention = save_attention
            model.conv_reg[iconv].save_attention = save_attention