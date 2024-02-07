import torch
import torch.nn as nn
from torch_geometric.nn.conv import GravNetConv

from .gnn_lsh import CombinedGraphLayer

from torch.backends.cuda import sdp_kernel
from pyg.logger import _logger


class GravNetLayer(nn.Module):
    def __init__(self, embedding_dim, space_dimensions, propagate_dimensions, k, dropout):
        super(GravNetLayer, self).__init__()
        self.conv1 = GravNetConv(
            embedding_dim, embedding_dim, space_dimensions=space_dimensions, propagate_dimensions=propagate_dimensions, k=k
        )
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, batch_index):
        x_new = self.conv1(x, batch_index)
        x_new = self.dropout(x_new)
        x = self.norm1(x + x_new)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=2, width=128, dropout=0.1, attention_type="efficient"):
        super(SelfAttentionLayer, self).__init__()
        self.act = nn.ELU
        self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(
            nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act()
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.attention_type = attention_type
        _logger.info("using attention_type={}".format(attention_type))
        self.attn_params = {
            "math": {"enable_math": True, "enable_mem_efficient": False, "enable_flash": False},
            "efficient": {"enable_math": False, "enable_mem_efficient": True, "enable_flash": False},
            "flash": {"enable_math": False, "enable_mem_efficient": False, "enable_flash": True},
        }

    def forward(self, x, mask):
        # explicitly call the desired attention mechanism
        with sdp_kernel(**self.attn_params[self.attention_type]):
            mha_out = self.mha(x, x, x, need_weights=False)[0]

        x = self.norm0(x + mha_out)
        x = self.norm1(x + self.seq(x))
        x = self.dropout(x)
        x = x * (~mask.unsqueeze(-1))
        return x


class MambaLayer(nn.Module):
    def __init__(self, embedding_dim=128, width=128, dropout=0.1, d_state=16, d_conv=4, expand=2):
        super(MambaLayer, self).__init__()
        self.act = nn.ELU
        from mamba_ssm import Mamba

        self.mamba = Mamba(
            d_model=embedding_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(
            nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act()
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.mamba(x)
        x = self.norm0(x + self.seq(x))
        x = self.dropout(x)
        x = x * (~mask.unsqueeze(-1))
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
    def __init__(self, mode, embed_dim, width, act, dropout):
        super(RegressionOutput, self).__init__()
        self.mode = mode

        # single output
        if self.mode == "direct" or self.mode == "additive" or self.mode == "multiplicative":
            self.nn = ffn(embed_dim, 1, width, act, dropout)
        # two outputs
        elif self.mode == "linear":
            self.nn = ffn(embed_dim, 2, width, act, dropout)

    def forward(self, x, orig_value):
        nn_out = self.nn(x)

        if self.mode == "direct":
            return nn_out
        elif self.mode == "additive":
            return orig_value + nn_out
        elif self.mode == "multiplicative":
            return orig_value * nn_out
        elif self.mode == "linear":
            return orig_value * nn_out[..., 0:1] + nn_out[..., 1:2]


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        num_classes=8,
        embedding_dim=128,
        width=128,
        num_convs=2,
        dropout=0.0,
        activation="elu",
        # gravnet specific parameters
        k=32,
        propagate_dimensions=32,
        space_dimensions=4,
        conv_type="gravnet",
        attention_type="flash",
        # gnn-lsh specific parameters
        bin_size=640,
        max_num_bins=200,
        distance_dim=128,
        layernorm=True,
        num_node_messages=2,
        ffn_dist_hidden_dim=128,
        # self-attention specific parameters
        num_heads=2,
        # mamba specific parameters
        d_state=16,
        d_conv=4,
        expand=2,
        pt_mode="linear",
        eta_mode="linear",
        sin_phi_mode="linear",
        cos_phi_mode="linear",
        energy_mode="linear",
    ):
        super(MLPF, self).__init__()

        self.conv_type = conv_type

        if activation == "elu":
            self.act = nn.ELU
        elif activation == "relu":
            self.act = nn.ReLU
        elif activation == "relu6":
            self.act = nn.ReLU6
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU

        self.dropout = dropout
        self.input_dim = input_dim
        self.num_convs = num_convs

        self.bin_size = bin_size

        # embedding of the inputs
        if num_convs != 0:
            self.nn0 = ffn(input_dim, embedding_dim, width, self.act, dropout)
            if self.conv_type == "gravnet":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(num_convs):
                    self.conv_id.append(GravNetLayer(embedding_dim, space_dimensions, propagate_dimensions, k, dropout))
                    self.conv_reg.append(GravNetLayer(embedding_dim, space_dimensions, propagate_dimensions, k, dropout))
            elif self.conv_type == "attention":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(num_convs):
                    self.conv_id.append(
                        SelfAttentionLayer(embedding_dim, num_heads, width, dropout, attention_type=attention_type)
                    )
                    self.conv_reg.append(
                        SelfAttentionLayer(embedding_dim, num_heads, width, dropout, attention_type=attention_type)
                    )
            elif self.conv_type == "mamba":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(num_convs):
                    self.conv_id.append(MambaLayer(embedding_dim, width, dropout, d_state, d_conv, expand))
                    self.conv_reg.append(MambaLayer(embedding_dim, width, dropout, d_state, d_conv, expand))
            elif self.conv_type == "gnn_lsh":
                self.conv_id = nn.ModuleList()
                self.conv_reg = nn.ModuleList()
                for i in range(num_convs):
                    gnn_conf = {
                        "inout_dim": embedding_dim,
                        "bin_size": self.bin_size,
                        "max_num_bins": max_num_bins,
                        "distance_dim": distance_dim,
                        "layernorm": layernorm,
                        "num_node_messages": num_node_messages,
                        "dropout": dropout,
                        "ffn_dist_hidden_dim": ffn_dist_hidden_dim,
                    }
                    self.conv_id.append(CombinedGraphLayer(**gnn_conf))
                    self.conv_reg.append(CombinedGraphLayer(**gnn_conf))

        decoding_dim = input_dim + num_convs * embedding_dim

        # DNN that acts on the node level to predict the PID
        self.nn_id = ffn(decoding_dim, num_classes, width, self.act, dropout)

        # elementwise DNN for node momentum regression
        embed_dim = decoding_dim + num_classes
        self.nn_pt = RegressionOutput(pt_mode, embed_dim, width, self.act, dropout)
        self.nn_eta = RegressionOutput(eta_mode, embed_dim, width, self.act, dropout)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, embed_dim, width, self.act, dropout)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, embed_dim, width, self.act, dropout)
        self.nn_energy = RegressionOutput(energy_mode, embed_dim, width, self.act, dropout)

        # elementwise DNN for node charge regression, classes (-1, 0, 1)
        self.nn_charge = ffn(decoding_dim + num_classes, 3, width, self.act, dropout)

    # @torch.compile
    def forward(self, X_features, batch_or_mask):
        embeddings_id, embeddings_reg = [], []
        if self.num_convs != 0:
            embedding = self.nn0(X_features)
            if self.conv_type == "gravnet":
                batch_idx = batch_or_mask
                # perform a series of graph convolutions
                for num, conv in enumerate(self.conv_id):
                    conv_input = embedding if num == 0 else embeddings_id[-1]
                    embeddings_id.append(conv(conv_input, batch_idx))
                for num, conv in enumerate(self.conv_reg):
                    conv_input = embedding if num == 0 else embeddings_reg[-1]
                    embeddings_reg.append(conv(conv_input, batch_idx))
            else:
                mask = batch_or_mask
                for num, conv in enumerate(self.conv_id):
                    conv_input = embedding if num == 0 else embeddings_id[-1]
                    out_padded = conv(conv_input, ~mask)
                    embeddings_id.append(out_padded)
                for num, conv in enumerate(self.conv_reg):
                    conv_input = embedding if num == 0 else embeddings_reg[-1]
                    out_padded = conv(conv_input, ~mask)
                    embeddings_reg.append(out_padded)

        embedding_id = torch.cat([X_features] + embeddings_id, axis=-1)
        preds_id = self.nn_id(embedding_id)

        # regression
        embedding_reg = torch.cat([X_features] + embeddings_reg + [preds_id], axis=-1)

        # do some sanity checks on the PFElement input data
        # assert torch.all(torch.abs(input_[:, 3]) <= 1.0)  # sin_phi
        # assert torch.all(torch.abs(input_[:, 4]) <= 1.0)  # cos_phi
        # assert torch.all(input_[:, 1] >= 0.0)  # pt
        # assert torch.all(input_[:, 5] >= 0.0)  # energy

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(embedding_reg, X_features[..., 4:5])
        preds_energy = self.nn_energy(embedding_reg, X_features[..., 5:6])
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)

        pred_charge = self.nn_charge(embedding_reg)

        return preds_id, preds_momentum, pred_charge
