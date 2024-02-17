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
        self.attention_type = attention_type
        self.act = nn.ELU
        if self.attention_type == "flash_external":
            from flash_attn.modules.mha import MHA
            self.mha = MHA(embedding_dim, num_heads)
        else:
            self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(
            nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act()
        )
        self.dropout = torch.nn.Dropout(dropout)
        _logger.info("using attention_type={}".format(attention_type))
        #params for torch sdp_kernel
        self.attn_params = {
            "math": {"enable_math": True, "enable_mem_efficient": False, "enable_flash": False},
            "efficient": {"enable_math": False, "enable_mem_efficient": True, "enable_flash": False},
            "flash": {"enable_math": False, "enable_mem_efficient": False, "enable_flash": True},
        }

        self.add0 = torch.ao.nn.quantized.FloatFunctional()
        self.add1 = torch.ao.nn.quantized.FloatFunctional()
        self.mul = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x, mask):
        # explicitly call the desired attention mechanism
        if self.attention_type == "flash_external":
            mha_out = self.mha(x)
        else:
            with sdp_kernel(**self.attn_params[self.attention_type]):
                mha_out = self.mha(x, x, x, need_weights=False)[0]

        x = self.add0.add(x, mha_out)
        x = self.norm0(x)
        x = self.add1.add(x, self.seq(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.mul.mul(x, mask.unsqueeze(-1))
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
    def __init__(self, mode, embed_dim, width, act, dropout, elemtypes):
        super(RegressionOutput, self).__init__()
        self.mode = mode
        self.elemtypes = elemtypes

        # single output
        if self.mode == "direct" or self.mode == "additive" or self.mode == "multiplicative":
            self.nn = ffn(embed_dim, 1, width, act, dropout)
        # two outputs
        elif self.mode == "linear":
            self.add = torch.ao.nn.quantized.FloatFunctional()
            self.mul = torch.ao.nn.quantized.FloatFunctional()
            self.nn = ffn(embed_dim, 2, width, act, dropout)
        elif self.mode == "direct-elemtype":
            self.nn = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
        elif self.mode == "additive-elemtype":
            self.nn = ffn(embed_dim, len(self.elemtypes), width, act, dropout)
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
            return self.add.add(self.mul.mul(orig_value, nn_out[..., 0:1]), nn_out[..., 1:2])
        elif self.mode == "direct-elemtype":
            nn_out = self.nn(x)
            elemtype_mask = torch.cat([elems[..., 0:1]==elemtype for elemtype in self.elemtypes], axis=-1)
            res = torch.sum(elemtype_mask*nn_out, axis=-1, keepdims=True)
            return res
        elif self.mode == "additive-elemtype":
            nn_out = self.nn(x)
            elemtype_mask = torch.cat([elems[..., 0:1]==elemtype for elemtype in self.elemtypes], axis=-1)
            correction = torch.sum(elemtype_mask*nn_out, axis=-1, keepdims=True)
            return orig_value + correction
        elif self.mode == "linear-elemtype":
            nn_out1 = self.nn1(x)
            nn_out2 = self.nn2(x)
            elemtype_mask = torch.cat([elems[..., 0:1]==elemtype for elemtype in self.elemtypes], axis=-1)
            a = torch.sum(elemtype_mask*nn_out1, axis=-1, keepdims=True)
            b = torch.sum(elemtype_mask*nn_out2, axis=-1, keepdims=True)
            return orig_value*a + b


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
        input_encoding="joint",
        pt_mode="additive-elemtype",
        eta_mode="additive-elemtype",
        sin_phi_mode="additive-elemtype",
        cos_phi_mode="additive-elemtype",
        energy_mode="additive-elemtype",
        elemtypes=[1,4,5,6,8,9,10,11],
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

        self.input_encoding = input_encoding

        self.dropout = dropout
        self.input_dim = input_dim
        self.num_convs = num_convs

        self.bin_size = bin_size
        self.elemtypes = elemtypes

        # embedding of the inputs
        if num_convs != 0:
            if self.input_encoding == "joint":    
                self.nn0 = ffn(input_dim, embedding_dim, width, self.act, dropout)
            elif self.input_encoding == "split":    
                self.nn0 = nn.ModuleList()
                for ielem in range(len(self.elemtypes)):
                    self.nn0.append(ffn(input_dim, embedding_dim, width, self.act, dropout))

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
        self.nn_pt = RegressionOutput(pt_mode, embed_dim, width, self.act, dropout, self.elemtypes)
        self.nn_eta = RegressionOutput(eta_mode, embed_dim, width, self.act, dropout, self.elemtypes)
        self.nn_sin_phi = RegressionOutput(sin_phi_mode, embed_dim, width, self.act, dropout, self.elemtypes)
        self.nn_cos_phi = RegressionOutput(cos_phi_mode, embed_dim, width, self.act, dropout, self.elemtypes)
        self.nn_energy = RegressionOutput(energy_mode, embed_dim, width, self.act, dropout, self.elemtypes)

        # elementwise DNN for node charge regression, classes (-1, 0, 1)
        self.nn_charge = ffn(decoding_dim + num_classes, 3, width, self.act, dropout)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant1 = torch.ao.quantization.DeQuantStub()
        self.dequant2 = torch.ao.quantization.DeQuantStub()
        self.dequant3 = torch.ao.quantization.DeQuantStub()

    # @torch.compile
    def forward(self, X_features, batch_or_mask):
        X_features = self.quant(X_features)
        embeddings_id, embeddings_reg = [], []
        if self.num_convs != 0:
            if self.input_encoding == "joint":
                embedding = self.nn0(X_features)
            elif self.input_encoding == "split":
                embedding = torch.stack([nn0(X_features) for nn0 in self.nn0], axis=-1)
                elemtype_mask = torch.cat([X_features[..., 0:1]==elemtype for elemtype in self.elemtypes], axis=-1)
                embedding = torch.sum(embedding*elemtype_mask.unsqueeze(-2), axis=-1)
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
                    out_padded = conv(conv_input, mask)
                    embeddings_id.append(out_padded)
                for num, conv in enumerate(self.conv_reg):
                    conv_input = embedding if num == 0 else embeddings_reg[-1]
                    out_padded = conv(conv_input, mask)
                    embeddings_reg.append(out_padded)

        embedding_id = torch.cat([X_features] + embeddings_id, axis=-1)
        preds_id = self.nn_id(embedding_id)

        # regression input
        embedding_reg = torch.cat([X_features] + embeddings_reg + [preds_id], axis=-1)

        preds_id = self.dequant1(preds_id)

        # do some sanity checks on the PFElement input data
        # assert torch.all(torch.abs(input_[:, 3]) <= 1.0)  # sin_phi
        # assert torch.all(torch.abs(input_[:, 4]) <= 1.0)  # cos_phi
        # assert torch.all(input_[:, 1] >= 0.0)  # pt
        # assert torch.all(input_[:, 5] >= 0.0)  # energy

        # The PFElement feature order in X_features defined in fcc/postprocessing.py
        preds_pt = self.nn_pt(X_features, embedding_reg, X_features[..., 1:2])
        preds_eta = self.nn_eta(X_features, embedding_reg, X_features[..., 2:3])
        preds_sin_phi = self.nn_sin_phi(X_features, embedding_reg, X_features[..., 3:4])
        preds_cos_phi = self.nn_cos_phi(X_features, embedding_reg, X_features[..., 4:5])
        preds_energy = self.nn_energy(X_features, embedding_reg, X_features[..., 5:6])
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_sin_phi, preds_cos_phi, preds_energy], axis=-1)
        preds_momentum = self.dequant2(preds_momentum)

        pred_charge = self.nn_charge(embedding_reg)
        pred_charge = self.dequant3(pred_charge)

        return preds_id, preds_momentum, pred_charge
