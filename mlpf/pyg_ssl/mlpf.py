import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.utils
from torch_geometric.nn.conv import GravNetConv  # also returns edge index

from .utils import NUM_CLASSES

# from pyg_ssl.gravnet import GravNetConv  # also returns edge index


class GravNetLayer(nn.Module):
    def __init__(self, embedding_dim, space_dimensions, propagate_dimensions, k, dropout):
        super(GravNetLayer, self).__init__()
        self.conv1 = GravNetConv(
            embedding_dim, embedding_dim, space_dimensions=space_dimensions, propagate_dimensions=propagate_dimensions, k=k
        )
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, batch_index):
        # possibly do something with edge index
        # x_new, edge_index, edge_weight = self.conv1(x, batch_index)
        x_new = self.conv1(x, batch_index)
        x_new = self.dropout(x_new)
        x = self.norm1(x + x_new)
        return x


class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, width=128, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.act = nn.ELU
        self.mha = torch.nn.MultiheadAttention(embedding_dim, 8, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.seq = torch.nn.Sequential(
            nn.Linear(embedding_dim, width), self.act(), nn.Linear(width, embedding_dim), self.act()
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):

        x = self.norm0(x + self.mha(x, x, x, key_padding_mask=mask, need_weights=False)[0])
        x = self.norm1(x + self.seq(x))
        x = self.dropout(x)
        x = x * (~mask.unsqueeze(-1))
        return x


def ffn(input_dim, output_dim, width, act, dropout, ssl):
    if ssl:
        return nn.Sequential(
            nn.Linear(input_dim, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Linear(width, output_dim),
        )
    else:
        return nn.Sequential(
            nn.Linear(input_dim, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            act(),
            torch.nn.LayerNorm(width),
            nn.Linear(width, output_dim),
        )


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        embedding_dim=128,
        width=126,
        num_convs=2,
        k=32,
        propagate_dimensions=32,
        space_dimensions=4,
        dropout=0.4,
        ssl=False,
        VICReg_embedding_dim=0,
    ):
        super(MLPF, self).__init__()

        self.act = nn.ELU
        self.dropout = dropout
        self.input_dim = input_dim
        self.num_convs = num_convs
        self.ssl = ssl  # boolean that is True for ssl and False for native mlpf

        # embedding of the inputs
        if num_convs != 0:
            self.nn0 = nn.Sequential(
                nn.Linear(input_dim, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, embedding_dim),
            )

            self.conv_type = "gravnet"
            # GNN that uses the embeddings learnt by VICReg as the input features
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
                    self.conv_id.append(SelfAttentionLayer(embedding_dim))
                    self.conv_reg.append(SelfAttentionLayer(embedding_dim))

        decoding_dim = input_dim + num_convs * embedding_dim
        if ssl:
            decoding_dim += VICReg_embedding_dim

        # DNN that acts on the node level to predict the PID
        self.nn_id = ffn(decoding_dim, NUM_CLASSES, width, self.act, dropout, ssl)

        # elementwise DNN for node momentum regression
        self.nn_pt = ffn(decoding_dim + NUM_CLASSES, 1, width, self.act, dropout, ssl)
        self.nn_eta = ffn(decoding_dim + NUM_CLASSES, 1, width, self.act, dropout, ssl)
        self.nn_phi = ffn(decoding_dim + NUM_CLASSES, 2, width, self.act, dropout, ssl)
        self.nn_energy = ffn(decoding_dim + NUM_CLASSES, 1, width, self.act, dropout, ssl)

        # elementwise DNN for node charge regression, classes (-1, 0, 1)
        self.nn_charge = ffn(decoding_dim + NUM_CLASSES, 3, width, self.act, dropout, ssl)

    def forward(self, batch):

        # unfold the Batch object
        if self.ssl:
            input_ = batch.x.float()[:, : self.input_dim]
            VICReg_embeddings = batch.x.float()[:, self.input_dim :]
        else:
            input_ = batch.x.float()

        batch_idx = batch.batch

        embeddings_id = []
        embeddings_reg = []

        if self.num_convs != 0:
            embedding = self.nn0(input_)

            if self.conv_type == "gravnet":
                # perform a series of graph convolutions
                for num, conv in enumerate(self.conv_id):
                    conv_input = embedding if num == 0 else embeddings_id[-1]
                    embeddings_id.append(conv(conv_input, batch_idx))
                for num, conv in enumerate(self.conv_reg):
                    conv_input = embedding if num == 0 else embeddings_reg[-1]
                    embeddings_reg.append(conv(conv_input, batch_idx))
            elif self.conv_type == "attention":
                for num, conv in enumerate(self.conv_id):
                    conv_input = embedding if num == 0 else embeddings_id[-1]
                    input_padded, mask = torch_geometric.utils.to_dense_batch(conv_input, batch_idx)
                    out_padded = conv(input_padded, ~mask)
                    out_stacked = torch.cat([out_padded[i][mask[i]] for i in range(out_padded.shape[0])])
                    assert out_stacked.shape[0] == conv_input.shape[0]
                    embeddings_id.append(out_stacked)
                for num, conv in enumerate(self.conv_reg):
                    conv_input = embedding if num == 0 else embeddings_reg[-1]
                    input_padded, mask = torch_geometric.utils.to_dense_batch(conv_input, batch_idx)
                    out_padded = conv(input_padded, ~mask)
                    out_stacked = torch.cat([out_padded[i][mask[i]] for i in range(out_padded.shape[0])])
                    assert out_stacked.shape[0] == conv_input.shape[0]
                    embeddings_reg.append(out_stacked)

        if self.ssl:
            embedding_id = torch.cat([input_] + embeddings_id + [VICReg_embeddings], axis=-1)
        else:
            embedding_id = torch.cat([input_] + embeddings_id, axis=-1)

        # predict the PIDs
        preds_id = self.nn_id(embedding_id)

        if self.ssl:
            embedding_reg = torch.cat([input_] + embeddings_reg + [preds_id] + [VICReg_embeddings], axis=-1)
        else:
            embedding_reg = torch.cat([input_] + embeddings_reg + [preds_id], axis=-1)

        # predict the 4-momentum, add it to the (pt, eta, sin phi, cos phi, E) of the input PFelement
        # the feature order is defined in fcc/postprocessing.py -> track_feature_order, cluster_feature_order
        preds_pt = self.nn_pt(embedding_reg) + input_[:, 1:2]
        preds_eta = self.nn_eta(embedding_reg) + input_[:, 2:3]
        preds_phi = self.nn_phi(embedding_reg) + input_[:, 3:5]
        preds_energy = self.nn_energy(embedding_reg) + input_[:, 5:6]
        preds_momentum = torch.cat([preds_pt, preds_eta, preds_phi, preds_energy], axis=-1)
        pred_charge = self.nn_charge(embedding_reg)
        return preds_id, preds_momentum, pred_charge
