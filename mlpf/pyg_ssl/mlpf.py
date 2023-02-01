import torch
import torch.nn as nn
from torch_geometric.nn.conv import GravNetConv

from .utils import NUM_CLASSES


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        width=126,
        num_convs=5,
        k=8,
        embedding_dim=128,
        native_mlpf=False,
        propagate_dimensions=32,
        space_dimensions=4,
    ):
        super(MLPF, self).__init__()

        self.act = nn.ELU
        self.native_mlpf = native_mlpf  # boolean that is true for native mlpf and false for ssl

        if native_mlpf:
            # embedding of the inputs that is necessary for native mlpf training
            self.nn0 = nn.Sequential(
                nn.Linear(input_dim, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, embedding_dim),
            )

        # GNN that uses the embeddings learnt by VICReg as the input features
        self.conv_id = nn.ModuleList()
        self.conv_reg = nn.ModuleList()
        for i in range(num_convs):
            self.conv_id.append(
                GravNetConv(
                    embedding_dim,
                    embedding_dim,
                    space_dimensions=space_dimensions,
                    propagate_dimensions=propagate_dimensions,
                    k=k,
                )
            )
            self.conv_reg.append(
                GravNetConv(
                    embedding_dim,
                    embedding_dim,
                    space_dimensions=space_dimensions,
                    propagate_dimensions=propagate_dimensions,
                    k=k,
                )
            )

        # DNN that acts on the node level to predict the PID
        self.nn_id = nn.Sequential(
            nn.Linear(input_dim + num_convs * embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, NUM_CLASSES),
        )

        # elementwise DNN for node momentum regression
        self.nn_reg = nn.Sequential(
            nn.Linear(input_dim + num_convs * embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 4),
        )

        # elementwise DNN for node charge regression
        self.nn_charge = nn.Sequential(
            nn.Linear(input_dim + num_convs * embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 1),
        )

    def forward(self, batch):

        # unfold the Batch object
        input_ = batch.x.float()
        batch = batch.batch

        # if `native_mlpf` then use then embed the inputs first (otherwise VICReg provides the embeddings)
        if self.native_mlpf:
            embedding = self.nn0(input_)
        else:
            embedding = input_

        embeddings_id = []
        embeddings_reg = []

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv_id):
            conv_input = embedding if num == 0 else embeddings_id[-1]
            embeddings_id.append(conv(conv_input, batch))

        for num, conv in enumerate(self.conv_reg):
            conv_input = embedding if num == 0 else embeddings_reg[-1]
            embeddings_reg.append(conv(conv_input, batch))

        embedding_id = torch.cat([input_] + embeddings_id, axis=-1)
        embedding_reg = torch.cat([input_] + embeddings_reg, axis=-1)

        # predict the PIDs
        preds_id = self.nn_id(embedding_id)

        # predict the 4-momentum, add it to the (pt, eta, phi, E) of the PFelement
        preds_momentum = self.nn_reg(embedding_reg) + input_[:, 1:5]

        pred_charge = self.nn_charge(embedding_reg)

        return preds_id, preds_momentum, pred_charge
