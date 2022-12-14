from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import GravNetConv, MessagePassing
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_scatter import scatter

from .utils import NUM_CLASSES


# downstream model
class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        width=126,
        num_convs=2,
        k=8,
    ):
        super(MLPF, self).__init__()

        self.act = nn.ELU

        # GNN that uses the embeddings learnt by VICReg as the input features
        self.conv = nn.ModuleList()
        for i in range(num_convs):
            self.conv.append(
                GravNetConv(
                    input_dim,
                    input_dim,
                    space_dimensions=4,
                    propagate_dimensions=22,
                    k=k,
                )
            )

        # DNN that acts on the node level to predict the PID
        self.nn = nn.Sequential(
            nn.Linear(input_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, NUM_CLASSES),
        )

    def forward(self, batch):

        # unfold the Batch object
        input_ = batch.x.float()
        batch = batch.batch

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv):
            embedding = conv(input_, batch)

        # predict the PIDs
        preds_id = self.nn(embedding)

        return preds_id
