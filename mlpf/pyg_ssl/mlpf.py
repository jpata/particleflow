import torch.nn as nn
from torch_geometric.nn.conv import GravNetConv

from .utils import NUM_CLASSES


class MLPF(nn.Module):
    def __init__(
        self,
        input_dim=34,
        width=126,
        num_convs=2,
        k=8,
        embedding_dim=34,
        native_mlpf=False,
    ):
        super(MLPF, self).__init__()

        self.act = nn.ELU
        self.native_mlpf = native_mlpf  # boolean that is true for native mlpf and false for ssl

        if native_mlpf:
            # embedding of the inputs that is necessary for native mlpf training
            self.nn0 = nn.Sequential(
                nn.Linear(input_dim, 126),
                self.act(),
                nn.Linear(126, 126),
                self.act(),
                nn.Linear(126, 126),
                self.act(),
                nn.Linear(126, embedding_dim),
            )

        # GNN that uses the embeddings learnt by VICReg as the input features
        self.conv = nn.ModuleList()
        for i in range(num_convs):
            self.conv.append(
                GravNetConv(
                    embedding_dim,
                    embedding_dim,
                    space_dimensions=4,
                    propagate_dimensions=22,
                    k=k,
                )
            )

        # DNN that acts on the node level to predict the PID
        self.nn = nn.Sequential(
            nn.Linear(embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, NUM_CLASSES),
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

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv):
            embedding = conv(embedding, batch)

        # predict the PIDs
        preds_id = self.nn(embedding)

        return preds_id
