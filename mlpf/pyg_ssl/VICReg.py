from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import GravNetConv, MessagePassing
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_scatter import scatter

from .utils import CLUSTERS_X, COMMON_X, TRACKS_X


# define the Encoder that learns latent representations of tracks and clusters
# these representations will be used by MLPF which is the downstream task
class ENCODER(nn.Module):
    def __init__(
        self,
        width=126,
        embedding_dim=34,
        num_convs=2,
        space_dim=4,
        propagate_dim=22,
        k=8,
    ):
        super(ENCODER, self).__init__()

        self.act = nn.ELU

        ### 1. different embedding of tracks/clusters
        self.nn1 = nn.Sequential(
            nn.Linear(TRACKS_X, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, embedding_dim),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(CLUSTERS_X, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, embedding_dim),
        )

        ### 2. same GNN for tracks/clusters
        self.conv = nn.ModuleList()
        for i in range(num_convs):
            self.conv.append(
                GravNetConv(
                    embedding_dim,
                    embedding_dim,
                    space_dimensions=space_dim,
                    propagate_dimensions=propagate_dim,
                    k=k,
                )
            )

    def forward(self, tracks, clusters):

        embedding_tracks = self.nn1(tracks.x.float())
        embedding_clusters = self.nn2(clusters.x.float())

        # perform a series of graph convolutions
        for num, conv in enumerate(self.conv):
            embedding_tracks = conv(embedding_tracks, tracks.batch)
            embedding_clusters = conv(embedding_clusters, clusters.batch)

        return embedding_tracks, embedding_clusters


# define the decoder that expands the latent representations of tracks and clusters
class DECODER(nn.Module):
    def __init__(
        self,
        input_dim=34,
        width=126,
        output_dim=200,
    ):
        super(DECODER, self).__init__()

        self.act = nn.ELU

        ############################ DECODER
        self.expander = nn.Sequential(
            nn.Linear(input_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, output_dim),
        )

    def forward(self, out_tracks, out_clusters):

        return self.expander(out_tracks), self.expander(out_clusters)
