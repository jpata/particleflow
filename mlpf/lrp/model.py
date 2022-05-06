import pickle as pkl
import os.path as osp
import os
import sys
from glob import glob

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F

from typing import Optional, Union
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor

try:
    from torch_cluster import knn
except ImportError:
    knn = None
from torch_cluster import knn_graph


class MLPF(nn.Module):
    """
    GNN model based on Gravnet...

    Forward pass returns
        preds: tensor of predictions containing a concatenated representation of the pids and p4
        A: dict() object containing adjacency matrices for each message passing
        msg_activations: dict() object containing activations before each message passing
    """

    def __init__(self,
                 input_dim=12, output_dim_id=6, output_dim_p4=6,
                 embedding_dim=64, hidden_dim1=64, hidden_dim2=60,
                 num_convs=2, space_dim=4, propagate_dim=30, k=8):
        super(MLPF, self).__init__()

        # self.act = nn.ReLU
        self.act = nn.ELU

        # (1) embedding
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            self.act(),
            nn.Linear(hidden_dim1, hidden_dim1),
            self.act(),
            nn.Linear(hidden_dim1, hidden_dim1),
            self.act(),
            nn.Linear(hidden_dim1, embedding_dim),
        )

        self.conv = nn.ModuleList()
        for i in range(num_convs):
            self.conv.append(GravNetConv_LRP(embedding_dim, embedding_dim, space_dim, propagate_dim, k))

        # (3) DNN layer: classifiying pid
        self.nn2 = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, output_dim_id),
        )

        # (4) DNN layer: regressing p4
        self.nn3 = nn.Sequential(
            nn.Linear(input_dim + output_dim_id, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, hidden_dim2),
            self.act(),
            nn.Linear(hidden_dim2, output_dim_p4),
        )

    def forward(self, batch):

        x0 = batch.x

        # embed the inputs
        embedding = self.nn1(x0)

        # preform a series of graph convolutions
        A = {}
        msg_activations = {}
        for num, conv in enumerate(self.conv):
            embedding, A[f'conv.{num}'], msg_activations[f'conv.{num}'] = conv(embedding)

        # predict the pid's
        preds_id = self.nn2(torch.cat([x0, embedding], axis=-1))

        # predict the p4's
        preds_p4 = self.nn3(torch.cat([x0, preds_id], axis=-1))

        return torch.cat([preds_id, preds_p4], axis=-1), A, msg_activations


class GravNetConv_LRP(MessagePassing):
    """
    Copied from pytorch_geometric source code
    Edits:
      - retrieve adjacency matrix (we call A), and the activations before the message passing step (we call msg_activations)
      - switched the execution of self.lin_s & self.lin_p so that the message passing step can substitute out of the box self.lin_s for lrp purposes
      - used reduce='sum' instead of reduce='mean' in the message passing
      - removed skip connection

    The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    The neighbors are constructed in a learnable low-dimensional projection of
    the feature space.
    A second projection of the input feature space is then propagated from the
    neighbors to each vertex using distance weights that are derived by
    applying a Gaussian function to the distances.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}_s|), (|\mathcal{V}_t|))` if bipartite
          *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite


    """

    def __init__(self, in_channels: int, out_channels: int,
                 space_dimensions: int, propagate_dimensions: int, k: int,
                 num_workers: int = 1, **kwargs):
        super().__init__(flow='source_to_target', **kwargs)

        if knn is None:
            raise ImportError('`GravNetConv` requires `torch-cluster`.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.num_workers = num_workers

        self.lin_p = Linear(in_channels, propagate_dimensions)
        self.lin_s = Linear(in_channels, space_dimensions)
        self.lin_out = Linear(propagate_dimensions, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_s.reset_parameters()
        self.lin_p.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""

        is_bipartite: bool = True
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
            is_bipartite = False

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in 'GravNetConv'")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # embed the inputs before message passing
        msg_activations = self.lin_p(x[0])

        # transform to the space dimension to build the graph
        s_l: Tensor = self.lin_s(x[0])
        s_r: Tensor = self.lin_s(x[1]) if is_bipartite else s_l

        edge_index = knn(s_l, s_r, self.k, b[0], b[1]).flip([0])
        # edge_index = knn_graph(s_l, self.k, b[0], b[1]).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_r[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        # return the adjacency matrix of the graph for lrp purposes
        A = to_dense_adj(edge_index.to('cpu'), edge_attr=edge_weight.to('cpu'))[0]  # adjacency matrix

        # message passing
        out = self.propagate(edge_index, x=(msg_activations, None),
                             edge_weight=edge_weight,
                             size=(s_l.size(0), s_r.size(0)))

        return self.lin_out(out), A, msg_activations

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        out_mean = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce='sum')
        return out_mean

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k={self.k})')
