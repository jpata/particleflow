import numpy as np
import mplhep

import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

import LRP

#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=12, hidden_dim=256, hidden_dim_nn1=64, input_encoding=12, encoding_dim=64,
        output_dim_id=6,
        output_dim_p4=6,
        space_dim=4, propagate_dimensions=22, nearest=16,
        target="gen", nn1=True, nn3=True):

        super(PFNet7, self).__init__()

        self.act = nn.LeakyReLU
        self.act_f = torch.nn.functional.leaky_relu
        self.elu = nn.ELU

        # (1) DNN
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_nn1),
            self.elu(),
            nn.Linear(hidden_dim_nn1, hidden_dim_nn1),
            self.elu(),
            nn.Linear(hidden_dim_nn1, input_encoding),
        )

        # (2) CNN: Gravnet layer
        self.conv1 = LRP.GravNetConv(input_encoding, encoding_dim, space_dim, propagate_dimensions, nearest)

        # (3) DNN layer: classifying PID
        self.nn2 = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        # (4) DNN layer: regressing p4
        self.nn3 = nn.Sequential(
            nn.Linear(encoding_dim + output_dim_id + input_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, hidden_dim),
            self.elu(),
            nn.Linear(hidden_dim, output_dim_p4),
        )


    def forward(self, data):

        x0 = data.x

        # Encoder/Decoder step
        x = self.nn1(x0)

        # Gravnet step
        x, edge_index, edge_weight, after_message, before_message = self.conv1(x)
        x = self.act_f(x)                 # act by nonlinearity

        # DNN to predict PID
        pred_ids = self.nn2(x)

        # DNN to predict p4
        nn3_input = torch.cat([x0, pred_ids, x], axis=-1)
        pred_p4 = self.nn3(nn3_input)

        return pred_ids, pred_p4, data.ygen_id, data.ygen, data.ycand_id, data.ycand, edge_index, edge_weight, after_message, before_message
