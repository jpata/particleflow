import sys
import os
import math

from comet_ml import Experiment

import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv, DenseGCNConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from gravnet import GravNetConv
from torch_geometric.data import Data, DataListLoader, Batch
from torch.utils.data import random_split

import torch_cluster

from glob import glob
import numpy as np
import os.path as osp
import pickle

import math
import time
import numba
import tqdm
import sklearn
import pandas

import mplhep

from sklearn.metrics import accuracy_score

import graph_data
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels
from sklearn.metrics import confusion_matrix


#Model with gravnet clustering
class PFNet7(nn.Module):
    def __init__(self,
        input_dim=12, hidden_dim=32, encoding_dim=256,
        output_dim_id=6,
        output_dim_p4=6,
        convlayer="gravnet-radius",
        convlayer2="none",
        space_dim=2, nearest=3, dropout_rate=0.0, activation="leaky_relu", return_edges=False, radius=0.1, input_encoding=0):

        super(PFNet7, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_edges = return_edges
        self.convlayer = convlayer
        self.input_encoding = input_encoding

        if activation == "leaky_relu":
            self.act = nn.LeakyReLU
            self.act_f = torch.nn.functional.leaky_relu
        elif activation == "selu":
            self.act = nn.SELU
            self.act_f = torch.nn.functional.selu
        elif activation == "relu":
            self.act = nn.ReLU
            self.act_f = torch.nn.functional.relu

        # if you want to add an initial encoding of the input
        conv_in_dim = input_dim
        if self.input_encoding>0:
            self.nn1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.act(),
                nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
                self.act(),
                nn.Linear(hidden_dim, encoding_dim),
            )
            conv_in_dim = encoding_dim

        # (1) GNN layer
        if convlayer == "gravnet-knn":
            self.conv1 = GravNetConv(conv_in_dim, encoding_dim, space_dim, hidden_dim, nearest, neighbor_algo="knn")
        elif convlayer == "gravnet-radius":
            self.conv1 = GravNetConv(conv_in_dim, encoding_dim, space_dim, hidden_dim, nearest, neighbor_algo="radius", radius=radius)
        else:
            raise Exception("Unknown convolution layer: {}".format(convlayer))

        #decoding layer receives the raw inputs and the gravnet output
        num_decode_in = input_dim + encoding_dim

        # (2) another GNN layer if you want
        self.convlayer2 = convlayer2
        if convlayer2 == "none":
            self.conv2_1 = None
            self.conv2_2 = None
        elif convlayer2 == "sgconv":
            self.conv2_1 = SGConv(num_decode_in, hidden_dim, K=1)
            self.conv2_2 = SGConv(num_decode_in, hidden_dim, K=1)
            num_decode_in += hidden_dim
        elif convlayer2 == "graphunet":
            self.conv2_1 = GraphUNet(num_decode_in, hidden_dim, hidden_dim, 2, pool_ratios=0.1)
            self.conv2_2 = GraphUNet(num_decode_in, hidden_dim, hidden_dim, 2, pool_ratios=0.1)
            num_decode_in += hidden_dim
        elif convlayer2 == "gatconv":
            self.conv2_1 = GATConv(num_decode_in, hidden_dim, 4, concat=False, dropout=dropout_rate)
            self.conv2_2 = GATConv(num_decode_in, hidden_dim, 4, concat=False, dropout=dropout_rate)
            num_decode_in += hidden_dim
        else:
            raise Exception("Unknown convolution layer: {}".format(convlayer2))

        # (3) dropout layer if you want
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # (4) DNN layer: classifying PID
        self.nn2 = nn.Sequential(
            nn.Linear(num_decode_in, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_id),
        )

        # (5) DNN layer: regressing p4
        self.nn3 = nn.Sequential(
            nn.Linear(num_decode_in + output_dim_id, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            self.act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )

    def forward(self, data):

        #encode the inputs
        x = data.x

        if self.input_encoding:
            x = self.nn1(x)

        #Run a clustering of the inputs that returns the new_edge_index.. this is the KNN step..
        new_edge_index, x = self.conv1(x)
        x1 = self.act_f(x)

        #run a second convolution
        if self.convlayer2 != "none":
            conv2_input = torch.cat([data.x, x1], axis=-1)
            x2_1 = self.act_f(self.conv2_1(conv2_input, new_edge_index))
            x2_2 = self.act_f(self.conv2_2(conv2_input, new_edge_index))
            nn2_input = torch.cat([data.x, x1, x2_1], axis=-1)
        else:
            nn2_input = torch.cat([data.x, x1], axis=-1)

        #Decode convolved graph nodes to pdgid and p4
        cand_ids = self.nn2(self.dropout1(nn2_input))

        if self.convlayer2 != "none":
            nn3_input = torch.cat([data.x, x1, x2_2, cand_ids], axis=-1)
        else:
            nn3_input = torch.cat([data.x, x1, cand_ids], axis=-1)

        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn3(self.dropout1(nn3_input))
        return cand_ids, cand_p4, new_edge_index
