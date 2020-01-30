import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset

import networkx
import numpy as np
import scipy
import scipy.sparse

class EdgeConvWithEdgeAttr(MessagePassing):
    def __init__(self, nn, aggr='max', **kwargs):
        super(EdgeConvWithEdgeAttr, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        return self.nn(torch.cat([x_i, x_j - x_i, pseudo], dim=1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class EdgeNet(nn.Module):
    def __init__(self, device, input_dim=3, hidden_dim=32, edge_dim=1, output_dim=1, n_iters=1, aggr='add'):
        super(EdgeNet, self).__init__()
        self.device = device
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim)+edge_dim, 2*hidden_dim),
                               nn.ReLU(),
                               nn.Linear(2*hidden_dim, hidden_dim),
                               nn.Tanh()
        )
        self.n_iters = n_iters
        
        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(hidden_dim+input_dim)+edge_dim,2*hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(2*hidden_dim, output_dim),
                                         nn.Sigmoid())
        self.candidatenetwork = nn.Sequential(nn.Linear(3*(hidden_dim + input_dim),hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, 3*3),
                                         )

        self.nodenetwork = EdgeConvWithEdgeAttr(nn=convnn,aggr=aggr)

    def forward(self, data):
        X = self.batchnorm(data.x)
        H = self.inputnet(X)
        x = torch.cat([H,X],dim=-1)
        for i in range(self.n_iters):
            H = self.nodenetwork(x,data.edge_index,data.edge_attr)
            x = torch.cat([H,X],dim=-1)
        row,col = data.edge_index        
        output = self.edgenetwork(torch.cat([x[row], x[col], data.edge_attr],dim=-1)).squeeze(-1)
       
        m = scipy.sparse.coo_matrix((output.detach().cpu().numpy()>0.9, (row.cpu().numpy(), col.cpu().numpy())))
        m = m.todense()
        g = networkx.from_numpy_matrix(m)
        node_cluster = torch.zeros(x.shape[0], dtype=torch.float)
        for isg, nodes in enumerate(networkx.connected_components(g)):
            for node in nodes:
                node_cluster[node] = isg
        preds_y = []
        pred_candidates = torch.zeros((x.shape[0], 3), dtype=torch.float).to(device=self.device)
        for cl in node_cluster.unique():
            m = node_cluster == cl
            w = torch.where(m)
            sx = x[m]
            nelem = sx.shape[0]
            if nelem <= 3:
                padded_x = torch.nn.functional.pad(sx, (0, 0, 0, 3 - sx.shape[0]))
                pred_y = self.candidatenetwork(padded_x.view(3*sx.shape[1])).view((3,3))
                preds_y += [pred_y[:nelem]]
                pred_candidates[w] = pred_y[:nelem]
        return output, pred_candidates
