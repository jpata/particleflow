import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset

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
    def __init__(self, input_dim=3, hidden_dim=32, edge_dim=1, output_dim=1, n_iters=1, aggr='add'):
        super(EdgeNet, self).__init__()
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

        self.nodenetwork = EdgeConvWithEdgeAttr(nn=convnn,aggr=aggr)

    def forward(self, data):
        X = self.batchnorm(data.x)
        H = self.inputnet(X)
        data.x = torch.cat([H,X],dim=-1)
        for i in range(self.n_iters):
            H = self.nodenetwork(data.x,data.edge_index,data.edge_attr)
            data.x = torch.cat([H,X],dim=-1)
        row,col = data.edge_index        
        output = self.edgenetwork(torch.cat([data.x[row],data.x[col],data.edge_attr],dim=-1)).squeeze(-1)
        return output
