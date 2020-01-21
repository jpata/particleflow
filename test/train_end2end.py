import setGPU

import torch
print(torch.__version__)
import torch_geometric
print(torch_geometric.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv
from torch_geometric.nn import TopKPooling, SAGPooling
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Dataset, Data, DataLoader

from glob import glob
import numpy as np
import os.path as osp

import math
import time
import numba
import tqdm
import sys
import os
#import termplotlib as tpl
  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_data import PFGraphDataset

device = torch.device('cuda')

class EdgeConvWithEdgeAttr(MessagePassing):
    def __init__(self, nn_phi, nn_gamma, aggr='max', **kwargs):
        super(EdgeConvWithEdgeAttr, self).__init__(aggr=aggr, **kwargs)
        self.nn_phi = nn_phi
        self.nn_gamma = nn_gamma
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_phi)
        reset(self.nn_gamma)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        return self.nn_phi(torch.cat([x_i, x_j - x_i, pseudo], dim=1))

    def __repr__(self):
        return '{}(nn_phi={}, nn_gamma={})'.format(self.__class__.__name__, self.nn_phi, self.nn_gamma)
   
    def update(self, aggr_out):
        return self.nn_gamma(aggr_out) 

#Original graph net
class PFNet2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, edge_dim=1, output_dim=4, n_iters=1, aggr='mean'):
        super(PFNet2, self).__init__()
        
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim)+edge_dim, 2*hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(2*hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
        )
        gamma_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
        )
        convnn2 = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim)+hidden_dim, 2*hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(2*hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
        )
        gamma_nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, 4),
        )

        self.n_iters = n_iters
        
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim + input_dim)
        self.batchnorm3 = nn.BatchNorm1d(hidden_dim + input_dim)

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

        self.nodenetwork = EdgeConvWithEdgeAttr(nn_phi=convnn, nn_gamma=gamma_nn, aggr=aggr)
        self.pooling1 = EdgePooling(hidden_dim + input_dim)
        self.pooling2 = EdgePooling(hidden_dim + input_dim)
        self.outnetwork = nn.Sequential(
            nn.Linear(hidden_dim+input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):        
        X = self.batchnorm1(data.x)
        X = data.x
        H = self.inputnet(X)
        x = torch.cat([H,X], dim=-1)

        for i in range(self.n_iters):
            x = self.batchnorm2(x)
            H = self.nodenetwork(x, data.edge_index, data.edge_attr)
            x = torch.cat([H,X], dim=-1)

        x = self.batchnorm3(x)
        r, edge_index, batch, unpool_info1 = self.pooling1(x, data.edge_index, data.batch)
        r, edge_index, batch, unpool_info2 = self.pooling2(r, edge_index, batch)
        r, edge_index, batch = self.pooling2.unpool(r, unpool_info2)
        r, edge_index, batch = self.pooling1.unpool(r, unpool_info1)

        r = self.outnetwork(r)
        #Is this output candidate enabled?
        r1 = torch.sigmoid(r[:, 0])
        #Momentum components
        r[:, 0] = r1
        return r

#GCN based
class PFNet3(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, edge_dim=1, output_dim=4, n_iters=1, aggr='mean'):
        super(PFNet3, self).__init__()
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)        

        self.outnetwork = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        x = self.inputnet(data.x)
        x = self.batchnorm1(x)
        #x = torch.nn.functional.dropout(x, p=0.3)
        x = torch.nn.functional.elu(self.conv1(x, data.edge_index))
        x = torch.nn.functional.elu(self.conv2(x, data.edge_index))
        x = torch.nn.functional.elu(self.conv3(x, data.edge_index))
        #x = torch.nn.functional.dropout(x, p=0.3)
        r = self.outnetwork(x)
        r1 = torch.sigmoid(r[:, 0])
        r[:, 0] = r1
        return r

#Dense all to all (batch size 1 only)
class PFNet4(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputnet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.edgenet = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = self.inputnet(data.x)
        r1 = torch.sigmoid(x[:, 0])
        x[:, 0] = r1
        edges = self.edgenet(data.edge_attr).squeeze(-1)
        return edges, x

#Based on GraphUNet
class PFNet5(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet5, self).__init__()
        self.unet = GraphUNet(input_dim, hidden_dim, hidden_dim,
                              depth=2, pool_ratios=0.2)
        self.outnetwork = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)

    def reset_parameters(self):
        self.unet.reset_parameters()
        self.outnetwork.reset_parameters()
        self.batchnorm1.reset_parameters()

    def forward(self, data):
        r = self.unet(data.x, data.edge_index, data.batch)
        r = self.outnetwork(self.batchnorm1(r))
        r1 = torch.sigmoid(r[:, 0])
        r[:, 0] = r1
        return r

class PFNet6(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet6, self).__init__()
   
        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim) 
        #self.bn2 = nn.BatchNorm1d(input_dim + hidden_dim)
        self.edgenet = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = GCNConv(hidden_dim, hidden_dim) 
        #self.pooling = TopKPooling(hidden_dim, ratio=0.9)
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        batch = data.batch
        
        #encode the inputs
        x = self.inp(data.x)
        edge_index = data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = data.edge_attr.squeeze(-1)

        #Run a graph convolution to embed the nodes
        x = self.conv1(x, data.edge_index, edge_weight)
        #x = self.bn1(x)
        
        #Compute new edge weights based on embedded node pairs
        xpairs = torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=-1)
        edge_weight = self.edgenet(xpairs).squeeze(-1)
        
        #Run a second convolution
        x = self.conv2(x, data.edge_index, edge_weight)

        #Pooling step
        #x, edge_index2, edge_weight2, batch, perm, _ = self.pooling(x, data.edge_index, edge_weight, batch)
        #up = torch.zeros((data.x.shape[0], self.hidden_dim)).to(device)
        #up[perm] = x
        up = x

        #Postprocessing, add initial inputs to encoded hidden layer
        #m = (up[:, 0]!=0).to(dtype=torch.float)
        up = torch.cat([data.x, up], axis=-1)
        #up = self.bn2(up)

        r = self.nn1(up)
        #for i in range(r.shape[1]):
        #    r[:, i] *= m
        r[:, 0] = torch.sigmoid(r[:, 0])
        #r2 = (r[:, 0] > 0.9).to(dtype=torch.float)
        #r[:, 1] *= r2
        #r[:, 2] *= r2
        #r[:, 3] *= r2

        return edge_weight, r

def loss_by_cluster(y_pred, y_true, batches, true_block_ids):
    losses = []
    #hack to create unique IDs by shifting batch number and block id
    block_batch_ids = (10000*batches) + true_block_ids

    uniqs, counts = block_batch_ids.unique(return_counts=True)

    #Compute the sum within each cluster, then average
    offsets = torch.nn.functional.pad(torch.cumsum(counts, axis=0), (1, 0))
   
    #Based on jagged arrays 
    yps = torch.nn.functional.pad(y_pred.cumsum(axis=0), (0, 0, 1, 0))
    yts = torch.nn.functional.pad(y_true.cumsum(axis=0), (0, 0, 1, 0))
    jagged_sum_pred = (yps[offsets[1:]] - yps[offsets[:-1]])
    jagged_sum_tgt = (yts[offsets[1:]] - yts[offsets[:-1]])
   
    #Divide by number of elements in cluster to compute mean 
    #for i in range(jagged_sum_pred.shape[1]):
    #    jagged_sum_pred[:, i] /= counts
    #    jagged_sum_tgt[:, i] /= counts
  
    #Compare aggregated quantities cluster by cluster 
    #loss = torch.nn.functional.mse_loss(torch.clamp(jagged_sum_pred[:, 0], 0, 1), torch.clamp(jagged_sum_tgt[:, 0], 0, 1))
    #loss += torch.nn.functional.mse_loss(jagged_sum_pred[:, 1:], jagged_sum_tgt[:, 1:]) 
    
    #Compare losses cluster by cluster 
    loss1 = torch.nn.functional.binary_cross_entropy(y_pred[:, 0], y_true[:, 0])*5
    loss2 = torch.nn.functional.mse_loss(y_pred[:, 1:], y_true[:, 1:])
    loss = loss1 + loss2
    return (jagged_sum_pred, jagged_sum_tgt, loss)
 
def train(model, loader, batch_size, epoch, optimizer):
    corrs_batch = []
    
    num_pred = []
    num_true = []
   
    is_train = not (optimizer is None)

    losses = []
    for i, data in enumerate(loader):
        data = data.to(device)
        if is_train:
            optimizer.zero_grad()
        edges, output = model(data)

        data.y_candidates[:, 0] = (data.y_candidates[:, 0]!=0).to(dtype=torch.float)
        
        #Loss for output candidate enabled/disabled
        l1 = torch.nn.functional.binary_cross_entropy(output[:, 0], data.y_candidates[:, 0])

        #Loss for candidate p4 properties
        l2 = torch.nn.functional.mse_loss(output[:, 1:], data.y_candidates[:, 1:]) / 5.0
        
        #Loss for edges enabled/disabled in clustering
        l3 = torch.nn.functional.binary_cross_entropy(edges, data.y)

        batch_loss = l1 + l2 + l3
        losses += [(l1.item(), l2.item(), l3.item())]
        
        if is_train:
            batch_loss.backward()
        batch_loss_item = batch_loss.item()
        if is_train:
            optimizer.step()

        #Compute correlation  
        corr_pt = 0.0 
        msk = (output[:, 0]>0.9)
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                output[msk, 1].detach().cpu().numpy(),
                data.y_candidates[msk, 1].detach().cpu().numpy())[0,1]

        corrs_batch += [corr_pt]
  
    corr = np.mean(corrs_batch)
    losses = tuple([sum([l[i] for l in losses]) for i in range(len(losses[0]))])
    return losses, corr

def make_plots(path, losses_train, losses_test, corrs_train, corrs_test, test_loader):
   
    try:
        os.makedirs(path)
    except Exception as e:
        pass
 
    fig = plt.figure(figsize=(5,5))
    plt.plot(losses_train)
    plt.plot(losses_test)
    plt.ylim(0,2*losses_train[-1])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(path + "loss.pdf")
    del fig
    plt.clf()
    
    fig = plt.figure(figsize=(5,5))
    plt.plot(corrs_train)
    plt.plot(corrs_test)
    plt.xlabel("epoch")
    plt.ylabel("pt correlation")
    plt.savefig(path + "corr.pdf")
    del fig
    plt.clf()
 
    for i, data in enumerate(test_loader):
        if i>5:
            break
        d = data.to(device=device)
        edges, output = model(d)
        msk = output[:, 0] > 0.9        
        fig = plt.figure(figsize=(5,5))
        plt.scatter(
            data.y_candidates[msk, 1].detach().cpu().numpy(),
            output[msk, 1].detach().cpu().numpy(),
            marker=".", alpha=0.5)
        plt.plot([0,5],[0,5])
        plt.xlim(0,5)
        plt.ylim(0,5)
        plt.xlabel("pt_true")
        plt.ylabel("pt_pred")
        plt.savefig(path + "pt_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        plt.scatter(
            data.y_candidates[msk, 2].detach().cpu().numpy(),
            output[msk, 2].detach().cpu().numpy(),
            marker=".", alpha=0.5)
        plt.plot([-5,5],[-5,5])
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xlabel("eta_true")
        plt.ylabel("eta_pred")
        plt.savefig(path + "eta_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        plt.scatter(
            data.y_candidates[msk, 3].detach().cpu().numpy(),
            output[msk, 3].detach().cpu().numpy(),
            marker=".", alpha=0.5)
        plt.plot([-5,5],[-5,5])
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xlabel("phi_true")
        plt.ylabel("phi_pred")
        plt.savefig(path + "phi_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
    
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(0,10,40)
        plt.hist(data.y_candidates[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(output[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.ylabel("pt")
        plt.savefig(path + "pt.pdf")
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(-5,5,40)
        plt.hist(data.y_candidates[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(output[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.ylabel("eta")
        plt.savefig(path + "eta.pdf")
        del fig
        plt.clf()
        
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(-5,5,40)
        plt.hist(data.y_candidates[msk, 3].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(output[msk, 3].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.ylabel("phi")
        plt.savefig(path + "phi.pdf")
        del fig
        plt.clf()
        plt.close("all")
    plt.close("all")

if __name__ == "__main__":
    full_dataset = PFGraphDataset(root='data/TTbar_run3')
    full_dataset.raw_dir = "data/TTbar_run3"
    full_dataset.processed_dir = "data/TTbar_run3/processed_jd"
  
    model_name = sys.argv[1]
 
    input_dim = 8
    edge_dim = 1
    
    batch_size = 20
    n_train = 100
    n_epochs = 1000
    lr = 1e-4
    hidden_dim = 64
    patience = 200

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=2*n_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    if model_name == "simple":
        model = PFNet4(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    elif model_name == "gnn":
        model = PFNet6(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss = torch.nn.MSELoss()
    loss2 = torch.nn.BCELoss()
    
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)
    
    model.train()
    
    losses1 = []
    losses2 = []
    corrs = []
    losses1_t = []
    losses2_t = []
    corrs_t = []
    best_test_loss = 99999.9
    stale_epochs = 0
   
    t0_initial = time.time() 
    for j in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            break

        model.train()
        losses, c = train(model, train_loader, batch_size, j, optimizer)
        l = sum(list(losses))
        losses1 += [l]    
        corrs += [c]

        model.eval()
        losses_t, c_t = train(model, test_loader, batch_size, j, None)
        l_t = sum(list(losses_t))
        losses1_t += [l_t]
        corrs_t += [c_t]
        
        if l_t < best_test_loss:
            best_test_loss = l_t 
            stale_epochs = 0
        else:
            stale_epochs += 1
        if j > 0 and j%20 == 0:
            make_plots("data/{0}/epoch_{1}/".format(model_name, j), losses1, losses1_t, corrs, corrs_t, test_loader)
        
        t1 = time.time()
        eta = (n_epochs - j)*((t1 - t0_initial)/(j + 1)) / 60.0
        print("epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.5f}/{:.5f} l1={:.2f} l2={:.2f} l3={:.2f} stale={} eta={:.1f}m".format(
            j, n_epochs, t1 - t0, l, l_t, c, c_t, losses_t[0], losses_t[1], losses_t[2], stale_epochs, eta))
    make_plots("data/{0}/epoch_{1}/".format(model_name, j), losses1, losses1_t, corrs, corrs_t, test_loader)


