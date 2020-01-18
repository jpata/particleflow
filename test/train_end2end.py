import setGPU

import torch
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet
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
import termplotlib as tpl

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
            nn.Linear(input_dim*10000, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim*10000),
        )

    def forward(self, data):
        num_nodes = data.x.shape[0]
        x = torch.nn.functional.pad(data.x, (0, 0, 0, 10000-num_nodes))
        x = x.reshape((self.input_dim*10000))
        x = self.inputnet(x)
        x = x.reshape((10000,self.output_dim))[:num_nodes]
        r1 = torch.sigmoid(x[:, 0])
        x[:, 0] = r1
        return x

#Based on GraphUNet
class PFNet5(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet5, self).__init__()
        self.unet = GraphUNet(input_dim, hidden_dim, output_dim,
                              depth=3, pool_ratios=0.2)
        self.batchnorm1 = nn.BatchNorm1d(1)

    def forward(self, data):
        r = self.unet(data.x, data.edge_index)
        r1 = torch.sigmoid(self.batchnorm1(r[:, :1]))[:, 0]
        r[:, 0] = r1
        return r

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
   
    #Divide by number of elements in cluster 
    for i in range(jagged_sum_pred.shape[1]):
        jagged_sum_pred[:, i] /= counts
        jagged_sum_tgt[:, i] /= counts
  
    #Compare aggregated quantities cluster by cluster 
    loss = torch.nn.functional.binary_cross_entropy(torch.clamp(jagged_sum_pred[:, 0], 0, 1), torch.clamp(jagged_sum_tgt[:, 0], 0, 1))
    loss += torch.nn.functional.mse_loss(jagged_sum_pred[:, 1:], jagged_sum_tgt[:, 1:]) 
    #Compare losses cluster by cluster 
    #loss = torch.nn.functional.binary_cross_entropy(y_pred[:, 0], y_true[:, 0])
    #loss += torch.nn.functional.mse_loss(y_pred[:, 1:], y_true[:, 1:])
    return (jagged_sum_pred, jagged_sum_tgt, loss)
 
def train(model, loader, batch_size, epoch, optimizer):
    losses_batch1 = []
    losses_batch2 = []
    corrs_batch = []
    
    num_pred = []
    num_true = []
   
    is_train = not (optimizer is None)

    for i, data in enumerate(loader):
        data = data.to(device)
        if is_train:
            optimizer.zero_grad()
        output = model(data)

        data.y_candidates[:, 0] = (data.y_candidates[:, 0]!=0).to(dtype=torch.float)
        pooled_p, pooled_t, batch_loss = loss_by_cluster(output, data.y_candidates, data.batch, data.block_ids)
        if is_train and i == 0:
            print(output[:5])
            print(data.y_candidates[:5])
            print(data.block_ids[:5])
            print(pooled_p[:5])
            print(pooled_t[:5])
        
        if is_train:
            batch_loss.backward()
        batch_loss_item = batch_loss.item()
        if is_train:
            optimizer.step()
   
        corr_pt = np.corrcoef(
            pooled_p[:, 1].detach().cpu().numpy(),
            pooled_t[:, 1].detach().cpu().numpy())[0,1]
        corrs_batch += [corr_pt]
        losses_batch1 += [batch_loss_item]
    
    l1 = np.mean(losses_batch1)
    corr = np.mean(corrs_batch)
    return l1, corr

if __name__ == "__main__":
    full_dataset = PFGraphDataset(root='data/TTbar_run3')
    full_dataset.raw_dir = "data/TTbar_run3"
    full_dataset.processed_dir = "data/TTbar_run3/processed_jd"
    
    data = full_dataset.get(0)
    input_dim = data.x.shape[1]
    edge_dim = 1
    
    batch_size = 20
    n_epochs = 1000
    n_train = 1
    lr = 2*1e-3
    hidden_dim = 128
    n_iters = 3
    patience = n_epochs

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=2*n_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    model = PFNet2(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    #model = PFNet3(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    #model = PFNet4(input_dim=input_dim, hidden_dim=1024).to(device)
    #model = PFNet5(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
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
    
    for j in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            break

        model.train()
        l1, l2, c = train(model, train_loader, batch_size, j, optimizer)
        losses1 += [l1]    
        losses2 += [l2]    
        corrs += [c]

        model.eval()
        l1_t, l2_t, c_t = train(model, test_loader, batch_size, j, None)
        losses1_t += [l1_t]    
        losses2_t += [l2_t]    
        corrs_t += [c_t]
        l = l1_t + l2_t
        
        print(chr(27)+'[2j')
        print('\033c')
        print('\x1bc')

        if l < best_test_loss:
            best_test_loss = l 
            stale_epochs = 0
        else:
            print('Stale epoch {} {:.5f}/{:.5f}'.format(stale_epochs, l, best_test_loss))
            stale_epochs += 1
        t1 = time.time()
        print("epoch={}/{} dt={:.2f}s l1={:.5f}/{:.5f} l2={:.5f}/{:.5f} c={:.5f}/{:.5f}".format(
            j, n_epochs, t1 - t0, l1, l1_t, l2, l2_t, c, c_t))
        
        if len(corrs_t) > 0:
            #fig = tpl.figure()
            #fig.plot(range(len(corrs_t)), corrs_t, width=80, height=15, ylim=(0,1))
            #fig.show()
            fig = tpl.figure()
            fig.plot(range(len(losses1)), losses1, width=80, height=15, ylim=(0,3))
            fig.show()
  
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(5,5))
    plt.plot(losses1)
    plt.plot(losses1_t)
    plt.ylim(0,2*losses1[-1])
    plt.savefig("loss1.pdf")

    plt.figure(figsize=(5,5))
    plt.plot(losses2)
    plt.plot(losses2_t)
    plt.ylim(0,2*losses2[-1])
    plt.savefig("loss2.pdf")
    
    plt.figure(figsize=(5,5))
    plt.plot(corrs)
    plt.plot(corrs_t)
    plt.savefig("corr.pdf")
    
    #d = data.to(device=device)
    #is_pred, output = model(d)
    #
    #plt.figure(figsize=(5,5))
    #msk = is_pred > 0.9
    #plt.scatter(
    #    data.y[msk][:, 0].detach().cpu().numpy(),
    #    output[msk][:, 0].detach().cpu().numpy(),
    #    marker=".", alpha=0.5)
    #plt.plot([0,5],[0,5])
    #plt.xlim(0,5)
    #plt.ylim(0,5)
    #plt.savefig("pt_corr.pdf")
    #
    #plt.figure(figsize=(5,5))
    #plt.scatter(
    #    data.y[msk, 1].detach().cpu().numpy(),
    #    output[msk, 1].detach().cpu().numpy(),
    #    marker=".", alpha=0.5)
    #plt.plot([-5,5],[-5,5])
    #plt.xlim(-5,5)
    #plt.ylim(-5,5)
    #plt.savefig("eta_corr.pdf")
    #
    #plt.figure(figsize=(5,5))
    #plt.scatter(
    #    data.y[msk, 2].detach().cpu().numpy(),
    #    output[msk, 2].detach().cpu().numpy(),
    #    marker=".", alpha=0.5)
    #plt.plot([-5,5],[-5,5])
    #plt.xlim(-5,5)
    #plt.ylim(-5,5)
    #plt.savefig("phi_corr.pdf")
    #
    #plt.figure(figsize=(5,5))
    #b = np.linspace(0,10,40)
    #plt.hist(data.y[msk, 0].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.hist(output[msk, 0].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.savefig("pt.pdf")
    #
    #plt.figure(figsize=(5,5))
    #b = np.linspace(-5,5,40)
    #plt.hist(data.y[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.hist(output[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.savefig("eta.pdf")
    #
    #plt.figure(figsize=(5,5))
    #b = np.linspace(-5,5,40)
    #plt.hist(data.y[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.hist(output[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
    #plt.savefig("phi.pdf")
