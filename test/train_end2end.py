import sys
import os

from comet_ml import Experiment

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
except Exception as e:
    print("Could not import setGPU, running CPU-only")

multi_gpu = False

import torch
print("torch", torch.__version__)
import torch_geometric
print("torch_geometric", torch_geometric.__version__)

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

from sklearn.metrics import accuracy_score

import graph_data
from graph_data import PFGraphDataset, elem_to_id, class_to_id, class_labels
from plot_utils import plot_confusion_matrix

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

device = torch.device('cuda')
#device = torch.device("cuda:0")

def prepare_dataframe(model, loader):
    model.eval()
    dfs = []
    eval_time = 0
    #for i, data in tqdm.tqdm(enumerate(loader),total=len(loader)):
    for i, data in enumerate(loader):
        if not multi_gpu:
            data = data.to(device)

        pred_id_onehot, pred_momentum = model(data)
        _, pred_id = torch.max(pred_id_onehot, -1)
        pred_momentum[pred_id==0] = 0
        if not multi_gpu:
            data = [data]

        x = torch.cat([d.x.to("cpu") for d in data])
        gen_id = torch.cat([d.y_gen_id.to("cpu") for d in data])
        gen_p4 = torch.cat([d.ygen[:, :4].to("cpu") for d in data])
        cand_id = torch.cat([d.y_candidates_id.to("cpu") for d in data])
        cand_p4 = torch.cat([d.ycand[:, :4].to("cpu") for d in data])

        df = pandas.DataFrame()

        df["elem_type"] = [int(graph_data.elem_labels[i]) for i in torch.argmax(x[:, :len(graph_data.elem_labels)], axis=-1).numpy()]
        df["gen_pid"] = [int(graph_data.class_labels[i]) for i in gen_id.numpy()]
        df["gen_eta"] = gen_p4[:, 0].numpy()
        df["gen_phi"] = gen_p4[:, 1].numpy()
        df["gen_e"] = gen_p4[:, 2].numpy()
        df["gen_charge"] = gen_p4[:, 3].numpy()

        df["cand_pid"] = [int(graph_data.class_labels[i]) for i in cand_id.numpy()]
        df["cand_eta"] = cand_p4[:, 0].numpy()
        df["cand_phi"] = cand_p4[:, 1].numpy()
        df["cand_e"] = cand_p4[:, 2].numpy()
        df["cand_charge"] = cand_p4[:, 3].numpy()
        df["pred_pid"] = [int(graph_data.class_labels[i]) for i in pred_id.detach().cpu().numpy()]

        df["pred_eta"] = pred_momentum[:, 0].detach().cpu().numpy()
        df["pred_phi"] = pred_momentum[:, 1].detach().cpu().numpy()
        df["pred_e"] = pred_momentum[:, 2].detach().cpu().numpy()
        df["pred_charge"] = pred_momentum[:, 3].detach().cpu().numpy()

        dfs += [df]

    df = pandas.concat(dfs, ignore_index=True)
 
    #Print some stats for each target particle type 
    #for pid in [211, -211, 130, 22, -11, 11, 13, -13, 1, 2]:
    #    msk_gen = df["gen_pid"] == pid
    #    msk_pred = df["pred_pid"] == pid

    #    npred = int(np.sum(msk_pred))
    #    ngen = int(np.sum(msk_gen))
    #    tpr = np.sum(msk_gen & msk_pred) / npred
    #    fpr = np.sum(~msk_gen & msk_pred) / npred
    #    eff = np.sum(msk_gen & msk_pred) / ngen

    #    mu = 0.0
    #    sigma = 0.0
    #    if np.sum(msk_pred) > 0:
    #        pts = df[msk_gen & msk_pred][["gen_pt", "pred_pt"]].values
    #        r = pts[:, 1]/pts[:, 0]
    #        mu, sigma = np.mean(r), np.std(r)
    #    print("pid={pid} Ngen={ngen} Npred={npred} eff={eff:.4f} tpr={tpr:.4f} fpr={fpr:.4f} pt_mu={pt_mu:.4f} pt_s={pt_s:.4f}".format(
    #        pid=pid, ngen=ngen, npred=npred, eff=eff, tpr=tpr, fpr=fpr, pt_mu=mu, pt_s=sigma
    #    ))
    #sumpt_cand = df[df["cand_pid"]!=0]["cand_pt"].sum()/len(dfs)
    #sumpt_gen = df[df["gen_pid"]!=0]["gen_pt"].sum()/len(dfs)
    #sumpt_pred = df[df["pred_pid"]!=0]["pred_pt"].sum()/len(dfs)
    #print("sumpt_cand={:.2f} sumpt_gen={:.2f} sumpt_pred={:.2f}".format(sumpt_cand, sumpt_gen, sumpt_pred))
 
    return df

def get_model_fname(dataset, model, n_train, lr, target_type):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_{}__npar_{}__cfg_{}__user_{}__ntrain_{}__lr_{}__{}'.format(
        model_name,
        dataset.split("/")[-1],
        target_type,
        model_params,
        model_cfghash,
        model_user,
        n_train,
        lr, int(time.time()))
    return model_fname


#Dense GCN
class PFNet5(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, embedding_dim=32, output_dim_id=len(class_to_id), output_dim_p4=4, dropout_rate=0.5, convlayer="sgconv", space_dim=2, nearest=3):
        super(PFNet5, self).__init__()
        self.input_dim = input_dim
        act = nn.LeakyReLU
        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, embedding_dim),
            act(),
        )
        self.conv = DenseGCNConv(embedding_dim, embedding_dim)

        self.nn1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_id),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(embedding_dim + output_dim_id, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )

    def forward(self, data):
        x = data.x
        batch = data.batch

        x = self.inp(x)

        xdense, mask = torch_geometric.utils.to_dense_batch(x, data.batch)
        print(xdense.shape)
        adj_dense = torch_geometric.utils.to_dense_adj(data.edge_index, data.batch, data.edge_attr)
        inds = torch.triu_indices(adj_dense.shape[1], adj_dense.shape[1], device=device)
       
        adj_dense[:, inds[0], inds[1]] += torch.sqrt((xdense[:, inds[0], 0] - xdense[:, inds[1], 0])**2 + (xdense[:, inds[0], 1] - xdense[:, inds[1], 1])**2)

        x = self.conv(xdense, adj_dense, mask)[mask]

        cand_ids = self.nn1(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn2(torch.cat([cand_ids, x], axis=-1))

        return torch.nn.functional.sigmoid(data.edge_attr), cand_ids, cand_p4


#Baseline model with graph attention convolution & edge classification, slow to train 
class PFNet6(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, embedding_dim=32, output_dim_id=len(class_to_id), output_dim_p4=4, dropout_rate=0.5, convlayer="sgconv", space_dim=2, nearest=3):
        super(PFNet6, self).__init__()

        act = nn.SELU

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, embedding_dim),
            act(),
        )
        if convlayer == "sgconv":
            self.conv1 = SGConv(embedding_dim, embedding_dim, K=2)
        elif convlayer == "gatconv":
            self.conv1 = GATConv(embedding_dim, embedding_dim, heads=1, concat=False)

        #pairs of embedded nodes + edge
        self.num_node_features_edgecls = 2
        self.edgenet = nn.Sequential(
            nn.Linear(2*self.num_node_features_edgecls + 1, 32),
            act(),
            nn.Linear(32, 32),
            act(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        if convlayer == "sgconv":
            self.conv2 = SGConv(embedding_dim, embedding_dim, K=2)
        elif convlayer == "gatconv":
            self.conv2 = GATConv(embedding_dim, embedding_dim, heads=1, concat=False)

        self.nn1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_id),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(embedding_dim + output_dim_id, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def forward(self, data):
        batch = data.batch

        #encode the inputs
        #x = self.inp(data.x)
        x = data.x
        edge_index = data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = data.edge_attr.squeeze(-1)

        #embed the nodes
        x = self.inp(x)
 
        #Run a graph convolution on the embedded nodes
        x = torch.nn.functional.selu(self.conv1(x, data.edge_index))

        #Compute new edge weights based on embedded node pairs
        xpairs = torch.cat([x[edge_index[0, :], :self.num_node_features_edgecls], x[edge_index[1, :], :self.num_node_features_edgecls], edge_weight.unsqueeze(-1)], axis=-1)
        edge_weight2 = self.edgenet(xpairs).squeeze(-1)
        edge_mask = edge_weight2 > 0.5
        row, col = data.edge_index
        row2, col2 = row[edge_mask], col[edge_mask]

        #Run a second convolution with the new edges
        x = torch.nn.functional.selu(self.conv2(x, torch.stack([row2, col2])))

        #Final candidate inference
        cand_ids = self.nn1(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn2(torch.cat([cand_ids, x], axis=-1))

        return edge_weight2, cand_ids, cand_p4

#Simplified model with SGConv, no edge classification, fast to train
class PFNet7(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim_id=len(class_to_id), output_dim_p4=4, convlayer="gravnet-knn", space_dim=2, nearest=3, dropout_rate=0.0):
        super(PFNet7, self).__init__()

        act = nn.LeakyReLU
        self.convlayer = convlayer

        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
        )
        #self.conv0 = SGConv(hidden_dim, hidden_dim, K=2)

        if convlayer == "gravnet-knn":
            self.conv1 = GravNetConv(hidden_dim, hidden_dim, space_dim, hidden_dim, nearest, neighbor_algo="knn") 
        elif convlayer == "gravnet-radius":
            self.conv1 = GravNetConv(hidden_dim, hidden_dim, space_dim, hidden_dim, nearest, neighbor_algo="radius") 
        elif convlayer == "sgconv":
            self.conv1 = SGConv(hidden_dim, hidden_dim, K=3)
        elif convlayer == "gatconv":
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, output_dim_id),
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_dim + output_dim_id, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        #print("forward", data.batch.device, len(torch.unique(data.batch)))
        #edge_weight = data.edge_attr.squeeze(-1)
        #edge_index = data.edge_index
        
        x = self.nn1(data.x)
        #x = torch.nn.functional.leaky_relu(self.conv0(x, edge_index))
        #x = data.x
        
        #Run a convolution
        new_edge_index, x = self.conv1(x)
        x = torch.nn.functional.leaky_relu(x)
        
        #Decode convolved graph nodes to pdgid and p4
        cand_ids = self.nn2(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn3(torch.cat([x, cand_ids], axis=-1))

        return cand_ids, cand_p4

class PFNet8(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, embedding_dim=64, output_dim_id=len(class_to_id), output_dim_p4=4, dropout_rate=0.5, convlayer="sgconv", space_dim=2, nearest=3):
        super(PFNet8, self).__init__()

        act = nn.SELU

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.unet = GraphUNet(embedding_dim, hidden_dim, embedding_dim, 3, pool_ratios=0.2, act=torch.nn.functional.selu)

        self.nn1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_id),
            act(),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(embedding_dim + output_dim_id, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, output_dim_p4),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def forward(self, data):
        batch = data.batch

        #encode the inputs
        #x = self.inp(data.x)
        x = data.x
        edge_index = data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = data.edge_attr.squeeze(-1)

        x = self.inp(x)
        x = torch.nn.functional.selu(self.unet(x, edge_index, batch))
 
        #Final candidate inference
        cand_ids = self.nn1(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn2(torch.cat([cand_ids, x], axis=-1))

        return torch.sigmoid(edge_weight), cand_ids, cand_p4


model_classes = {
    "PFNet5": PFNet5,
    "PFNet6": PFNet6,
    "PFNet7": PFNet7,
    "PFNet8": PFNet8,
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=80, help="number of training events")
    parser.add_argument("--n_val", type=int, default=20, help="number of validation events")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="patience before early stopping")
    parser.add_argument("--n_plot", type=int, default=10, help="make plots every iterations")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--outpath", type=str, default = 'data/', help="Output folder")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=1.0, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering classification")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet")
    parser.add_argument("--space_dim", type=int, default=2, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--nearest", type=int, default=3, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--overwrite", action='store_true', help="overwrite if model output exists")
    parser.add_argument("--disable-comet", action='store_true', help="disable comet-ml")
    args = parser.parse_args()
    return args

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type):
    with torch.no_grad(): 
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type)
    return ret

def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    losses = np.zeros((len(loader), 3))
    corrs_batch = np.zeros(len(loader))
    accuracies_batch = np.zeros(len(loader))

    num_samples = 0
    for i, data in enumerate(loader):
        t0 = time.time()
        num_samples += len(data)
        
        if not multi_gpu:
            data = data.to(device)

        if is_train:
            optimizer.zero_grad()

        #print("Calling model with N={}".format(len(data)))
        cand_id_onehot, cand_momentum = model(data)
        _dev = cand_id_onehot.device
        _, indices = torch.max(cand_id_onehot, -1)
        if not multi_gpu:
            data = [data]
        
        if args.target == "gen":
            target_ids = torch.cat([d.y_gen_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ygen[:, :4] for d in data]).to(_dev)
        elif args.target == "cand":
            target_ids = torch.cat([d.y_candidates_id for d in data]).to(_dev)
            target_p4 = torch.cat([d.ycand[:, :4] for d in data]).to(_dev)

        vs, cs = torch.unique(target_ids, return_counts=True)
        weights = torch.zeros(len(class_to_id)).to(device=_dev)
        for k, v in zip(vs, cs):
            weights[k] = 1.0/float(v)


        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        #msk = (indices != 0)
        msk = ((indices != 0) & (target_ids != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids))

        accuracies_batch[i] = accuracy_score(target_ids[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        #Loss for output candidate id (multiclass)
        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, target_ids, weight=weights)
        else:
            l1 = torch.tensor(0.0).to(device=_dev)

        #Loss for candidate p4 properties (regression)
        l2 = torch.tensor(0.0).to(device=_dev)
        if l2m > 0.0:
            #l2 = l2m*torch.nn.functional.mse_loss(cand_momentum, target[1])
            #modular loss for phi, seems to consume more memory
            l2 = l2m*torch.nn.functional.mse_loss(cand_momentum[msk2], target_p4[msk2])

        batch_loss = l1 + l2
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()
        
        if is_train:
            batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t1 = time.time()
        print('{}/{} batch_loss={:.2f} dt={:.1f}s'.format(i, len(loader), batch_loss_item, t1-t0), end='\r', flush=True)
        if is_train:
            optimizer.step()

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0 
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_momentum[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        i += 1

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.sum(axis=0)
    return num_samples, losses, corr, acc

def make_plots(model, n_epoch, path, losses_train, losses_val, corrs_train, corrs_val, accuracies, accuracies_v, val_loader):
    try:
        os.makedirs(path)
    except Exception as e:
        pass

    modpath = path + 'weights.pth'
    torch.save(model.state_dict(), modpath)

    df = prepare_dataframe(model, val_loader)
    df.to_pickle(path + "df.pkl.bz2")

    np.savetxt(path+"losses_train.txt", losses_train)
    np.savetxt(path+"losses_val.txt", losses_val)
    np.savetxt(path+"corrs_train.txt", corrs_train)
    np.savetxt(path+"corrs_val.txt", corrs_val)
    np.savetxt(path+"accuracies_train.txt", accuracies)
    np.savetxt(path+"accuracies_val.txt", accuracies_v)

#    for i in range(losses_train.shape[1]):
#        fig = plt.figure(figsize=(5,5))
#        ax = plt.axes()
#        plt.ylabel("train loss")
#        plt.plot(losses_train[:n_epoch, i])
#        ax2=ax.twinx()
#        ax2.plot(losses_val[:n_epoch, i], color="orange")
#        ax2.set_ylabel("val loss", color="orange")
#        plt.xlabel("epoch")
#        plt.tight_layout()
#        plt.savefig(path + "loss_{0}.pdf".format(i))
#        del fig
#        plt.clf()

if __name__ == "__main__":
    args = parse_args()
    
    full_dataset = PFGraphDataset(args.dataset)

    #one-hot encoded element ID + element parameters
    input_dim = 26

    #one-hot particle ID and momentum
    output_dim_id = len(class_to_id)
    output_dim_p4 = 4

    edge_dim = 1

    patience = args.patience

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=args.n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.n_train, stop=args.n_train+args.n_val))
    print("train_dataset", len(train_dataset))
    print("val_dataset", len(val_dataset))


    if not multi_gpu:
        def collate(items):
            l = sum(items, [])
            return Batch.from_data_list(l)
    else:
        def collate(items):
            l = sum(items, [])
            return l

    train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    train_loader.collate_fn = collate
    val_loader = DataListLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    val_loader.collate_fn = collate

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'convlayer': args.convlayer,
                    'space_dim': args.space_dim,
                    'nearest': args.nearest}

    # need your api key in a .comet.config file: see https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
    experiment = Experiment(project_name="particeflow", disabled=args.disable_comet)
    experiment.log_parameters(dict(model_kwargs, **{'model': args.model, 'lr':args.lr,
                                                    'l1': args.l1, 'l2':args.l2, 'l3':args.l3,
                                                    'n_train':args.n_train, 'target':args.target}))
                    
    model = model_class(**model_kwargs)
        
    if multi_gpu:
        model = torch_geometric.nn.DataParallel(model)

    model.to(device)

    model_fname = get_model_fname(args.dataset, model, args.n_train, args.lr, args.target)
    outpath = osp.join(args.outpath, model_fname)
    if osp.isdir(outpath):
        if args.overwrite:
            print("model output {} already exists, deleting it".format(outpath))
            import shutil
            shutil.rmtree(outpath)
        else:
            print("model output {} already exists, please delete it".format(outpath))
            sys.exit(0)
    try:
        os.makedirs(outpath)
    except Exception as e:
        pass

    with open('{}/model_kwargs.pkl'.format(outpath), 'wb') as f:
        pickle.dump(model_kwargs, f,  protocol=pickle.HIGHEST_PROTOCOL)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = torch.nn.MSELoss()
    loss2 = torch.nn.BCELoss()
    
    print(model)
    print(model_fname)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)
    
    model.train()
    
    losses_train = np.zeros((args.n_epochs+1, 3))
    losses_val = np.zeros((args.n_epochs+1, 3))

    corrs = []
    corrs_v = []
    accuracies = []
    accuracies_v = []
    best_val_loss = 99999.9
    stale_epochs = 0

    initial_epochs = 10
   
    t0_initial = time.time()
    print("Training over {} epochs".format(args.n_epochs))
    for j in range(args.n_epochs + 1):
        t0 = time.time()
        
        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break
        with experiment.train():
            model.train()
            num_samples_train, losses, c, acc = train(model, train_loader, j, optimizer, args.l1, args.l2, args.l3, args.target)
            l = sum(losses)
            losses_train[j] = losses
            corrs += [c]
            accuracies += [acc]
            experiment.log_metric('loss',l, step=j)
            experiment.log_metric('loss1',losses[0], step=j)
            experiment.log_metric('loss2',losses[1], step=j)
            experiment.log_metric('loss3',losses[2], step=j)
            experiment.log_metric('corrs',c, step=j)
            experiment.log_metric('accuracy',acc, step=j)
            
        with experiment.validate():
            model.eval()
            num_samples_val, losses_v, c_v, acc_v = test(model, val_loader, j, args.l1, args.l2, args.l3, args.target)
            l_v = sum(losses_v)
            losses_val[j] = losses_v
            corrs_v += [c_v]
            accuracies_v += [acc_v]
            experiment.log_metric('loss',l_v, step=j)
            experiment.log_metric('loss1',losses_v[0], step=j)
            experiment.log_metric('loss2',losses_v[1], step=j)
            experiment.log_metric('loss3',losses_v[2], step=j)
            experiment.log_metric('corrs',c_v, step=j)
            experiment.log_metric('accuracy',acc_v, step=j)
            
        if l_v < best_val_loss:
            best_val_loss = l_v
            stale_epochs = 0
            make_plots(
                model, j, "{0}/epoch_{1}/".format(outpath, "best"),
                losses_train, losses_val, corrs, corrs_v,
                accuracies, accuracies_v, val_loader)
        else:
            stale_epochs += 1

        if j > 0 and j%args.n_plot == 0:
            make_plots(
                model, j, "{0}/epoch_{1}/".format(outpath, j),
                losses_train, losses_val, corrs, corrs_v,
                accuracies, accuracies_v, val_loader)
 
        t1 = time.time()
        epochs_remaining = args.n_epochs - j
        time_per_epoch = (t1 - t0_initial)/(j + 1) 
        eta = epochs_remaining*time_per_epoch/60

        spd = (num_samples_val+num_samples_train)/time_per_epoch
        losses_str = "[" + ",".join(["{:.4f}".format(x) for x in losses_v]) + "]"
        print("epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.2f}/{:.2f} a={:.2f}/{:.2f} partial_losses={} stale={} eta={:.1f}m spd={:.2f} samples/s".format(
            j, args.n_epochs,
            t1 - t0, l, l_v, c, c_v, acc, acc_v,
            losses_str, stale_epochs, eta, spd))

    make_plots(
        model, j, "{0}/epoch_{1}/".format(outpath, "last"),
        losses_train, losses_val, corrs, corrs_v,
        accuracies, accuracies_v, val_loader)
