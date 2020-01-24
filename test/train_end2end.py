use_gpu = False
try:
    import setGPU
    use_gpu = True
except Exception as e:
    print("GPU not available")

import torch
print("torch", torch.__version__)
import torch_geometric
print("torch_geometric", torch_geometric.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, MessagePassing, EdgePooling, GATConv, GCNConv, JumpingKnowledge, GraphUNet, DynamicEdgeConv
from torch_geometric.nn import TopKPooling, SAGPooling, SGConv
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
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from graph_data import PFGraphDataset

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

device = torch.device('cuda' if use_gpu else 'cpu')

#all candidate pdg-ids (multiclass labels)
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

#detector element labels
elem_labels = [ 1.,  2.,  3.,  4.,  5.,  8.,  9., 11.]

#map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}

#map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}

#Data normalization constants for faster convergence.
#These are just estimated with a printout and rounding, don't need to be super accurate
x_means = torch.tensor([ 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
x_stds = torch.tensor([ 1.0, 22.0,  2.6,  1.8,  1.3,  1.9,  1.3,  1.0]).to(device)
y_candidates_means = torch.tensor([0.0, 0.0, 0.0]).to(device)
y_candidates_stds = torch.tensor([1.8, 2.0, 1.5]).to(device)

def get_model_fname(model, n_train, lr):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']
    
    model_fname = '{}__npar_{}__cfg_{}__user_{}__ntrain_{}__lr_{}'.format(
        model_name, model_params,
        model_cfghash, model_user, n_train, lr)
    return model_fname

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlim(-1, len(target_names))
    plt.ylim(-1, len(target_names))
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()

#Baseline dense models
class PFDenseNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFDenseNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputnet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim),
        )
        self.edgenet = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        r = self.inputnet(data.x)
        edges = self.edgenet(data.edge_attr).squeeze(-1)

        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = r[:, n_onehot:]
        return edges, cand_ids, cand_p4

class PFDenseAllToAllNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFDenseAllToAllNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_elements = 10000
        self.inputnet = nn.Sequential(
            nn.Linear(input_dim*self.max_elements, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim*self.max_elements),
        )
        self.edgenet = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        x = data.x
        xb, batch_mask = torch_geometric.utils.to_dense_batch(x, data.batch)
        n_elem = xb.shape[1]
        assert(self.max_elements > n_elem)
        n_batch = xb.shape[0]
        xb = torch.nn.functional.pad(xb, (0, 0, 0, self.max_elements - n_elem, 0, 0))
        xb = torch.reshape(xb, (n_batch, input_dim*self.max_elements))
        r = self.inputnet(xb)
        r = torch.reshape(r, (n_batch, self.max_elements, output_dim))
        r = r[:, :n_elem, :]
        r = r[batch_mask]
        edges = self.edgenet(data.edge_attr).squeeze(-1)

        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = r[:, n_onehot:]
        return edges, cand_ids, cand_p4

#Based on GraphUNet
class PFGraphUNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFGraphUNet, self).__init__()
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

#Predict only particle id
class PFNetOnlyID(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=None):
        super(PFNetOnlyID, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=False)
        self.nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        edge_weight = data.edge_attr.squeeze(-1)
        
        x = torch.nn.functional.leaky_relu(self.conv1(data.x, data.edge_index))
        r = self.nn1(x)
        
        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = torch.zeros((data.x.shape[0], 3)).to(device=device)
        return edge_weight, cand_ids, cand_p4

class PFNet6(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet6, self).__init__()
   
        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=8, concat=False) 
        
        #pairs of nodes + edge
        self.edgenet = nn.Sequential(
            nn.Linear(2*hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
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
        x = torch.nn.functional.leaky_relu(self.conv1(x, data.edge_index))
        
        #Compute new edge weights based on embedded node pairs
        xpairs = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_weight.unsqueeze(-1)], axis=-1)
        edge_weight = self.edgenet(xpairs).squeeze(-1)
        
        #Run a second convolution with the new edges
        x = torch.nn.functional.leaky_relu(self.conv2(x, data.edge_index, edge_weight))
        
        up = x

        up = torch.cat([data.x, up], axis=-1)

        #Final candidate inference
        r = self.nn1(up)

        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = r[:, n_onehot:]
        return edge_weight, cand_ids, cand_p4

class PFNet7(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super(PFNet7, self).__init__()
   
        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = SGConv(hidden_dim, hidden_dim, K=1) 
        self.edgenet = nn.Sequential(
            nn.Linear(2*hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = SGConv(hidden_dim, hidden_dim, K=1)
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
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
        x = torch.nn.functional.leaky_relu(self.conv1(x, data.edge_index, edge_weight))
        
        #Compute new edge weights based on embedded node pairs
        xpairs = torch.cat([x[edge_index[0]], x[edge_index[1]], edge_weight.unsqueeze(-1)], axis=-1)
        edge_weight = self.edgenet(xpairs).squeeze(-1)
        
        #Run a second convolution with the new edge weight
        x = torch.nn.functional.leaky_relu(self.conv2(x, data.edge_index, edge_weight))

        up = x

        up = torch.cat([data.x, up], axis=-1)

        r = self.nn1(up)
        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = r[:, n_onehot:]
        return edge_weight, cand_ids, cand_p4


model_classes = {
    "PFDenseNet": PFDenseNet,
    "PFDenseAllToAllNet": PFDenseAllToAllNet,
    "PFGraphUNet": PFGraphUNet,
    "PFNetOnlyID": PFNetOnlyID,
    "PFNet6": PFNet6,
    "PFNet7": PFNet7,
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=100, help="number of training events")
    parser.add_argument("--n_test", type=int, default=20, help="number of testing events")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=1.0, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering classification")
    args = parser.parse_args()
    return args


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

#Do any in-memory transformations to data
def data_prep(data):
    new_ids = torch.zeros_like(data.x[:, 0])
    for k, v in elem_to_id.items():
        m = data.x[:, 0] == v
        new_ids[m] = k
    data.x[:, 0] = new_ids

    #Convert pdg-ids to consecutive class labels
    new_ids = torch.zeros_like(data.y_candidates[:, 0])
    for k, v in class_to_id.items():
        m = data.y_candidates[:, 0] == v
        new_ids[m] = k
    data.y_candidates[:, 0] = new_ids
  
    #randomize the target order - then we should not be able to predict anything!
    #perm = torch.randperm(len(data.y_candidates))
    #data.y_candidates = data.y_candidates[perm]
    
    data.x -= x_means
    data.x /= x_stds

    #Create a one-hot encoded vector of the class labels
    id_onehot = torch.nn.functional.one_hot(data.y_candidates[:, 0].to(dtype=torch.long), num_classes=len(class_to_id))

    #one-hot encode the input categorical
    elem_id_onehot = torch.nn.functional.one_hot(data.x[:, 0].to(dtype=torch.long), num_classes=len(elem_to_id))
    data.x = torch.cat([elem_id_onehot.to(dtype=torch.float), data.x[:, 1:]], axis=-1)

    #Extract the ids and momenta
    data.y_candidates_id = data.y_candidates[:, 0].to(dtype=torch.long)
    data.y_candidates = data.y_candidates[:, 1:]
    #normalize and center the target momenta (roughly)
    data.y_candidates -= y_candidates_means
    data.y_candidates /= y_candidates_stds
    
    data.x[torch.isnan(data.x)] = 0.0
    data.y_candidates[torch.isnan(data.y_candidates)] = 0.0

    #Compute weights for candidate pdgids
    # data.y_candidates_id_weights = torch.zeros_like(data.y_candidates_id).to(dtype=torch.float)
    # uniqs, counts = torch.unique(data.y_candidates_id, return_counts=True)
    # data.y_candidates_id_weights_cls = torch.zeros(len(class_labels), dtype=torch.float)
    # for cls_id, num_cls in zip(uniqs, counts):
    #     data.y_candidates_id_weights_cls[cls_id] = 1.0 / float(num_cls)
    #     m = data.y_candidates_id == cls_id
    #     data.y_candidates_id_weights[m] = 1.0 / float(num_cls)

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

@torch.no_grad()
def test(model, loader, batch_size, epoch, l1m, l2m, l3m):
    return train(model, loader, batch_size, epoch, None, l1m, l2m, l3m)

def train(model, loader, batch_size, epoch, optimizer, l1m, l2m, l3m):

    is_train = not (optimizer is None)

    losses = np.zeros((len(loader), 3))
    corrs_batch = np.zeros(len(loader))
    accuracies_batch = np.zeros(len(loader))

    for i, data in enumerate(loader):
        data = data.to(device)
        data_prep(data)

        if is_train:
            optimizer.zero_grad()

        edges, cand_id_onehot, cand_momentum = model(data)
        _, indices = torch.max(cand_id_onehot, -1)

        accuracies_batch[i] = accuracy_score(data.y_candidates_id.detach().cpu().numpy(), indices.detach().cpu().numpy())

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        msk = (indices != 0) & (data.y_candidates_id != 0)

        #Loss for output candidate id (multiclass)
        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, data.y_candidates_id)
        else:
            l1 = torch.tensor(0.0).to(device=device)

        #Loss for candidate p4 properties (regression)
        if l2m > 0.0:
            l2 = l2m*torch.nn.functional.mse_loss(cand_momentum, data.y_candidates) / 10.0
        else:
            l2 = torch.tensor(0.0).to(device=device)

        #Loss for edges enabled/disabled in clustering (binary)
        if l3m > 0.0:
            l3 = l3m*torch.nn.functional.binary_cross_entropy(edges, data.y) * 2.0
        else:
            l3 = torch.tensor(0.0).to(device=device)

        batch_loss = l1 + l2 + l3
        losses[i, 0] = l1.item()
        losses[i, 1] = l2.item()
        losses[i, 2] = l3.item()
        
        if is_train:
            batch_loss.backward()
        batch_loss_item = batch_loss.item()

        if is_train:
            optimizer.step()

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0 
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_momentum[msk, 0].detach().cpu().numpy(),
                data.y_candidates[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.sum(axis=0)
    return losses, corr, acc

def make_plots(model, n_epoch, path, losses_train, losses_test, corrs_train, corrs_test, accuracies, accuracies_t, test_loader):
   
    try:
        os.makedirs(path)
    except Exception as e:
        pass

    modpath = path + model_fname + '.best.pth'
    torch.save(model.state_dict(), modpath)
    np.savetxt(path+"losses_train.txt", losses_train)
    np.savetxt(path+"losses_test.txt", losses_test)
    np.savetxt(path+"corrs_train.txt", corrs_train)
    np.savetxt(path+"corrs_test.txt", corrs_test)
    np.savetxt(path+"accuracies_train.txt", accuracies)
    np.savetxt(path+"accuracies_test.txt", accuracies_t)

    for i in range(losses_train.shape[1]):
        fig = plt.figure(figsize=(5,5))
        ax = plt.axes()
        plt.ylabel("train loss")
        plt.plot(losses_train[:n_epoch, i])
        ax2=ax.twinx()
        ax2.plot(losses_test[:n_epoch, i], color="orange")
        ax2.set_ylabel("test loss", color="orange")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig(path + "loss_{0}.pdf".format(i))
        del fig
        plt.clf()
 
    fig = plt.figure(figsize=(5,5))
    plt.plot(corrs_train)
    plt.plot(corrs_test)
    plt.xlabel("epoch")
    plt.ylabel("pt correlation")
    plt.tight_layout()
    plt.savefig(path + "corr.pdf")
    del fig
    plt.clf()

    fig = plt.figure(figsize=(5,5))
    plt.plot(accuracies)
    plt.plot(accuracies_t)
    plt.xlabel("epoch")
    plt.ylabel("classification accuracy")
    plt.tight_layout()
    plt.savefig(path + "acc.pdf")
    del fig
    plt.clf()
 
    #plot the first 5 batches from the test dataset
    num_preds = []
    num_trues = []
    cand_ids_true = []
    cand_ids_pred = []
    for i, data in enumerate(test_loader):
        d = data.to(device=device)
        data_prep(data)

        edges, cand_id_onehot, cand_momentum = model(d)
        #undo the normalization and centering
        #cand_momentum *= y_candidates_stds
        #cand_momentum += y_candidates_means

        _, cand_ids_pred_batch = torch.max(cand_id_onehot, -1)
        cand_momentum[cand_ids_pred_batch==0] = 0.0
        sumpt_pred = cand_momentum[:, 0].sum().detach().cpu().numpy()
        sumpt_true = data.y_candidates[:, 0].sum().detach().cpu().numpy()

        cand_ids_batched = torch_geometric.utils.to_dense_batch(cand_ids_pred_batch, batch=data.batch)
        num_pred = (cand_ids_batched[0]!=0).sum(axis=1) 
        num_true = (torch_geometric.utils.to_dense_batch(data.y_candidates_id, batch=data.batch)[0]!=0).sum(axis=1)
        num_preds += list(num_pred.cpu().numpy())
        num_trues += list(num_true.cpu().numpy())
        cand_ids_true += list(data.y_candidates_id.detach().cpu().numpy())
        cand_ids_pred += list(cand_ids_pred_batch.detach().cpu().numpy())
        
        msk = (cand_ids_pred_batch != 0) & (data.y_candidates_id != 0)
        if i>5:
            break

        #Get the first 1000 candidates in the batch
        inds2 = torch.nonzero(msk)
        perm = torch.randperm(len(inds2))
        inds2 = inds2[perm[:1000]]
        if len(inds2) == 0:
            break 

        fig = plt.figure(figsize=(5,5))
        v1 = data.y_candidates[inds2, 0].detach().cpu().numpy()[:, 0]
        v2 = cand_momentum[inds2, 0].detach().cpu().numpy()[:, 0]
        c = np.corrcoef(v1, v2)[0,1]
        plt.scatter(
            v1[:1000],
            v2[:1000],
            marker=".", alpha=0.5)
        plt.plot([0,5],[0,5])
        plt.xlim(0,5)
        plt.ylim(0,5)
        plt.xlabel("pt_true")
        plt.ylabel("pt_pred")
        plt.title("corr = {:.2f}".format(c))
        plt.tight_layout()
        plt.savefig(path + "pt_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        v1 = data.y_candidates[inds2, 1].detach().cpu().numpy()[:, 0]
        v2 = cand_momentum[inds2, 1].detach().cpu().numpy()[:, 0]
        c = np.corrcoef(v1, v2)[0,1]
        plt.scatter(
            v1[:1000],
            v2[:1000],
            marker=".", alpha=0.5)
        plt.plot([-5,5],[-5,5])
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xlabel("eta_true")
        plt.ylabel("eta_pred")
        plt.title("corr = {:.2f}".format(c))
        plt.tight_layout()
        plt.savefig(path + "eta_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        v1 = data.y_candidates[inds2, 2].detach().cpu().numpy()[:, 0]
        v2 = cand_momentum[inds2, 2].detach().cpu().numpy()[:, 0]
        c = np.corrcoef(v1, v2)[0,1]
        plt.scatter(
            v1[:1000],
            v2[:1000],
            marker=".", alpha=0.5)
        plt.plot([-5,5],[-5,5])
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.xlabel("phi_true")
        plt.ylabel("phi_pred")
        plt.title("corr = {:.2f}".format(c))
        plt.tight_layout()
        plt.savefig(path + "phi_corr_{0}.pdf".format(i))
        del fig
        plt.clf()
    
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(0,10,60)
        plt.hist(data.y_candidates[msk, 0].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(cand_momentum[msk, 0].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.xlabel("pt")
        plt.tight_layout()
        plt.savefig(path + "pt_{0}.pdf".format(i))
        del fig
        plt.clf()
 
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(-5,5,60)
        plt.hist(data.y_candidates[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(cand_momentum[msk, 1].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.xlabel("eta")
        plt.tight_layout()
        plt.savefig(path + "eta_{0}.pdf".format(i))
        del fig
        plt.clf()
        
        fig = plt.figure(figsize=(5,5))
        b = np.linspace(-5,5,60)
        plt.hist(data.y_candidates[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.hist(cand_momentum[msk, 2].detach().cpu().numpy(), bins=b, lw=2, histtype="step");
        plt.xlabel("phi")
        plt.tight_layout()
        plt.savefig(path + "phi_{0}.pdf".format(i))
        del fig
        plt.clf()
        plt.close("all")
   
    confusion = sklearn.metrics.confusion_matrix(
        cand_ids_true, cand_ids_pred,
        labels=range(len(class_labels)))
    np.savetxt(path+"confusion.txt", confusion)
    plot_confusion_matrix(cm = confusion, target_names=[int(x) for x in class_labels], normalize=False)
    plt.savefig(path + "confusion.pdf")
    plt.clf()

    fig = plt.figure(figsize=(5,5))
    c = np.corrcoef(num_preds, num_trues)[0,1]
    plt.scatter(
        num_trues,
        num_preds,
        marker=".", alpha=0.5)
    plt.plot([0,5000],[0,5000], color="black")
    plt.xlim(0,5000)
    plt.ylim(0,5000)
    plt.xlabel("num_true")
    plt.ylabel("num_pred")
    plt.title("corr = {:.2f}".format(c))
    plt.tight_layout()
    plt.savefig(path + "num_corr.pdf")
    del fig
    plt.clf()
    plt.close("all")

if __name__ == "__main__":
    full_dataset = PFGraphDataset(root='data/TTbar_run3')
    full_dataset.raw_dir = "data/TTbar_run3"
    full_dataset.processed_dir = "data/TTbar_run3/processed_jd2"

    args = parse_args()

    #one-hot encoded element ID + 7 element parameters (energy, eta, phi, track stuff)
    input_dim = 15

    #one-hot particle ID and 3 momentum components
    output_dim = len(class_to_id) + 3
    edge_dim = 1

    patience = args.n_epochs

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=args.n_train))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.n_train, stop=args.n_train+args.n_test))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    model_class = model_classes[args.model]
    model = model_class(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim).to(device)
    model_fname = get_model_fname(model, args.n_train, args.lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = torch.nn.MSELoss()
    loss2 = torch.nn.BCELoss()
    
    print(model)
    print(model_fname)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)
    
    model.train()
    
    losses_train = np.zeros((args.n_epochs, 3))
    losses_test = np.zeros((args.n_epochs, 3))

    corrs = []
    corrs_t = []
    accuracies = []
    accuracies_t = []
    best_test_loss = 99999.9
    stale_epochs = 0
   
    t0_initial = time.time() 
    for j in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            break

        model.train()
        losses, c, acc = train(model, train_loader, args.batch_size, j, optimizer, args.l1, args.l2, args.l3)
        l = sum(losses)
        losses_train[j] = losses
        corrs += [c]
        accuracies += [acc]

        model.eval()
        losses_t, c_t, acc_t = test(model, test_loader, args.batch_size, j, args.l1, args.l2, args.l3)
        l_t = sum(losses_t)
        losses_test[j] = losses_t
        corrs_t += [c_t]
        accuracies_t += [acc_t]

        if l_t < best_test_loss:
            best_test_loss = l_t 
            stale_epochs = 0
        else:
            stale_epochs += 1
        if j > 0 and j%20 == 0:
            make_plots(model, j, "data/{0}/epoch_{1}/".format(model_fname, j),
                losses_train, losses_test, corrs, corrs_t, accuracies, accuracies_t, test_loader)
        
        t1 = time.time()
        epochs_remaining = args.n_epochs - j
        time_per_epoch = (t1 - t0_initial)/(j + 1) 
        eta = epochs_remaining*time_per_epoch/60

        print("{} epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.2f}/{:.2f} a={:.2f}/{:.2f} l1={:.5f} l2={:.5f} l3={:.5f} stale={} eta={:.1f}m".format(
            model_fname, j, args.n_epochs,
            t1 - t0, l, l_t, c, c_t, acc, acc_t,
            losses_t[0], losses_t[1], losses_t[2], stale_epochs, eta))

    make_plots(model, j, "data/{0}/epoch_{1}/".format(model_fname, j),
        losses_train, losses_test, accuracies, accuracies_t, corrs, corrs_t, test_loader)
