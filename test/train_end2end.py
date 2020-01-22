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
import sklearn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_data import PFGraphDataset

device = torch.device('cuda' if use_gpu else 'cpu')

#all candidate pdg-ids (multiclass labels)
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

#map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}

def get_model_fname(model):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']
    
    model_fname = '%s_%d_%s_%s'%(model_name,model_params,
                                 model_cfghash, model_user)
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

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)
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
            nn.Linear(hidden_dim, hidden_dim),
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
        n_onehot = len(class_to_id)
        cand_ids = r[:, :n_onehot]
        cand_p4 = r[:, n_onehot:]
        return edge_weight, cand_ids, cand_p4

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

    #Convert pdg-ids to consecutive class labels
    for k, v in class_to_id.items():
        m = data.y_candidates[:, 0] == v
        data.y_candidates[m, 0] = k

    #Create a one-hot encoded vector of the class labels
    id_onehot = torch.nn.functional.one_hot(data.y_candidates[:, 0].to(dtype=torch.long), num_classes=len(class_to_id))
    
    #Extract the ids and momenta
    data.y_candidates_id = data.y_candidates[:, 0].to(dtype=torch.long)
    data.y_candidates = data.y_candidates[:, 1:]


def train(model, loader, batch_size, epoch, optimizer):
    corrs_batch = []

    is_train = not (optimizer is None)

    losses = np.zeros((len(loader), 3))

    for i, data in enumerate(loader):
        data = data.to(device)
        data_prep(data)

        if is_train:
            optimizer.zero_grad()

        edges, cand_id_onehot, cand_momentum = model(data)
        _, indices = torch.max(cand_id_onehot, -1)

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        msk = (indices != 0) & (data.y_candidates_id != 0)

        # msk_batch, _ = torch_geometric.utils.to_dense_batch(msk, data.batch)
        # ids_batch, _ = torch_geometric.utils.to_dense_batch(data.y_candidates_id!=0, data.batch)

        # ncand_pred = msk_batch.sum(axis=1).numpy()
        # ncand_true = ids_batch.sum(axis=1).numpy()

        #Loss for output candidate id (multiclass)
        l1 = torch.nn.functional.cross_entropy(cand_id_onehot, data.y_candidates_id)

        #Loss for candidate p4 properties (regression)
        l2 = torch.nn.functional.mse_loss(cand_momentum, data.y_candidates) / 10.0
        
        #Loss for edges enabled/disabled in clustering (binary)
        l3 = torch.nn.functional.binary_cross_entropy(edges, data.y) * 2.0

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

        corrs_batch += [corr_pt]

    corr = np.mean(corrs_batch)
    losses = losses.sum(axis=0)
    return losses, corr

def make_plots(n_epoch, path, losses_train, losses_test, corrs_train, corrs_test, test_loader):
   
    try:
        os.makedirs(path)
    except Exception as e:
        pass

    for i in range(losses_train.shape[1]):
        fig = plt.figure(figsize=(5,5))
        plt.plot(losses_train[:n_epoch, i])
        plt.plot(losses_test[:n_epoch, i])
        plt.ylim(0.5*losses_train[:n_epoch, i][-1],2*losses_train[:n_epoch, i][-1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(path + "loss_{0}.pdf".format(i))
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
 
    #plot the first 5 batches from the test dataset 
    for i, data in enumerate(test_loader):
        if i>5:
            break
        d = data.to(device=device)
        data_prep(data)

        edges, cand_id_onehot, cand_momentum = model(d)
        _, indices = torch.max(cand_id_onehot, -1)
        msk = (indices != 0) & (data.y_candidates_id != 0)
        inds2 = torch.nonzero(msk)
        perm = torch.randperm(len(inds2))
        inds2 = inds2[perm[:1000]]

        confusion = sklearn.metrics.confusion_matrix(
            data.y_candidates_id.detach().cpu().numpy(), indices.detach().cpu().numpy(),
            labels=range(len(class_labels)))
        plot_confusion_matrix(cm = confusion, target_names=class_labels, normalize=False)
        plt.savefig(path + "confusion_{0}.pdf".format(i))
        plt.clf()

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
    plt.close("all")

if __name__ == "__main__":
    full_dataset = PFGraphDataset(root='data/TTbar_run3')
    full_dataset.raw_dir = "data/TTbar_run3"
    full_dataset.processed_dir = "data/TTbar_run3/processed_jd"

    model_choice = sys.argv[1]

    input_dim = 8

    #one-hot particle ID and 3 momentum components
    output_dim = len(class_to_id) + 3
    edge_dim = 1

    batch_size = 20
    n_train = 100
    n_epochs = 1000
    lr = 1e-4
    hidden_dim = 64
    patience = n_epochs

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=2*n_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    if model_choice == "simple":
        model = PFNet4(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    elif model_choice == "gnn":
        model = PFNet6(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    
    model_fname = get_model_fname(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss = torch.nn.MSELoss()
    loss2 = torch.nn.BCELoss()
    
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)
    
    model.train()
    
    losses_train = np.zeros((n_epochs, 3))
    losses_test = np.zeros((n_epochs, 3))

    corrs = []
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
        l = sum(losses)
        losses_train[j] = losses
        corrs += [c]

        model.eval()
        losses_t, c_t = train(model, test_loader, batch_size, j, None)
        l_t = sum(losses_t)
        losses_test[j] = losses_t
        corrs_t += [c_t]
        
        if l_t < best_test_loss:
            best_test_loss = l_t 
            stale_epochs = 0
        else:
            stale_epochs += 1
        if j > 0 and j%20 == 0:
            make_plots(j, "data/{0}/epoch_{1}/".format(model_fname, j), losses_train, losses_test, corrs, corrs_t, test_loader)
        
        t1 = time.time()
        epochs_remaining = n_epochs - j
        time_per_epoch = (t1 - t0_initial)/(j + 1) 
        eta = epochs_remaining*time_per_epoch/60

        print("epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.2f}/{:.2f} l1={:.5f} l2={:.5f} l3={:.5f} stale={} eta={:.1f}m".format(
            j, n_epochs, t1 - t0, l, l_t, c, c_t, losses_t[0], losses_t[1], losses_t[2], stale_epochs, eta))
    make_plots(j, "data/{0}/epoch_{1}/".format(model_fname, j), losses_train, losses_test, corrs, corrs_t, test_loader)
