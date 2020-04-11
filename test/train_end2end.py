import sys
import os
if not ("CUDA_VISIBLE_DEVICES" in os.environ):
    import setGPU

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
from torch_geometric.data import Data, DataLoader, Batch

from glob import glob
import numpy as np
import os.path as osp

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

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

device = torch.device('cuda')

def prepare_dataframe(model, loader):
    model.eval()
    dfs = []
    eval_time = 0
    for i, data in enumerate(loader):
        data = data.to(device)

        t0 = time.time()
        _, pred_id_onehot, pred_momentum = model(data)
        _, pred_id = torch.max(pred_id_onehot, -1)
        t1 = time.time()
        eval_time += (t1 - t0)
        pred_momentum[pred_id==0] = 0

        data = data.to("cpu")

        df = pandas.DataFrame()

        df["elem_type"] = [int(graph_data.elem_labels[i]) for i in torch.argmax(data.x[:, :len(graph_data.elem_labels)], axis=-1).cpu().numpy()]
        df["gen_pid"] = [int(graph_data.class_labels[i]) for i in data.gen[0].cpu().numpy()]
        df["gen_pt"] = data.gen[1][:, 0].cpu().numpy()
        df["gen_eta"] = data.gen[1][:, 1].cpu().numpy()
        df["gen_phi"] = data.gen[1][:, 2].cpu().numpy()
        df["gen_e"] = data.gen[1][:, 3].cpu().numpy()

        df["cand_pid"] = [int(graph_data.class_labels[i]) for i in data.cand[0].cpu().numpy()]
        df["cand_pt"] = data.cand[1][:, 0].cpu().numpy()
        df["cand_eta"] = data.cand[1][:, 1].cpu().numpy()
        df["cand_phi"] = data.cand[1][:, 2].cpu().numpy()
        df["cand_e"] = data.cand[1][:, 3].cpu().numpy()
        df["pred_pid"] = [int(graph_data.class_labels[i]) for i in pred_id.cpu().numpy()]

        df["pred_pt"] = pred_momentum[:, 0].detach().cpu().numpy()
        df.loc[df["pred_pid"]==0, "pred_pt"] = 0
        df["pred_eta"] = pred_momentum[:, 1].detach().cpu().numpy()
        df["pred_phi"] = pred_momentum[:, 2].detach().cpu().numpy()
        df["pred_e"] = pred_momentum[:, 3].detach().cpu().numpy()

        dfs += [df]

    print("eval_time {:.2f} ms/batch".format(1000.0*eval_time/len(loader)))

    df = pandas.concat(dfs, ignore_index=True)
 
    #Print some stats for each target particle type 
    for pid in [211, -211, 130, 22, -11, 11, 13, -13, 1, 2]:
        msk_gen = df["gen_pid"] == pid
        msk_pred = df["pred_pid"] == pid

        npred = int(np.sum(msk_pred))
        ngen = int(np.sum(msk_gen))
        tpr = np.sum(msk_gen & msk_pred) / npred
        fpr = np.sum(~msk_gen & msk_pred) / npred
        eff = np.sum(msk_gen & msk_pred) / ngen

        mu = 0.0
        sigma = 0.0
        if np.sum(msk_pred) > 0:
            pts = df[msk_gen & msk_pred][["gen_pt", "pred_pt"]].values
            r = pts[:, 1]/pts[:, 0]
            mu, sigma = np.mean(r), np.std(r)
        print("pid={pid} Ngen={ngen} Npred={npred} eff={eff:.4f} tpr={tpr:.4f} fpr={fpr:.4f} pt_mu={pt_mu:.4f} pt_s={pt_s:.4f}".format(
            pid=pid, ngen=ngen, npred=npred, eff=eff, tpr=tpr, fpr=fpr, pt_mu=mu, pt_s=sigma
        ))
    sumpt_cand = df[df["cand_pid"]!=0]["cand_pt"].sum()/len(dfs)
    sumpt_gen = df[df["gen_pid"]!=0]["gen_pt"].sum()/len(dfs)
    sumpt_pred = df[df["pred_pid"]!=0]["pred_pt"].sum()/len(dfs)
    print("sumpt_cand={:.2f} sumpt_gen={:.2f} sumpt_pred={:.2f}".format(sumpt_cand, sumpt_gen, sumpt_pred))
 
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
            nn.Linear(hidden_dim, len(class_to_id) + output_dim),
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


#Baseline model with graph attention convolution & edge classification, slow to train 
class PFNet6(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4, dropout_rate=0.5):
        super(PFNet6, self).__init__()

        act = nn.LeakyReLU

        self.inp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

        #pairs of embedded nodes + edge
        self.edgenet = nn.Sequential(
            nn.Linear(2*hidden_dim + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)

        self.nn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, len(class_to_id)),
        )
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim + len(class_to_id), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, 4),
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
        xpairs = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]], edge_weight.unsqueeze(-1)], axis=-1)
        edge_weight2 = self.edgenet(xpairs).squeeze(-1)
        edge_mask = edge_weight2 > 0.5
        row, col = data.edge_index
        row2, col2 = row[edge_mask], col[edge_mask]

        #Run a second convolution with the new edges
        x = torch.nn.functional.leaky_relu(self.conv2(x, torch.stack([row2, col2])))

        #Final candidate inference
        cand_ids = self.nn1(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn2(torch.cat([cand_ids, x], axis=-1))

        return edge_weight2, cand_ids, cand_p4

#Simplified model with SGConv, no edge classification, fast to train
class PFNet7(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4, dropout_rate=0.5):
        super(PFNet7, self).__init__()
  
        act = nn.LeakyReLU
        self.nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = SGConv(hidden_dim, hidden_dim) 
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, len(class_to_id)),
        )
        self.nn3 = nn.Sequential(
            nn.Linear(hidden_dim + len(class_to_id), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=dropout_rate),
            act(),
            nn.Linear(hidden_dim, 4),
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, data):
        batch = data.batch
        edge_weight = data.edge_attr.squeeze(-1)
        edge_index = data.edge_index
        
        #Run a convolution
        x = self.nn1(data.x)
        x = torch.nn.functional.leaky_relu(self.conv1(x, edge_index))
        
        cand_ids = self.nn2(x)
        cand_p4 = data.x[:, len(elem_to_id):len(elem_to_id)+4] + self.nn3(torch.cat([x, cand_ids], axis=-1))
        return torch.sigmoid(edge_weight), cand_ids, cand_p4


model_classes = {
    "PFDenseNet": PFDenseNet,
    "PFNet6": PFNet6,
    "PFNet7": PFNet7,
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=80, help="number of training events")
    parser.add_argument("--n_test", type=int, default=20, help="number of testing events")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--n_plot", type=int, default=10, help="make plots every iterations")
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden dimension")
    parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=1.0, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering classification")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    args = parser.parse_args()
    return args

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type):
    return train(model, loader, epoch, None, l1m, l2m, l3m, target_type)

def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    losses = np.zeros((len(loader), 3))
    corrs_batch = np.zeros(len(loader))
    accuracies_batch = np.zeros(len(loader))

    for i, data in enumerate(loader):

        if target_type == "gen":
            data.cand = None
            del data.cand
        elif target_type == "cand":
            data.gen = None
            del data.gen

        data = data.to(device)
        
        target = getattr(data, target_type)
        target = (target[0].to(device), target[1].to(device), target[2].to(device))

        vs, cs = torch.unique(target[0], return_counts=True)
        weights = torch.zeros(len(class_to_id)).to(device=device)
        for k, v in zip(vs, cs):
            weights[k] = 1.0/float(v)

        if is_train:
            optimizer.zero_grad()

        edges, cand_id_onehot, cand_momentum = model(data)
        _, indices = torch.max(cand_id_onehot, -1)

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        #msk = (indices != 0)
        msk = (indices != 0) & (target[0] != 0)

        accuracies_batch[i] = accuracy_score(target[0][msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        #Loss for output candidate id (multiclass)
        if l1m > 0.0:
            l1 = l1m * torch.nn.functional.cross_entropy(cand_id_onehot, target[0], weight=weights)
        else:
            l1 = torch.tensor(0.0).to(device=device)

        #Loss for candidate p4 properties (regression)
        l2 = torch.tensor(0.0).to(device=device)
        if l2m > 0.0:
            l2 += l2m*torch.nn.functional.mse_loss(cand_momentum[:, 0], target[1][:, 0])
            l2 += l2m*torch.nn.functional.mse_loss(cand_momentum[:, 1], target[1][:, 1])
            zs = torch.zeros_like(target[1][:, 2])
            l2 += l2m*torch.nn.functional.mse_loss(torch.fmod(cand_momentum[:, 2] - target[1][:, 2] + np.pi, 2*np.pi) - np.pi, zs)
            l2 += l2m*torch.nn.functional.mse_loss(cand_momentum[:, 3], target[1][:, 3])

        #Loss for edges enabled/disabled in clustering (binary)
        if l3m > 0.0:
            l3 = l3m*torch.nn.functional.binary_cross_entropy(edges, target[2])
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
                target[1][msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        i += 1

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.sum(axis=0)
    return losses, corr, acc

def make_plots(model, n_epoch, path, losses_train, losses_test, corrs_train, corrs_test, accuracies, accuracies_t, test_loader):
   
    try:
        os.makedirs(path)
    except Exception as e:
        pass

    modpath = path + 'weights.pth'
    torch.save(model.state_dict(), modpath)

    df = prepare_dataframe(model, test_loader)
    df.to_pickle(path + "df.pkl.bz2")

    np.savetxt(path+"losses_train.txt", losses_train)
    np.savetxt(path+"losses_test.txt", losses_test)
    np.savetxt(path+"corrs_train.txt", corrs_train)
    np.savetxt(path+"corrs_test.txt", corrs_test)
    np.savetxt(path+"accuracies_train.txt", accuracies)
    np.savetxt(path+"accuracies_test.txt", accuracies_t)

#    for i in range(losses_train.shape[1]):
#        fig = plt.figure(figsize=(5,5))
#        ax = plt.axes()
#        plt.ylabel("train loss")
#        plt.plot(losses_train[:n_epoch, i])
#        ax2=ax.twinx()
#        ax2.plot(losses_test[:n_epoch, i], color="orange")
#        ax2.set_ylabel("test loss", color="orange")
#        plt.xlabel("epoch")
#        plt.tight_layout()
#        plt.savefig(path + "loss_{0}.pdf".format(i))
#        del fig
#        plt.clf()

if __name__ == "__main__":
    args = parse_args()
    
    full_dataset = PFGraphDataset(args.dataset)

    #one-hot encoded element ID + element parameters
    input_dim = 23

    #one-hot particle ID and momentum
    output_dim = 7

    edge_dim = 1

    patience = args.n_epochs

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=args.n_train))
    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=args.n_train, stop=args.n_train+args.n_test))

    def collate(batch):
        return batch

    train_loader = DataLoader(train_dataset, batch_size=None, batch_sampler=None, pin_memory=True, shuffle=False)
    train_loader.collate_fn = collate
    test_loader = DataLoader(test_dataset, batch_size=None, batch_sampler=None, pin_memory=True, shuffle=False)
    test_loader.collate_fn = collate
    print("train_loader", len(train_loader))
    print("test_loader", len(test_loader))

    model_class = model_classes[args.model]
    model = model_class(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim, dropout_rate=args.dropout).to(device)
    model_fname = get_model_fname(args.dataset, model, args.n_train, args.lr, args.target)
    if os.path.isdir("data/" + model_fname):
        print("model output data/{} already exists, please delete it".format(model_fname))
        sys.exit(0)

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

    initial_epochs = 10
   
    t0_initial = time.time()
    print("Training over {} epochs".format(args.n_epochs)) 
    for j in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        model.train()

        losses, c, acc = train(model, train_loader, j, optimizer, args.l1, args.l2, args.l3, args.target)
        l = sum(losses)
        losses_train[j] = losses
        corrs += [c]
        accuracies += [acc]

        model.eval()
        losses_t, c_t, acc_t = test(model, test_loader, j, args.l1, args.l2, args.l3, args.target)
        l_t = sum(losses_t)
        losses_test[j] = losses_t
        corrs_t += [c_t]
        accuracies_t += [acc_t]

        if l_t < best_test_loss:
            best_test_loss = l_t 
            stale_epochs = 0
        else:
            stale_epochs += 1
        if j > 0 and j%args.n_plot == 0:
            make_plots(
                model, j, "data/{0}/epoch_{1}/".format(model_fname, j),
                losses_train, losses_test, corrs, corrs_t,
                accuracies, accuracies_t, test_loader)
 
        t1 = time.time()
        epochs_remaining = args.n_epochs - j
        time_per_epoch = (t1 - t0_initial)/(j + 1) 
        eta = epochs_remaining*time_per_epoch/60

        losses_str = "[" + ",".join(["{:.4f}".format(x) for x in losses_t]) + "]"
        print("epoch={}/{} dt={:.2f}s l={:.5f}/{:.5f} c={:.2f}/{:.2f} a={:.2f}/{:.2f} partial_losses={} stale={} eta={:.1f}m".format(
            j, args.n_epochs,
            t1 - t0, l, l_t, c, c_t, acc, acc_t,
            losses_str, stale_epochs, eta))

    make_plots(
        model, j, "data/{0}/epoch_{1}/".format(model_fname, j),
        losses_train, losses_test, corrs, corrs_t,
        accuracies, accuracies_t, test_loader)
