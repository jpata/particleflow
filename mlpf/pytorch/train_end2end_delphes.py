import sys
import os

from comet_ml import Experiment

#Check if the GPU configuration has been provided
try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
except Exception as e:
    print("Could not import setGPU, running CPU-only")

import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

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
import tqdm
import sklearn
import pandas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from model import PFNet7
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import from_data_to_loader

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
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

model_classes = {
    "PFNet7": PFNet7,
}

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target).sum(axis=1) ** 2)

def compute_weights(target_ids, device):
    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(output_dim_id).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0/math.sqrt(float(v))
    return weights

@torch.no_grad()
def test(model, loader, epoch, l1m, l2m, l3m, target_type):
    with torch.no_grad():
        ret = train(model, loader, epoch, None, l1m, l2m, l3m, target_type, None)
    return ret


def train(model, loader, epoch, optimizer, l1m, l2m, l3m, target_type, scheduler):

    is_train = not (optimizer is None)

    if is_train:
        model.train()
    else:
        model.eval()

    #loss values for each batch: classification, regression
    losses = np.zeros((len(loader), 3))

    #accuracy values for each batch (monitor classification performance)
    accuracies_batch = np.zeros(len(loader))

    #correlation values for each batch (monitor regression performance)
    corrs_batch = np.zeros(len(loader))

    #epoch confusion matrix
    conf_matrix = np.zeros((output_dim_id, output_dim_id))

    #keep track of how many data points were processed
    num_samples = 0

    for i, batch in enumerate(loader):
        t0 = time.time()

        if not multi_gpu:
            batch = batch.to(device)

        if is_train:
            optimizer.zero_grad()

        # forward pass
        cand_id_onehot, cand_momentum, new_edge_index = model(batch)

        _dev = cand_id_onehot.device                   # store the device in dev
        _, indices = torch.max(cand_id_onehot, -1)     # picks the maximum PID location and stores the index (opposite of one_hot_embedding)

        num_samples += len(cand_id_onehot)

        # concatenate ygen/ycand over the batch to compare with the truth label
        # now: ygen/ycand is of shape [~5000*batch_size, 6] corresponding to the output of the forward pass
        if args.target == "gen":
            target_ids = batch.ygen_id
            target_p4 = batch.ygen
        elif args.target == "cand":
            target_ids = batch.ycand_id
            target_p4 = batch.ycand

        #Predictions where both the predicted and true class label was nonzero
        #In these cases, the true candidate existed and a candidate was predicted
        # target_ids_msk reverts the one_hot_embedding
        # msk is a list of booleans of shape [~5000*batch_size] where each boolean correspond to whether a candidate was predicted
        _, target_ids_msk = torch.max(target_ids, -1)
        msk = ((indices != 0) & (target_ids_msk != 0)).detach().cpu()
        msk2 = ((indices != 0) & (indices == target_ids_msk))

        accuracies_batch[i] = accuracy_score(target_ids_msk[msk].detach().cpu().numpy(), indices[msk].detach().cpu().numpy())

        # a manual rescaling weight given to each class
        weights = compute_weights(torch.max(target_ids,-1)[1], _dev)

        #Loss for output candidate id (multiclass)
        l1 = l1m * torch.nn.functional.cross_entropy(target_ids, indices, weight=weights)

        #Loss for candidate p4 properties (regression)
        l2 = l2m * torch.nn.functional.mse_loss(target_p4[msk2], cand_momentum[msk2])

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
            if not scheduler is None:
                scheduler.step()

        #Compute correlation of predicted and true pt values for monitoring
        corr_pt = 0.0
        if msk.sum()>0:
            corr_pt = np.corrcoef(
                cand_momentum[msk, 0].detach().cpu().numpy(),
                target_p4[msk, 0].detach().cpu().numpy())[0,1]

        corrs_batch[i] = corr_pt

        conf_matrix += confusion_matrix(target_ids_msk.detach().cpu().numpy(),
                                        np.argmax(cand_id_onehot.detach().cpu().numpy(),axis=1), labels=range(6))

    corr = np.mean(corrs_batch)
    acc = np.mean(accuracies_batch)
    losses = losses.mean(axis=0)
    return num_samples, losses, corr, acc, conf_matrix

def train_loop():
    t0_initial = time.time()

    losses_train = np.zeros((args.n_epochs, 3))
    losses_val = np.zeros((args.n_epochs, 3))

    corrs = []
    corrs_v = []
    accuracies = []
    accuracies_v = []
    best_val_loss = 99999.9
    stale_epochs = 0

    print("Training over {} epochs".format(args.n_epochs))
    for epoch in range(args.n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        with experiment.train():
            model.train()

            num_samples_train, losses, c, acc, conf_matrix = train(model, train_loader, epoch, optimizer,
                                                                   args.l1, args.l2, args.l3, args.target, scheduler)

            experiment.log_metric('lr', optimizer.param_groups[0]['lr'], step=epoch)
            l = sum(losses)
            losses_train[epoch] = losses
            corrs += [c]
            accuracies += [acc]
            experiment.log_metric('loss',l, step=epoch)
            experiment.log_metric('loss1',losses[0], step=epoch)
            experiment.log_metric('loss2',losses[1], step=epoch)
            experiment.log_metric('loss3',losses[2], step=epoch)
            experiment.log_metric('corrs',c, step=epoch)
            experiment.log_metric('accuracy',acc, step=epoch)
            experiment.log_confusion_matrix(matrix=conf_matrix, step=epoch,
                                            title='Confusion Matrix Full',
                                            file_name='confusion-matrix-full-train-%03d.json' % epoch,
                                            labels = [str(c) for c in range(output_dim_id)])

        with experiment.validate():
            model.eval()
            num_samples_val, losses_v, c_v, acc_v, conf_matrix_v = test(model, valid_loader, epoch,
                                                                        args.l1, args.l2, args.l3, args.target)
            l_v = sum(losses_v)
            losses_val[epoch] = losses_v
            corrs_v += [c_v]
            accuracies_v += [acc_v]
            experiment.log_metric('loss',l_v, step=epoch)
            experiment.log_metric('loss1',losses_v[0], step=epoch)
            experiment.log_metric('loss2',losses_v[1], step=epoch)
            experiment.log_metric('loss3',losses_v[2], step=epoch)
            experiment.log_metric('corrs',c_v, step=epoch)
            experiment.log_metric('accuracy',acc_v, step=epoch)
            experiment.log_confusion_matrix(matrix=conf_matrix_v, step=epoch,
                                            title='Confusion Matrix Full',
                                            file_name='confusion-matrix-full-val-%03d.json' % epoch,
                                            labels = [str(c) for c in range(output_dim_id)])

        if l_v < best_val_loss:
            best_val_loss = l_v
            stale_epochs = 0
        else:
            stale_epochs += 1

        t1 = time.time()
        epochs_remaining = args.n_epochs - epoch
        time_per_epoch = (t1 - t0_initial)/(epoch + 1)
        experiment.log_metric('time_per_epoch', time_per_epoch, step=epoch)
        eta = epochs_remaining*time_per_epoch/60

        spd = (num_samples_val+num_samples_train)/time_per_epoch
        losses_str = "[" + ",".join(["{:.4f}".format(x) for x in losses_v]) + "]"

        torch.save(model.state_dict(), "{0}/epoch_{1}_weights.pth".format(outpath, epoch))

        print("epoch={}/{} dt={:.2f}s loss_train={:.5f} loss_valid={:.5f} c={:.2f}/{:.2f} a={:.6f}/{:.6f} partial_losses={} stale={} eta={:.1f}m spd={:.2f} samples/s lr={}".format(
            epoch+1, args.n_epochs,
            t1 - t0, l, l_v, c, c_v, acc, acc_v,
            losses_str, stale_epochs, eta, spd, optimizer.param_groups[0]['lr']))

    print('Done with training.')

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=2, help="number of data files to use for training.. each file contains 100 events")
    parser.add_argument("--n_val", type=int, default=1, help="number of data files to use for validation.. each file contains 100 events")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="patience before early stopping")
    parser.add_argument("--hidden_dim", type=int, default=32, help="hidden dimension")
    parser.add_argument("--encoding_dim", type=int, default=256, help="encoded element dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of .pt files to load in parallel")
    parser.add_argument("--model", type=str, choices=sorted(model_classes.keys()), help="type of model to use", default="PFNet6")
    parser.add_argument("--target", type=str, choices=["cand", "gen"], help="Regress to PFCandidates or GenParticles", default="cand")
    parser.add_argument("--dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--outpath", type=str, default = 'experiments/', help="Output folder")
    parser.add_argument("--activation", type=str, default='leaky_relu', choices=["selu", "leaky_relu", "relu"], help="activation function")
    parser.add_argument("--optimizer", type=str, default='adam', choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="Loss multiplier for pdg-id classification")
    parser.add_argument("--l2", type=float, default=0.001, help="Loss multiplier for momentum regression")
    parser.add_argument("--l3", type=float, default=1.0, help="Loss multiplier for clustering")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--radius", type=float, default=0.1, help="Radius-graph radius")
    parser.add_argument("--convlayer", type=str, choices=["gravnet-knn", "gravnet-radius", "sgconv", "gatconv"], help="Convolutional layer", default="gravnet")
    parser.add_argument("--convlayer2", type=str, choices=["sgconv", "graphunet", "gatconv", "none"], help="Convolutional layer", default="none")
    parser.add_argument("--space_dim", type=int, default=2, help="Spatial dimension for clustering in gravnet layer")
    parser.add_argument("--nearest", type=int, default=3, help="k nearest neighbors in gravnet layer")
    parser.add_argument("--overwrite", action='store_true', help="overwrite if model output exists")
    parser.add_argument("--disable_comet", action='store_true', help="disable comet-ml")
    parser.add_argument("--input_encoding", type=int, help="use an input encoding layer", default=0)
    parser.add_argument("--load", type=str, help="Load the weight file", required=False, default=None)
    parser.add_argument("--scheduler", type=str, help="LR scheduler", required=False, default="none", choices=["none", "onecycle"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    # # the next part initializes some args values (to run the script not from terminal)
    # class objectview(object):
    #     def __init__(self, d):
    #         self.__dict__ = d
    #
    # args = objectview({'n_train': 2, 'n_val': 1, 'n_epochs': 3, 'patience': 100, 'hidden_dim':32, 'encoding_dim': 256,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'cand', 'dataset': '../../test_tmp_delphes/data/pythia8_ttbar',
    # 'outpath': 'experiments/', 'activation': 'leaky_relu', 'optimizer': 'adam', 'lr': 1e-4, 'l1': 1, 'l2': 0.001, 'l3': 1, 'dropout': 0.5,
    # 'radius': 0.1, 'convlayer': 'gravnet-radius', 'convlayer2': 'none', 'space_dim': 2, 'nearest': 3, 'overwrite': True,
    # 'disable_comet': True, 'input_encoding': 0, 'load': None, 'scheduler': 'none'})

    # define the dataset
    full_dataset = PFGraphDataset(args.dataset)

    # constructs a loader from the data to iterate over batches
    train_loader, valid_loader = from_data_to_loader(full_dataset, args.n_train, args.n_val, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    patience = args.patience

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'encoding_dim': args.encoding_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4,
                    'dropout_rate': args.dropout,
                    'convlayer': args.convlayer,
                    'convlayer2': args.convlayer2,
                    'radius': args.radius,
                    'space_dim': args.space_dim,
                    'activation': args.activation,
                    'nearest': args.nearest,
                    'input_encoding': args.input_encoding}

    #instantiate the model
    model = model_class(**model_kwargs)
    if args.load:
        s1 = torch.load(args.load, map_location=torch.device('cpu'))
        s2 = {k.replace("module.", ""): v for k, v in s1.items()}
        model.load_state_dict(s2)

    if multi_gpu:
        model = torch_geometric.nn.DataParallel(model)

    model.to(device)

    model_fname = get_model_fname(args.dataset, model, args.n_train, args.lr, args.target)

    # need your api key in a .comet.config file: see https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables
    experiment = Experiment(project_name="particleflow", disabled=args.disable_comet)
    experiment.set_model_graph(repr(model))
    experiment.log_parameters(dict(model_kwargs, **{'model': args.model, 'lr':args.lr, 'model_fname': model_fname,
                                                    'l1': args.l1, 'l2':args.l2,
                                                    'n_train':args.n_train, 'target':args.target, 'optimizer': args.optimizer}))
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

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=int(len(train_loader)),
            epochs=args.n_epochs + 1,
            anneal_strategy='linear',
        )

    print(model)
    print(model_fname)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("params", params)

    model.train()

    train_loop()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     train_loop()

    # print(prof.key_averages().table(sort_by="cuda_time_total"))
