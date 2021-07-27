from glob import glob
import sys, os
import os.path as osp
import pickle as pkl
import _pickle as cPickle
import math, time, tqdm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib, mplhep
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Check if the GPU configuration has been provided
import torch
use_gpu = torch.cuda.device_count()>0
multi_gpu = torch.cuda.device_count()>1

try:
    if not ("CUDA_VISIBLE_DEVICES" in os.environ):
        import setGPU
        if multi_gpu:
            print('Will use multi_gpu..')
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        else:
            print('Will use single_gpu..')
except Exception as e:
    print("Could not import setGPU, running CPU-only")

#define the global base device
if use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, DataLoader, DataListLoader, Batch
from torch_geometric.nn import GravNetConv
from torch.utils.data import random_split
import torch_cluster

sys.path.insert(1, '../')
sys.path.insert(1, '../../../plotting/')
sys.path.insert(1, '../../../mlpf/plotting/')
import args
from args import parse_args
from graph_data_delphes import PFGraphDataset, one_hot_embedding
from data_preprocessing import data_to_loader_ttbar, data_to_loader_qcd

from plot_utils import plot_confusion_matrix
from model_LRP_dnn import PFNet7

from LRP_dnn import LRP
from model_io import model_io

import networkx as nx
from torch_geometric.utils.convert import to_networkx
from tabulate import tabulate

# NOTE: this script works by loading an already trained model

#Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

#Get a unique directory name for the model
def get_model_fname(dataset, model, n_train, n_epochs, lr, target_type, batch_size, task, title):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']

    model_fname = '{}_{}_ntrain_{}_nepochs_{}_batch_size_{}_lr_{}_{}'.format(
        model_name,
        target_type,
        n_train,
        n_epochs,
        batch_size,
        lr,
        task,
        title)
    return model_fname

def map_classid_to_classname(id):
    if id==0:
        return 'null'
    if id==1:
        return 'charged hadron'
    if id==2:
        return 'neutral hadron'
    if id==3:
        return 'photon'
    if id==4:
        return 'electron'
    if id==5:
        return 'muon'

if __name__ == "__main__":

    # args = parse_args()

    # the next part initializes some args values (to run the script not from terminal)
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    # args = objectview({'train': False, 'n_train': 1, 'n_valid': 1, 'n_test': 1, 'n_epochs': 10, 'patience': 100, 'hidden_dim':256, 'hidden_dim_nn1':64, 'input_encoding': 12, 'encoding_dim': 64,
    # 'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../test_tmp_delphes/data/pythia8_qcd',
    # 'outpath': '../../../prp/models/LRP/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 2e-4, 'dropout': 0.3,
    # 'space_dim': 8, 'propagate_dimensions': 22, 'nearest': 40, 'overwrite': True,
    # 'load': True, 'load_epoch': 0, 'load_model': 'LRP_DNN_PFNet7_gen_ntrain_1_nepochs_1_batch_size_1_lr_0.001_alpha_0.0002_both_dnnnoskip_nn1_nn3_nn4',
    # 'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': False, 'nn1': True, 'nn3': True, 'nn4': True, 'title': 'dnn', 'explain': True})

    args = objectview({'train': False, 'n_train': 1, 'n_valid': 1, 'n_test': 1, 'n_epochs': 10, 'patience': 100, 'hidden_dim':256, 'hidden_dim_nn1':64, 'input_encoding': 12, 'encoding_dim': 64,
    'batch_size': 1, 'model': 'PFNet7', 'target': 'gen', 'dataset': '../../../../test_tmp_delphes/data/pythia8_ttbar', 'dataset_qcd': '../../../../test_tmp_delphes/data/pythia8_qcd',
    'outpath': '../../../../test_tmp_delphes/experiments/LRP/', 'optimizer': 'adam', 'lr': 0.001, 'alpha': 2e-4, 'dropout': 0.3,
    'space_dim': 8, 'propagate_dimensions': 22, 'nearest': 40, 'overwrite': True,
    'load': True, 'load_epoch': 0, 'load_model': 'LRP_DNN_PFNet7_gen_ntrain_1_nepochs_1_batch_size_1_lr_0.001_alpha_0.0002_both_dnnnoskip_nn1_nn3_nn4',
    'evaluate': False, 'evaluate_on_cpu': False, 'classification_only': False, 'nn1': True, 'nn3': True, 'nn4': True, 'title': 'dnn', 'explain': True})

    # define the dataset (assumes the data exists as .pt files in "processed")
    print('Processing the data..')
    full_dataset_ttbar = PFGraphDataset(args.dataset)
    full_dataset_qcd = PFGraphDataset(args.dataset_qcd)

    # constructs a loader from the data to iterate over batches
    print('Constructing data loaders..')
    train_loader, valid_loader = data_to_loader_ttbar(full_dataset_ttbar, args.n_train, args.n_valid, batch_size=args.batch_size)
    test_loader = data_to_loader_qcd(full_dataset_qcd, args.n_test, batch_size=args.batch_size)

    # element parameters
    input_dim = 12

    #one-hot particle ID and momentum
    output_dim_id = 6
    output_dim_p4 = 6

    patience = args.patience

    model_classes = {"PFNet7": PFNet7}

    model_class = model_classes[args.model]
    model_kwargs = {'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'output_dim_id': output_dim_id,
                    'output_dim_p4': output_dim_p4}

    print('Loading a previously trained model..')
    model = model_class(**model_kwargs)
    outpath = args.outpath + args.load_model
    PATH = outpath + '/epoch_' + str(args.load_epoch) + '_weights.pth'

    state_dict = torch.load(PATH, map_location=device)

    # if model was trained using DataParallel then we have to load it differently
    if "DataParallel" in args.load_model:
        state_dict = torch.load(PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
            # print('name is:', name)
        state_dict=new_state_dict

    model.load_state_dict(state_dict)

    if args.explain:
        model.eval()
        print(model)

        signal =torch.tensor([1,0,0,0,0,0],dtype=torch.float32).to(device)

        # create some hooks to retrieve intermediate activations
        activation = {}
        hooks={}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0]
            return hook

        for name, module in model.named_modules():
            if (type(module)==nn.Linear) or (type(module)==nn.LeakyReLU) or (type(module)==nn.ELU):
                hooks[name] = module.register_forward_hook(get_activation("." + name))
            print(name)

        for i, batch in enumerate(train_loader):
            t0 = time.time()

            if multi_gpu:
                X = batch
            else:
                X = batch.to(device)

            if i==0:
                # code can be written better
                # basically i run at least one forward pass to get the activations to use their shape in defining the LRP layers
                pred_ids_one_hot, pred_p4, target_ids_one_hot, target_p4, cand_ids_one_hot, cand_p4 = model(X)
                model=model_io(model,state_dict,dict(),activation)
                explainer=LRP(model)

            else:
                pred_ids_one_hot, pred_p4, target_ids_one_hot, target_p4, cand_ids_one_hot, cand_p4 = model.model(X)

            to_explain={"A":activation,"inputs":dict(x=X.x,
                                                batch=X.batch),"y":target_ids_one_hot,"R":dict(), "pred":pred_ids_one_hot,
                                                "outpath":args.outpath, "load_model":args.load_model}

            model.set_dest(to_explain["A"])

            explainer.explain(to_explain,save=False,return_result=True, signal=signal)

            break

## -----------------------------------------------------------
# # to retrieve a stored variable in pkl file
# import _pickle as cPickle
# with open('../../../prp/models/LRP/LRP_DNN_PFNet7_gen_ntrain_1_nepochs_1_batch_size_1_lr_0.001_alpha_0.0002_both_dnnnoskip_nn1_nn3_nn4/R_scorez.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     s = cPickle.load(f)
