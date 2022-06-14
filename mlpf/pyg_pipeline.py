from torch_geometric.data import Data, Batch
from pyg import parse_args
from pyg import MLPF, training_loop, make_predictions, make_plots
from pyg import get_model_fname, save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg.dataset import PFGraphDataset, one_hot_embedding

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

import mplhep as hep
import time
import matplotlib.pyplot as plt
from glob import glob
import sys
import os
import os.path as osp
import shutil
import pickle as pkl
import json
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')

# Check if the GPU configuration has been provided
use_gpu = torch.cuda.device_count() > 0
multi_gpu = torch.cuda.device_count() > 1

if multi_gpu or use_gpu:
    print(f'Will use {torch.cuda.device_count()} gpu(s)')
else:
    print('Will use cpu')

# define the global base device
if use_gpu:
    device = torch.device('cuda:0')
    print("GPU model:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')


if __name__ == "__main__":

    args = parse_args()

    # load the dataset (assumes the datafiles exist as .pt files under <args.dataset>/processed)
    dataset = PFGraphDataset(args.dataset, args.data)

    # create dataloaders
    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=args.n_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train, stop=args.n_train + args.n_valid))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, prefetch_factor=2)

    # retrieve the dimensions of the PF-elements & PF-candidates to set the input/output dimension of the model
    if args.data == 'delphes':
        input_dim = len(features_delphes)
        num_classes = 6   # we have 6 classes/pids for delphes
    elif args.data == 'cms':
        input_dim = len(features_cms)
        num_classes = 9   # we have 9 classes/pids for cms (including taus)
    output_dim_p4 = len(target_p4)

    if args.load:  # load a pre-trained specified model
        outpath = args.outpath + args.load_model
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.load_model, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

        if multi_gpu:
            model = torch_geometric.nn.DataParallel(model)

    else:
        print('Instantiating a model..')
        model_kwargs = {'input_dim': input_dim,
                        'num_classes': num_classes,
                        'output_dim_p4': output_dim_p4,
                        'embedding_dim': args.embedding_dim,
                        'hidden_dim1': args.hidden_dim1,
                        'hidden_dim2': args.hidden_dim2,
                        'num_convs': args.num_convs,
                        'space_dim': args.space_dim,
                        'propagate_dim': args.propagate_dim,
                        'k': args.nearest,
                        }

        model = MLPF(**model_kwargs)

        # get a directory name for the model to store the model's weights and plots
        model_fname = get_model_fname(model, args.data, args.n_train, args.n_epochs, args.target, args.title)
        outpath = osp.join(args.outpath, model_fname)

        model.to(device)

        if multi_gpu:
            print("Parallelizing the training..")
            model = torch_geometric.nn.DataParallel(model)

        save_model(args, model_fname, outpath, model_kwargs)

        print(model)
        print(model_fname)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()

        torch.backends.cudnn.benchmark = True

        # run a training of the model
        training_loop(device, args.data, model, multi_gpu,
                      train_loader, valid_loader,
                      args.batch_events,
                      args.n_epochs, args.patience,
                      optimizer, args.alpha, args.target,
                      num_classes, outpath)

    model.eval()

    # evaluate on testing data..
    if args.load:
        epoch_on_plots = args.load_epoch
    else:
        epoch_on_plots = args.n_epochs - 1

    dataset_qcd = PFGraphDataset(args.dataset_qcd, args.data)

    make_directories_for_plots(outpath, 'test_data')
    make_predictions(device, args.data, model, multi_gpu, dataset, args.n_test, args.batch_size, args.batch_events, num_classes, outpath + '/test_data_plots/')
    make_plots(device, args.data, model, num_classes, outpath + '/test_data_plots/', args.target, epoch_on_plots, 'QCD')
#
#
# dataset = PFGraphDataset('../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/', 'cms')
# len(dataset)
#
# loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0, prefetch_factor=2)
#
# len(loader)
# for file in loader:
#     break
#
# len(file)
#
# file
