from torch_geometric.data import Data, Batch
from pyg import parse_args
from pyg import PFGraphDataset, one_hot_embedding
from pyg import MLPF, training_loop, training_loop_ddp, make_predictions, make_plots
from pyg import save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg import make_file_loaders

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

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide='ignore', invalid='ignore')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, args):
    print(f"Running training loop on rank {rank}: {torch.cuda.get_device_name(rank)}")

    setup(rank, world_size)

    if rank == 0:
        print(model)
        print(args.model_prefix)

    # retrieve the dimensions of the PF-elements & PF-candidates to set the input/output dimension of the model
    if args.data == 'delphes':
        input_dim = len(features_delphes)
        num_classes = 6   # we have 6 classes/pids for delphes
    elif args.data == 'cms':
        input_dim = len(features_cms)
        num_classes = 9   # we have 9 classes/pids for cms (including taus)
    output_dim_p4 = len(target_p4)

    outpath = osp.join(args.outpath, args.model_prefix)

    # load the dataset (assumes the datafiles exist as .pt files under <args.dataset>/processed)
    dataset = PFGraphDataset(args.dataset, args.data)

    # give each gpu a subset of the data
    hyper_train = int(args.n_train / world_size)
    hyper_valid = int(args.n_valid / world_size)

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=rank * hyper_train, stop=(rank + 1) * hyper_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train + rank * hyper_valid, stop=args.n_train + (rank + 1) * hyper_valid))

    # construct file loaders
    file_loader_train = make_file_loaders(train_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    file_loader_valid = make_file_loaders(valid_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    # create model and move it to GPU with id rank
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
    model = MLPF(**model_kwargs).to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    training_loop_ddp(rank, args.data, model, file_loader_train, file_loader_valid,
                      args.batch_size, args.batch_events, args.n_epochs, args.patience,
                      optimizer, args.alpha, args.target, num_classes, outpath)

    cleanup()


def run_demo(demo_fn, world_size, args):
    mp.spawn(demo_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":

    args = parse_args()

    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

    # save model_kwargs and hyperparameters
    save_model(args, args.model_prefix, outpath, model_kwargs)

    # run_demo(train, world_size, args)

    if args.load:
        # retrieve the dimensions of the PF-elements & PF-candidates to set the input/output dimension of the model
        if args.data == 'delphes':
            input_dim = len(features_delphes)
            num_classes = 6   # we have 6 classes/pids for delphes
        elif args.data == 'cms':
            input_dim = len(features_cms)
            num_classes = 9   # we have 9 classes/pids for cms (including taus)
        output_dim_p4 = len(target_p4)

        device = torch.device('cuda:0')
        outpath = osp.join(args.outpath, args.model_prefix)

        state_dict, model_kwargs, outpath = load_model(device, outpath, args.model_prefix, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

        model.to(device)

        model.eval()

    # evaluate on testing data..
    if args.load:
        epoch_on_plots = args.load_epoch
    else:
        epoch_on_plots = args.n_epochs - 1

    # make directories to hold testing plots
    make_directories_for_plots(outpath, 'test_data')

    # initialize the test dataset
    dataset_qcd = PFGraphDataset(args.dataset_qcd, args.data)
    test_dataset = torch.utils.data.Subset(dataset_qcd, np.arange(start=0, stop=args.n_test))

    # construct the test dataloader
    file_loader_test = make_file_loaders(test_dataset, num_files, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    # make predictions on the testing dataset
    if args.make_predictions:
        make_predictions(device, args.data, model, multi_gpu, file_loader_test, args.batch_size, args.batch_events, num_classes, outpath + '/test_data_plots/')

    # load the predictions and make plots (must have ran make_predictions before)
    if args.make_plots:
        make_plots(device, args.data, model, num_classes, outpath + '/test_data_plots/', args.target, epoch_on_plots, 'QCD')
