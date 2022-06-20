from pyg import parse_args
from pyg import PFGraphDataset, one_hot_embedding
from pyg import MLPF, training_loop, make_predictions, make_plots
from pyg import save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg import make_file_loaders

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Data, Batch

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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

# define the global base device
if torch.cuda.device_count():
    device = 0
    device_cuda = torch.device('cuda:0')
else:
    device = 'cpu'
    device_cuda = 'cpu'
multi_gpu = torch.cuda.device_count() > 1


def setup(rank, world_size):
    """
    Necessary setup function that sets up environment variables and initializes the process group to training using DistributedDataParallel (DDP).
    DDP relies on c10d ProcessGroup for communications. Hence, applications must create ProcessGroup instances before constructing DDP.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)   # should be faster for DistributedDataParallel on gpus


def cleanup():
    """
    Necessary function that destroys the spawned process group at the end.
    """

    dist.destroy_process_group()


def run_demo(demo_fn, world_size, args, dataset, model, num_classes, outpath):
    """
    Necessary function that spawns a process group of size=world_size processes to run a demo_fn on each gpu device that will be indexed by 'rank'.

    Args:
    demo_fn: function you wish to run on each gpu
    world_size: number of gpus available
    """

    mp.set_start_method('forkserver')

    mp.spawn(demo_fn,
             args=(world_size, args, dataset, model, num_classes, outpath),
             nprocs=world_size,
             join=True,
             )


def train_ddp(rank, world_size, args, dataset, model, num_classes, outpath):
    """
    A train_ddp() function that will be passed as a demo_fn to run_demo() to perform training over multiple gpus using DDP.

    It divides and distributes the training dataset appropriately, copies the model, and wraps the model with DDP on each device
    to allow synching of gradients, and finally, invokes the training_loop() to run synchronized training among devices.
    """

    print(f"Running training on rank {rank}: {torch.cuda.get_device_name(rank)}")

    setup(rank, world_size)

    # give each gpu a subset of the data
    hyper_train = int(args.n_train / world_size)
    hyper_valid = int(args.n_valid / world_size)

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=rank * hyper_train, stop=(rank + 1) * hyper_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train + rank * hyper_valid, stop=args.n_train + (rank + 1) * hyper_valid))

    # construct file loaders
    file_loader_train = make_file_loaders(world_size, train_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    file_loader_valid = make_file_loaders(world_size, valid_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    # create model and move it to GPU with id rank
    print(f'Copying the model on rank {rank}..')
    model = model.to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

    training_loop(rank, args.data, ddp_model, file_loader_train, file_loader_valid,
                  args.batch_size, args.n_epochs, args.patience,
                  optimizer, args.alpha, args.target, num_classes, outpath)

    cleanup()


def train(device, world_size, args, dataset, model, num_classes, outpath):
    """
    A train() function that will setup the training dataset and starts a training_loop on a single device (cuda or cpu).
    """

    if device == 'cpu':
        print(f"Running training on cpu")
    else:
        print(f"Running training on: {torch.cuda.get_device_name(device)}")

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=args.n_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train, stop=args.n_train + args.n_valid))

    # construct file loaders
    file_loader_train = make_file_loaders(world_size, train_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    file_loader_valid = make_file_loaders(world_size, valid_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

    # create model and move it to GPU with id rank
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_loop(device, args.data, model, file_loader_train, file_loader_valid,
                  args.batch_size, args.n_epochs, args.patience,
                  optimizer, args.alpha, args.target, num_classes, outpath)


if __name__ == "__main__":

    args = parse_args()

    world_size = torch.cuda.device_count()

    # load the dataset (assumes the datafiles exist as .pt files under <args.dataset>/processed)
    dataset = PFGraphDataset(args.dataset, args.data)

    # retrieve the dimensions of the PF-elements & PF-candidates to set the input/output dimension of the model
    if args.data == 'delphes':
        input_dim = len(features_delphes)
        num_classes = 6   # we have 6 classes/pids for delphes
    elif args.data == 'cms':
        input_dim = len(features_cms)
        num_classes = 9   # we have 9 classes/pids for cms (including taus)
    output_dim_p4 = len(target_p4)

    outpath = osp.join(args.outpath, args.model_prefix)

    if args.load:  # load a pre-trained specified model
        state_dict, model_kwargs, outpath = load_model(device_cuda, outpath, args.model_prefix, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)
        # model = torch.jit.script(model)

    else:   # instantiates and train a model
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

        # save model_kwargs and hyperparameters
        save_model(args, args.model_prefix, outpath, model_kwargs)

        print(model)
        print(args.model_prefix)

        print("Training over {} epochs".format(args.n_epochs))

        # run the training using DDP if more than one gpu is available
        if world_size >= 2:
            run_demo(train_ddp, world_size, args, dataset, model, num_classes, outpath)
        else:
            train(device, world_size, args, dataset, model, num_classes, outpath)

    # run the inference
    if args.make_predictions:
        if not osp.isdir(f'{outpath}/test_data'):
            os.makedirs(f'{outpath}/test_data')

        # load the dataset (assumes the datafiles exist as .pt files under <args.dataset>/processed)
        dataset_qcd = PFGraphDataset(args.dataset_qcd, args.data)
        test_dataset = torch.utils.data.Subset(dataset_qcd, np.arange(start=0, stop=args.n_test))

        # construct file loaders
        file_loader_test = make_file_loaders(world_size, test_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

        if multi_gpu:
            model = torch_geometric.nn.DataParallel(model)

        model.to(device)

        model.eval()

        # make predictions on the testing dataset (note: for DataParallel, batch size will be distributed among the number of gpus available)
        if world_size >= 2:
            batch_size = args.batch_size * world_size
        else:
            batch_size = args.batch_size
        make_predictions(device, args.data, model, multi_gpu, file_loader_test, batch_size, num_classes, outpath + '/test_data/')

    # load the predictions and make plots (must have ran make_predictions before)
    if args.make_plots:
        make_directories_for_plots(outpath, 'test_data')

        if args.load:
            epoch_on_plots = args.load_epoch
        else:
            epoch_on_plots = args.n_epochs - 1

        make_plots(args.data, num_classes, outpath + '/test_data/', args.target, epoch_on_plots, 'QCD')
