from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from torch_geometric.data.data import BaseData
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


class Collater:
    """
    This function was copied from torch_geometric.loader.Dataloader() source code.
    Edits were made such that the function can collate samples as lists of tuples of Data() objects instead of Batch() objects.
    This is needed becase pyg Dataloaders do not handle num_workers>0.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return batch

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')


def make_file_loaders(dataset, num_files, num_workers, prefetch_factor):
    """
    This function is only one line, but it's worth explaining why it's needed and what it's doing.
    It uses native torch Dataloaders with a custom collate_fn that allows loading Data() objects from pt files in a fast way.
    It is needed because pyg Dataloaders do not handle num_workers>0.

    Args:
        dataset: custom dataset
        num_files: number of files to load with a single get() call
        num_workers: number of workers to use for fetching files
        prefetch_factor: number of files to fetch in advance

    Returns:
        a torch iterable() that returns a list of 100 elements, each element is a tuple of size=num_files containing Data() objects
    """

    return torch.utils.data.DataLoader(dataset, num_files, num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=Collater(), pin_memory=True)


if __name__ == "__main__":

    args = parse_args()

    # load the dataset (assumes the datafiles exist as .pt files under <args.dataset>/processed)
    dataset = PFGraphDataset(args.dataset, args.data)

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=args.n_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train, stop=args.n_train + args.n_valid))

    # construct file loaders
    num_files = 1
    file_loader_train = make_file_loaders(train_dataset, num_files, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    file_loader_valid = make_file_loaders(valid_dataset, num_files, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)

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
                      file_loader_train, file_loader_valid,
                      args.batch_size, args.batch_events,
                      args.n_epochs, args.patience,
                      optimizer, args.alpha, args.target,
                      num_classes, outpath)

    # model.eval()
    #
    # # evaluate on testing data..
    # if args.load:
    #     epoch_on_plots = args.load_epoch
    # else:
    #     epoch_on_plots = args.n_epochs - 1
    #
    # dataset_qcd = PFGraphDataset(args.dataset_qcd, args.data)
    #
    # make_directories_for_plots(outpath, 'test_data')
    # make_predictions(device, args.data, model, multi_gpu, dataset, args.n_test, args.batch_size, args.batch_events, num_classes, outpath + '/test_data_plots/')
    # make_plots(device, args.data, model, num_classes, outpath + '/test_data_plots/', args.target, epoch_on_plots, 'QCD')

    # dataset = PFGraphDataset('../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi', 'cms')
