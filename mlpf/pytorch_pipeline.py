from pytorch_delphes import parse_args
from pytorch_delphes import PFGraphDataset, dataloader_ttbar, dataloader_qcd
from pytorch_delphes import MLPF, training_loop, make_predictions, make_plots
from pytorch_delphes import make_directories_for_plots
from pytorch_delphes import get_model_fname, save_model, load_model

import torch
import torch_geometric

import mplhep as hep
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

    """
    e.g. to train locally run as:
    python -u pytorch_pipeline.py --title='' --overwrite --target='gen' --n_epochs=1 --n_train=1 --n_valid=1 --n_test=1 --batch_size=1

    e.g. to load and evaluate run as:
    python -u pytorch_pipeline.py --load --load_model='MLPF_gen_ntrain_1_nepochs_1_clf_reg' --load_epoch=0 --target='gen' --n_test=1 --batch_size=2
    """

    args = parse_args()

    # load the dataset (assumes the data exists as .pt files under args.dataset/processed)
    print('Loading the data..')
    full_dataset_ttbar = PFGraphDataset(args.dataset)
    full_dataset_qcd = PFGraphDataset(args.dataset_qcd)

    # construct Dataloaders to facilitate looping over batches
    print('Building dataloaders..')
    train_loader, valid_loader = dataloader_ttbar(full_dataset_ttbar, multi_gpu, args.n_train, args.n_valid, batch_size=args.batch_size)
    test_loader = dataloader_qcd(full_dataset_qcd, multi_gpu, args.n_test, batch_size=args.batch_size)

    # PF-elements
    input_dim = 12

    # PF-candidates
    output_dim_id = 6
    output_dim_p4 = 6

    if args.load:
        outpath = args.outpath + args.load_model
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.load_model, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

        if multi_gpu:
            model = torch_geometric.nn.DataParallel(model)

        model.to(device)

    else:
        print('Instantiating a model..')
        model_kwargs = {'input_dim': input_dim,
                        'output_dim_id': output_dim_id,
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
        model_fname = get_model_fname(args.dataset, model, args.n_train, args.n_epochs, args.target, args.alpha, args.title)
        outpath = osp.join(args.outpath, model_fname)

        if multi_gpu:
            print("Parallelizing the training..")
            model = torch_geometric.nn.DataParallel(model)

        model.to(device)

        save_model(args, model_fname, outpath, model_kwargs)

        print(model)
        print(model_fname)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.train()
        training_loop(device, model, multi_gpu,
                      train_loader, valid_loader,
                      args.n_epochs, args.patience,
                      optimizer, args.alpha, args.target,
                      output_dim_id, outpath)

    model.eval()

    # evaluate on testing data..
    make_directories_for_plots(outpath, 'test_data')
    if args.load:
        make_predictions(model, multi_gpu, test_loader, outpath + '/test_data_plots/', device, args.load_epoch)
        make_plots(model, test_loader, outpath + '/test_data_plots/', args.target, device, args.load_epoch, 'QCD')
    else:
        make_predictions(model, multi_gpu, test_loader, outpath + '/test_data_plots/', device, args.n_epochs)
        make_plots(model, test_loader, outpath + '/test_data_plots/', args.target, device, args.n_epochs, 'QCD')

    # # evaluate on training data..
    # make_directories_for_plots(outpath, 'train_data')
    # make_predictions(model, multi_gpu, train_loader, outpath + '/train_data_plots', args.target, device, args.n_epochs)
    # make_plots(model, train_loader, outpath + '/train_data_plots', args.target, device, args.n_epochs, 'TTbar')
    #
    # # evaluate on validation data..
    # make_directories_for_plots(outpath, 'valid_data')
    # make_predictions(model, multi_gpu, valid_loader, outpath + '/valid_data_plots', args.target, device, args.n_epochs)
    # make_plots(model, valid_loader, outpath + '/valid_data_plots', args.target, device, args.n_epochs, 'TTbar')
