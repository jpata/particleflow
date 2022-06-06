from pyg import parse_args
from pyg import MLPF, training_loop, make_predictions, make_plots
from pyg import get_model_fname, save_model, load_model, make_directories_for_plots
from pyg import features_delphes, features_cms, target_p4
from pyg.dataset import PFGraphDataset

import torch
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader

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
    e.g. to train on delphes locally run as:
    python -u pyg_pipeline.py --data delphes --title='delphes' --overwrite --n_epochs=20 --dataset='../data/delphes/pythia8_ttbar' --dataset_qcd='../data/delphes/pythia8_ttbar'

    e.g. to train on cms locally run as:
    python -u pyg_pipeline.py --data cms --title='cms' --overwrite --n_epochs=1 --dataset='../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi' --dataset_qcd='../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi'

    e.g. to load and evaluate on delphes:
    python -u pyg_pipeline.py --data delphes --load --load_model='MLPF_delphes_gen_1files_20epochs_delphes' --load_epoch=19 --dataset='../data/delphes/pythia8_ttbar' --dataset_qcd='../data/delphes/pythia8_ttbar'

    e.g. to load and evaluate on cms:
    python -u pyg_pipeline.py --data cms --load --load_model='MLPF_cms_gen_1files_1epochs_cms' --load_epoch=0 --dataset='../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi' --dataset_qcd='../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi'
    """

    args = parse_args()

    # load the dataset (assumes the data exists as .pt files under args.dataset/processed)
    print(f'Loading the {args.data} data..')
    dataset = PFGraphDataset(args.dataset, args.data)
    dataset_qcd = PFGraphDataset(args.dataset_qcd, args.data)

    # retrieve the dimensions of the PF-elements & PF-candidates
    if args.data == 'delphes':
        input_dim = len(features_delphes)
        output_dim_id = 6   # we have 6 classes/pids for cms
    elif args.data == 'cms':
        input_dim = len(features_cms)
        output_dim_id = 9   # we have 8 classes/pids for cms
    output_dim_p4 = len(target_p4)

    if args.load:
        outpath = args.outpath + args.load_model
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.load_model, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

        if multi_gpu:
            model = torch_geometric.nn.DataParallel(model)

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
        training_loop(device, args.data, model, multi_gpu,
                      dataset, args.n_train, args.n_valid,
                      args.batch_size, args.batch_events,
                      args.n_epochs, args.patience,
                      optimizer, args.alpha, args.target,
                      output_dim_id, outpath)

    model.eval()

    # evaluate on testing data..
    if args.load:
        epoch_on_plots = args.load_epoch
    else:
        epoch_on_plots = args.n_epochs - 1

    make_directories_for_plots(outpath, 'test_data')
    make_predictions(device, args.data, model, multi_gpu, dataset, args.n_test, args.batch_size, args.batch_events, output_dim_id, outpath + '/test_data_plots/')
    make_plots(device, args.data, model, output_dim_id, outpath + '/test_data_plots/', args.target, epoch_on_plots, 'QCD')
