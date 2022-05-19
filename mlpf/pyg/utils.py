import json
import shutil
import os.path as osp
import sys
from glob import glob
import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Data, Batch
import torch
import mplhep as hep
import matplotlib.pyplot as plt
import os
import pickle as pkl
import math
import time
import tqdm
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")


pid_to_class_delphes = {0: 'null',
                        1: 'chhadron',
                        2: 'nhadron',
                        3: 'photon',
                        4: 'ele',
                        5: 'mu',
                        }

pid_to_class_cms = {0: 'null',
                    1: 'HFEM',
                    2: 'HFHAD',
                    3: 'ele',
                    4: 'mu',
                    5: 'photon',
                    6: 'nhadron',
                    7: 'chhadron',
                    8: 'tau',
                    }


features_delphes = ["Track|cluster", "$p_{T}|E_{T}$", r"$\eta$", r'$Sin(\phi)$', r'$Cos(\phi)$',
                    "P|E", r"$\eta_\mathrm{out}|E_{em}$", r"$Sin(\(phi)_\mathrm{out}|E_{had}$", r"$Cos(\phi)_\mathrm{out}|E_{had}$",
                    "charge", "is_gen_mu", "is_gen_el"]

features_cms = [
    "typ_idx", "pt", "eta", "phi", "e",
    "layer", "depth", "charge", "trajpoint",
    "eta_ecal", "phi_ecal", "eta_hcal", "phi_hcal", "muon_dt_hits", "muon_csc_hits", "muon_type",
    "px", "py", "pz", "deltap", "sigmadeltap",
    "gsf_electronseed_trkorecal",
    "gsf_electronseed_dnn1",
    "gsf_electronseed_dnn2",
    "gsf_electronseed_dnn3",
    "gsf_electronseed_dnn4",
    "gsf_electronseed_dnn5",
    "num_hits", "cluster_flags", "corr_energy",
    "corr_energy_err", "vx", "vy", "vz", "pterror", "etaerror", "phierror", "lambd", "lambdaerror", "theta", "thetaerror"
]

target_p4 = [
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "e",
]


def dataloader_ttbar(full_dataset, multi_gpu, n_train, n_valid, batch_size):
    """
    Builds training and validation dataloaders from a physics dataset for conveninet ML training

    Args:
        full_dataset: a delphes dataset that is a list of lists that contain Data() objects
        multi_gpu: boolean for multigpu batching
        n_train: number of files to use for training
        n_valid: number of files to use for validation

    Returns:
        train_loader: a pyg iterable DataLoader() that contains Batch objects for training
        valid_loader: a pyg iterable DataLoader() that contains Batch objects for validation
    """

    train_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_train))
    valid_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=n_train, stop=n_train + n_valid))

    # preprocessing the train_dataset in a good format for passing correct batches of events to the GNN
    train_data = []
    for i in range(len(train_dataset)):
        train_data = train_data + train_dataset[i]

    # preprocessing the valid_dataset in a good format for passing correct batches of events to the GNN
    valid_data = []
    for i in range(len(valid_dataset)):
        valid_data = valid_data + valid_dataset[i]

    if not multi_gpu:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataListLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataListLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def dataloader_qcd(full_dataset, multi_gpu, n_test, batch_size):
    """
    Builds a testing dataloader from a physics dataset for conveninet ML training

    Args:
        full_dataset: a delphes dataset that is a list of lists that contain Data() objects
        multi_gpu: boolean for multigpu batching
        n_test: number of files to use for testing

    Returns:
        test_loader: a pyg iterable DataLoader() that contains Batch objects for testing
    """

    test_dataset = torch.utils.data.Subset(full_dataset, np.arange(start=0, stop=n_test))

    # preprocessing the test_dataset in a good format for passing correct batches of events to the GNN
    test_data = []
    for i in range(len(test_dataset)):
        test_data = test_data + test_dataset[i]

    if not multi_gpu:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    else:
        test_loader = DataListLoader(test_data, batch_size=batch_size, shuffle=True)

    return test_loader


def get_model_fname(dataset, model, data, n_train, n_epochs, target, title):
    """
    Get a unique directory name for the model
    """

    model_name = type(model).__name__
    model_fname = '{}_{}_{}_{}files_{}epochs'.format(
        model_name,
        data,
        target,
        n_train,
        n_epochs
    )

    if title:
        model_fname = model_fname + '_' + title

    return model_fname


def save_model(args, model_fname, outpath, model_kwargs):
    if osp.isdir(outpath):
        print(args.load_model)
        if args.overwrite:
            print("model {} already exists, deleting it".format(model_fname))
            shutil.rmtree(outpath)
        else:
            print("model {} already exists, please delete it".format(model_fname))
            sys.exit(0)
    os.makedirs(outpath)

    with open(f'{outpath}/model_kwargs.pkl', 'wb') as f:  # dump model architecture
        pkl.dump(model_kwargs, f,  protocol=pkl.HIGHEST_PROTOCOL)

    with open(f'{outpath}/hyperparameters.json', 'w') as fp:  # dump hyperparameters
        json.dump({'lr': args.lr, 'batch_size': args.batch_size, 'alpha': args.alpha, 'nearest': args.nearest}, fp)


def load_model(device, outpath, model_directory, load_epoch):
    PATH = outpath + '/epoch_' + str(load_epoch) + '_weights.pth'

    print('Loading a previously trained model..')
    with open(outpath + '/model_kwargs.pkl', 'rb') as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(PATH, map_location=device)

    if "DataParallel" in model_directory:   # if the model was trained using DataParallel then we do this
        state_dict = torch.load(PATH, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        state_dict = new_state_dict

    return state_dict, model_kwargs, outpath


def make_plot(X, label, xlabel, ylabel, outpath, save_as):
    """
    Given a list X, makes a scatter plot of it and saves it
    """
    plt.style.use(hep.style.ROOT)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fig, ax = plt.subplots()
    ax.plot(range(len(X)), X, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    plt.savefig(outpath + save_as + '.png')
    plt.close(fig)

    with open(outpath + save_as + '.pkl', 'wb') as f:
        pkl.dump(X, f)


def make_directories_for_plots(outpath, tag):
    if not osp.isdir(f'{outpath}/{tag}_plots'):
        os.makedirs(f'{outpath}/{tag}_plots')
    if not osp.isdir(f'{outpath}/{tag}_plots/resolution_plots'):
        os.makedirs(f'{outpath}/{tag}_plots/resolution_plots')
    if not osp.isdir(f'{outpath}/{tag}_plots/distribution_plots'):
        os.makedirs(f'{outpath}/{tag}_plots/distribution_plots')
    if not osp.isdir(f'{outpath}/{tag}_plots/multiplicity_plots'):
        os.makedirs(f'{outpath}/{tag}_plots/multiplicity_plots')
    if not osp.isdir(f'{outpath}/{tag}_plots/efficiency_plots'):
        os.makedirs(f'{outpath}/{tag}_plots/efficiency_plots')
    if not osp.isdir(f'{outpath}/{tag}_plots/confusion_matrix_plots'):
        os.makedirs(f'{outpath}/{tag}_plots/confusion_matrix_plots')
