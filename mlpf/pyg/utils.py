import json
import shutil
import os.path as osp
import sys
from glob import glob

import torch_geometric
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from torch_geometric.data.data import BaseData

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


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def save_model(args, model_fname, outpath, model_kwargs):
    # if osp.isdir(outpath):
    #     if args.overwrite:
    #         print("model {} already exists, deleting it".format(model_fname))
    #         shutil.rmtree(outpath)
    #     else:
    #         print("model {} already exists, please delete it".format(model_fname))
    #         sys.exit(0)
    # os.makedirs(outpath)

    with open(f'{outpath}/model_kwargs.pkl', 'wb') as f:  # dump model architecture
        pkl.dump(model_kwargs, f,  protocol=pkl.HIGHEST_PROTOCOL)

    with open(f'{outpath}/hyperparameters.json', 'w') as fp:  # dump hyperparameters
        json.dump({'data': args.data,
                   'target': args.target,
                   'n_train': args.n_train,
                   'n_valid': args.n_valid,
                   'n_test': args.n_test,
                   'n_epochs': args.n_epochs,
                   'lr': args.lr,
                   'batch_size': args.batch_size,
                   'alpha': args.alpha,
                   'nearest': args.nearest,
                   'num_convs': args.num_convs,
                   'space_dim': args.space_dim,
                   'propagate_dim': args.propagate_dim,
                   'embedding_dim': args.embedding_dim,
                   'hidden_dim1': args.hidden_dim1,
                   'hidden_dim2': args.hidden_dim2,
                   }, fp)


def load_model(device, outpath, model_directory, load_epoch):
    PATH = outpath + '/epoch_' + str(load_epoch) + '_weights.pth'

    print('Loading a previously trained model..')
    with open(outpath + '/model_kwargs.pkl', 'rb') as f:
        model_kwargs = pkl.load(f)

    state_dict = torch.load(PATH, map_location=device)

    # # if the model was trained using DataParallel then we do this
    # state_dict = torch.load(PATH, map_location=device)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove module.
    #     new_state_dict[name] = v
    # state_dict = new_state_dict

    return state_dict, model_kwargs, outpath


def make_plot_from_lists(title, xaxis, yaxis, save_as, X, Xlabel, X_save_as, outpath):
    """
    Given a list A of lists B, makes a scatter plot of each list B and saves it.
    """

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fig, ax = plt.subplots()
    for i, var in enumerate(X):
        ax.plot(range(len(var)), var, label=Xlabel[i])
    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    ax.legend(loc='best')
    ax.set_title(title, fontsize=20)
    plt.savefig(outpath + save_as + '.pdf')
    plt.close(fig)

    for i, var in enumerate(X):
        with open(outpath + X_save_as[i] + '.pkl', 'wb') as f:
            pkl.dump(var, f)


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


def make_directories_for_plots(outpath, tag):
    if not osp.isdir(f'{outpath}/{tag}'):
        os.makedirs(f'{outpath}/{tag}')
    if not osp.isdir(f'{outpath}/{tag}/plots'):
        os.makedirs(f'{outpath}/{tag}/plots')
    if not osp.isdir(f'{outpath}/{tag}/plots/resolution_plots'):
        os.makedirs(f'{outpath}/{tag}/plots/resolution_plots')
    if not osp.isdir(f'{outpath}/{tag}/plots/distribution_plots'):
        os.makedirs(f'{outpath}/{tag}/plots/distribution_plots')
    if not osp.isdir(f'{outpath}/{tag}/plots/multiplicity_plots'):
        os.makedirs(f'{outpath}/{tag}/plots/multiplicity_plots')
    if not osp.isdir(f'{outpath}/{tag}/plots/efficiency_plots'):
        os.makedirs(f'{outpath}/{tag}/plots/efficiency_plots')
    if not osp.isdir(f'{outpath}/{tag}/plots/confusion_matrix_plots'):
        os.makedirs(f'{outpath}/{tag}/plots/confusion_matrix_plots')


def make_directory_for_predictions(outpath, tag):
    if not osp.isdir(f'{outpath}/{tag}'):
        os.makedirs(f'{outpath}/{tag}')
    if not osp.isdir(f'{outpath}/{tag}/predictions'):
        os.makedirs(f'{outpath}/{tag}/predictions')


def define_regions(num_eta_regions=10, num_phi_regions=10, max_eta=5, min_eta=-5, max_phi=1.5, min_phi=-1.5):
    """
    Defines regions in (eta,phi) space to make bins within an event and build graphs within these bins.

    Returns
        regions: a list of tuples ~ (eta_tuples, phi_tuples) where eta_tuples is a tuple ~ (eta_min, eta_max) that defines the limits of a region and equivalenelty phi
    """
    eta_step = (max_eta - min_eta) / num_eta_regions
    phi_step = (max_phi - min_phi) / num_phi_regions

    tuples_eta = []
    for j in range(num_eta_regions):
        tuple = (min_eta + eta_step * (j), min_eta + eta_step * (j + 1))
        tuples_eta.append(tuple)

    tuples_phi = []
    for i in range(num_phi_regions):
        tuple = (min_phi + phi_step * (i), min_phi + phi_step * (i + 1))
        tuples_phi.append(tuple)

    # make regions
    regions = []
    for i in range(len(tuples_eta)):
        for j in range(len(tuples_phi)):
            regions.append((tuples_eta[i], tuples_phi[j]))

    return regions


def batch_event_into_regions(data, regions):
    """
    Given an event and a set of regions in (eta,phi) space, returns a binned version of the event.

    Args
        data: a Batch() object containing the event and its information
        regions: a tuple of tuples containing the defined regions to bin an event (see define_regions)

    Returns
        data: a modified Batch() object of based on data, where data.batch seperates the events in the different bins
    """

    x = None
    for region in range(len(regions)):
        in_region_msk = (data.x[:, 2] > regions[region][0][0]) & (data.x[:, 2] < regions[region][0][1]) & (torch.arcsin(data.x[:, 3]) > regions[region][1][0]) & (torch.arcsin(data.x[:, 3]) < regions[region][1][1])

        if in_region_msk.sum() != 0:  # if region is not empty
            if x == None:   # first iteration
                x = data.x[in_region_msk]
                ygen = data.ygen[in_region_msk]
                ygen_id = data.ygen_id[in_region_msk]
                ycand = data.ycand[in_region_msk]
                ycand_id = data.ycand_id[in_region_msk]
                batch = region + torch.zeros([len(data.x[in_region_msk])])    # assumes events were already fed one at a time (i.e. batch_size=1)
            else:
                x = torch.cat([x, data.x[in_region_msk]])
                ygen = torch.cat([ygen, data.ygen[in_region_msk]])
                ygen_id = torch.cat([ygen_id, data.ygen_id[in_region_msk]])
                ycand = torch.cat([ycand, data.ycand[in_region_msk]])
                ycand_id = torch.cat([ycand_id, data.ycand_id[in_region_msk]])
                batch = torch.cat([batch, region + torch.zeros([len(data.x[in_region_msk])])])    # assumes events were already fed one at a time (i.e. batch_size=1)

    data = Batch(x=x,
                 ygen=ygen,
                 ygen_id=ygen_id,
                 ycand=ycand,
                 ycand_id=ycand_id,
                 batch=batch.long(),
                 )
    return data


class Collater:
    """
    This function was copied from torch_geometric.loader.Dataloader() source code.
    Edits were made such that the function can collate samples as a list of tuples of Data() objects instead of Batch() objects.
    This is needed becase pyg Dataloaders do not handle num_workers>0 since Batch() objects cannot be directly serialized using pkl.
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


def make_file_loaders(dataset, num_files=1, num_workers=0, prefetch_factor=2):
    """
    This function is only one line, but it's worth explaining why it's needed and what it's doing.
    It uses native torch Dataloaders with a custom collate_fn that allows loading Data() objects from pt files in a fast way.
    This is needed becase pyg Dataloaders do not handle num_workers>0 since Batch() objects cannot be directly serialized using pkl.

    Args:
        dataset: custom dataset
        num_files: number of files to load with a single get() call
        num_workers: number of workers to use for fetching files
        prefetch_factor: number of files to fetch in advance

    Returns:
        a torch iterable() that returns a list of 100 elements, each element is a tuple of size=num_files containing Data() objects
    """

    return torch.utils.data.DataLoader(dataset, num_files, num_workers=num_workers, prefetch_factor=prefetch_factor, collate_fn=Collater(), pin_memory=True)
