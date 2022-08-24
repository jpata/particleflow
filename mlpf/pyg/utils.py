import json
import os
import os.path as osp
import pickle as pkl
import shutil
import sys
from collections.abc import Sequence

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataListLoader, DataLoader

matplotlib.use("Agg")

features_delphes = [
    "Track|cluster",
    "$p_{T}|E_{T}$",
    r"$\eta$",
    r"$Sin(\phi)$",
    r"$Cos(\phi)$",
    "P|E",
    r"$\eta_\mathrm{out}|E_{em}$",
    r"$Sin(\(phi)_\mathrm{out}|E_{had}$",
    r"$Cos(\phi)_\mathrm{out}|E_{had}$",
    "charge",
    "is_gen_mu",
    "is_gen_el",
]

features_cms = [
    "typ_idx",
    "pt",
    "eta",
    "phi",
    "e",
    "layer",
    "depth",
    "charge",
    "trajpoint",
    "eta_ecal",
    "phi_ecal",
    "eta_hcal",
    "phi_hcal",
    "muon_dt_hits",
    "muon_csc_hits",
    "muon_type",
    "px",
    "py",
    "pz",
    "deltap",
    "sigmadeltap",
    "gsf_electronseed_trkorecal",
    "gsf_electronseed_dnn1",
    "gsf_electronseed_dnn2",
    "gsf_electronseed_dnn3",
    "gsf_electronseed_dnn4",
    "gsf_electronseed_dnn5",
    "num_hits",
    "cluster_flags",
    "corr_energy",
    "corr_energy_err",
    "vx",
    "vy",
    "vz",
    "pterror",
    "etaerror",
    "phierror",
    "lambd",
    "lambdaerror",
    "theta",
    "thetaerror",
]

target_p4 = [
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
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

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        if not args.overwrite:  # if not overwrite then exit
            print(f"model {model_fname} already exists, please delete it")
            sys.exit(0)

        print(f"model {model_fname} already exists, deleting it")

        filelist = [f for f in os.listdir(outpath) if not f.endswith(".txt")]  # don't remove the newly created logs.txt
        for f in filelist:
            try:
                os.remove(os.path.join(outpath, f))
            except IsADirectoryError:
                shutil.rmtree(os.path.join(outpath, f))

    with open(f"{outpath}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "data": args.data,
                "target": args.target,
                "n_train": args.n_train,
                "n_valid": args.n_valid,
                "n_test": args.n_test,
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "alpha": args.alpha,
                "nearest": args.nearest,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "embedding_dim": args.embedding_dim,
                "hidden_dim1": args.hidden_dim1,
                "hidden_dim2": args.hidden_dim2,
            },
            fp,
        )


def load_model(device, outpath, model_directory, load_epoch):
    if load_epoch == -1:
        PATH = outpath + "/best_epoch_weights.pth"
    else:
        PATH = outpath + "/epoch_" + str(load_epoch) + "_weights.pth"

    print("Loading a previously trained model..")
    with open(outpath + "/model_kwargs.pkl", "rb") as f:
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
    ax.legend(loc="best")
    ax.set_title(title, fontsize=20)
    plt.savefig(outpath + save_as + ".pdf")
    plt.close(fig)

    for i, var in enumerate(X):
        with open(outpath + X_save_as[i] + ".pkl", "wb") as f:
            pkl.dump(var, f)


def define_regions(num_eta_regions=10, num_phi_regions=10, max_eta=5, min_eta=-5, max_phi=1.5, min_phi=-1.5):
    """
    Defines regions in (eta,phi) space to make bins within an event and build graphs within these bins.

    Returns
        regions: a list of tuples ~ (eta_tuples, phi_tuples) where eta_tuples is a tuple ~ (eta_min, eta_max)
        that defines the limits of a region and equivalenelty phi
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
        in_region_msk = (
            (data.x[:, 2] > regions[region][0][0])
            & (data.x[:, 2] < regions[region][0][1])
            & (torch.arcsin(data.x[:, 3]) > regions[region][1][0])
            & (torch.arcsin(data.x[:, 3]) < regions[region][1][1])
        )

        if in_region_msk.sum() != 0:  # if region is not empty
            if x is None:  # first iteration
                x = data.x[in_region_msk]
                ygen = data.ygen[in_region_msk]
                ygen_id = data.ygen_id[in_region_msk]
                ycand = data.ycand[in_region_msk]
                ycand_id = data.ycand_id[in_region_msk]
                batch = region + torch.zeros(
                    [len(data.x[in_region_msk])]
                )  # assumes events were already fed one at a time (i.e. batch_size=1)
            else:
                x = torch.cat([x, data.x[in_region_msk]])
                ygen = torch.cat([ygen, data.ygen[in_region_msk]])
                ygen_id = torch.cat([ygen_id, data.ygen_id[in_region_msk]])
                ycand = torch.cat([ycand, data.ycand[in_region_msk]])
                ycand_id = torch.cat([ycand_id, data.ycand_id[in_region_msk]])
                batch = torch.cat(
                    [batch, region + torch.zeros([len(data.x[in_region_msk])])]
                )  # assumes events were already fed one at a time (i.e. batch_size=1)

    data = Batch(
        x=x,
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
    Edits were made such that the function can collate samples as a list of tuples
    of Data() objects instead of Batch() objects. This is needed becase pyg Dataloaders
    do not handle num_workers>0 since Batch() objects cannot be directly serialized using pkl.
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return batch

        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")


def make_file_loaders(world_size, dataset, num_files=1, num_workers=0, prefetch_factor=2):
    """
    This function is only one line, but it's worth explaining why it's needed
    and what it's doing. It uses native torch Dataloaders with a custom collate_fn
    that allows loading Data() objects from pt files in a fast way. This is needed
    becase pyg Dataloaders do not handle num_workers>0 since Batch() objects
    cannot be directly serialized using pkl.

    Args:
        world_size: number of gpus available
        dataset: custom dataset
        num_files: number of files to load with a single get() call
        num_workers: number of workers to use for fetching files
        prefetch_factor: number of files to fetch in advance

    Returns:
        a torch iterable() that returns a list of 100 elements,
        each element is a tuple of size=num_files containing Data() objects
    """
    if world_size > 0:
        return torch.utils.data.DataLoader(
            dataset,
            num_files,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=Collater(),
            pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            num_files,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=Collater(),
            pin_memory=False,
        )


def dataloader_ttbar(train_dataset, valid_dataset, batch_size):
    """
    Builds training and validation dataloaders from a physics dataset for conveninet ML training
    Args:
        train_dataset: a PFGraphDataset dataset that is a list of lists that contain Data() objects
        valid_dataset: a PFGraphDataset dataset that is a list of lists that contain Data() objects
    Returns:
        train_loader: a pyg iterable DataLoader() that contains Batch objects for training
        valid_loader: a pyg iterable DataLoader() that contains Batch objects for validation
    """

    # preprocessing the train_dataset in a good format for passing correct batches of events to the GNN
    train_data = []
    for data in train_dataset:
        train_data = train_data + data

    # preprocessing the valid_dataset in a good format for passing correct batches of events to the GNN
    valid_data = []
    for data in valid_dataset:
        valid_data = valid_data + data

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def dataloader_qcd(multi_gpu, test_dataset, batch_size):
    """
    Builds a testing dataloader from a physics dataset for conveninet ML training
    Args:
        test_dataset: a PFGraphDataset dataset that is a list of lists that contain Data() objects
    Returns:
        test_loader: a pyg iterable DataLoader() that contains Batch objects for testing
    """

    # preprocessing the test_dataset in a good format for passing correct batches of events to the GNN
    test_data = []
    for data in test_dataset:
        test_data = test_data + data

    if not multi_gpu:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    else:
        test_loader = DataListLoader(test_data, batch_size=batch_size, shuffle=True)

    return test_loader
