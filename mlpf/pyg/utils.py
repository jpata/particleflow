import json
import os
import os.path as osp
import pickle as pkl
import shutil
import sys
from collections.abc import Sequence

import matplotlib
import torch
from torch_geometric.data.data import BaseData

matplotlib.use("Agg")


# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
CLASS_LABELS = {
    "CMS": [0, 211, 130, 1, 2, 22, 11, 13, 15],
    "DELPHES": [0, 211, 130, 22, 11, 13],
    "CLIC": [0, 211, 130, 22, 11, 13],
}

CLASS_NAMES_LATEX = {
    "CMS": ["none", "Charged Hadron", "Neutral Hadron", "HFEM", "HFHAD", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$", r"$\tau$"],
    "DELPHES": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    "CLIC": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
}
CLASS_NAMES = {
    "CMS": ["none", "chhad", "nhad", "HFEM", "HFHAD", "gamma", "ele", "mu", "tau"],
    "DELPHES": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    "CLIC": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
}
CLASS_NAMES_CAPITALIZED = {
    "CMS": ["none", "Charged hadron", "Neutral hadron", "HFEM", "HFHAD", "Photon", "Electron", "Muon", "Tau"],
    "DELPHES": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    "CLIC": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
}

X_FEATURES = {
    "CMS": [
        "typ_idx",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
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
    ],
    "DELPHES": [
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
    ],
    "CLIC": [
        "type",
        "pt | et",
        "eta",
        "sin_phi",
        "cos_phi",
        "p | energy",
        "chi2 | position.x",
        "ndf | position.y",
        "dEdx | position.z",
        "dEdxError | iTheta",
        "radiusOfInnermostHit | energy_ecal",
        "tanLambda | energy_hcal",
        "D0 | energy_other",
        "omega | num_hits",
        "Z0 | sigma_x",
        "time | sigma_y",
        "Null | sigma_z",
    ],
}

Y_FEATURES = {
    "CMS": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "DELPHES": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "CLIC": ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy"],
}


def save_mlpf(args, outpath, mlpf, model_kwargs, mode="native"):

    if not osp.isdir(outpath):
        os.makedirs(outpath)

    else:  # if directory already exists
        if not args.overwrite:  # if not overwrite then exit
            print(f"model {args.prefix} already exists, please delete it")
            sys.exit(0)

        print("model already exists, deleting it")

        filelist = [f for f in os.listdir(outpath) if not f.endswith(".txt")]  # don't remove the newly created logs.txt
        for f in filelist:
            try:
                shutil.rmtree(os.path.join(outpath, f))
            except Exception:
                os.remove(os.path.join(outpath, f))

    with open(f"{outpath}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)
    print(f"Num of mlpf parameters: {num_mlpf_parameters}")

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "dataset": args.dataset,
                "n_train": args.n_train,
                "n_valid": args.n_valid,
                "n_test": args.n_test,
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "bs_mlpf": args.bs,
                "width": args.width,
                "embedding_dim": args.embedding_dim,
                "num_convs": args.num_convs,
                "space_dim": args.space_dim,
                "propagate_dim": args.propagate_dim,
                "k": args.nearest,
                "num_mlpf_parameters": num_mlpf_parameters,
                "mode": mode,
            },
            fp,
        )


def load_mlpf(device, outpath):

    PATH = outpath + "/best_epoch_weights.pth"
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

    return state_dict, model_kwargs


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


def make_file_loaders(world_size, data, num_files=1, num_workers=0, prefetch_factor=2):
    """
    This function uses native torch Dataloaders with a custom collate_fn that allows loading
    Data() objects from pt files in an efficient way. This is needed because pyg Dataloaders do
    not handle num_workers>0 since Batch() objects cannot be directly serialized using pkl.

    Args:
        world_size: number of gpus available.
        dataset: custom dataset.
        num_files: number of files to load with a single get() call.
        num_workers: number of workers to use for fetching files.
        prefetch_factor: number of files to fetch in advance.

    Returns:
        a torch iterable() that returns a list of 100 elements where each
        element is a tuple of size=num_files containing Data() objects.
    """

    pin_memory = world_size > 0

    # prevent a "too many open files" error
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
    torch.multiprocessing.set_sharing_strategy("file_system")

    return torch.utils.data.DataLoader(
        data,
        num_files,
        shuffle=False,
        num_workers=2,
        prefetch_factor=4,
        collate_fn=Collater(),
        pin_memory=pin_memory,
    )
