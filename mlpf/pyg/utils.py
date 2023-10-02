import json
import os
import os.path as osp
import pickle as pkl
from typing import List, Optional, Sequence, Union

import tensorflow_datasets as tfds
import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
CLASS_LABELS = {
    "cms": [0, 211, 130, 1, 2, 22, 11, 13, 15],
    "delphes": [0, 211, 130, 22, 11, 13],
    "clic": [0, 211, 130, 22, 11, 13],
}

CLASS_NAMES_LATEX = {
    "cms": ["none", "Charged Hadron", "Neutral Hadron", "HFEM", "HFHAD", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$", r"$\tau$"],
    "delphes": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
    "clic": ["none", "Charged Hadron", "Neutral Hadron", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"],
}
CLASS_NAMES = {
    "cms": ["none", "chhad", "nhad", "HFEM", "HFHAD", "gamma", "ele", "mu", "tau"],
    "delphes": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
    "clic": ["none", "chhad", "nhad", "gamma", "ele", "mu"],
}
CLASS_NAMES_CAPITALIZED = {
    "cms": ["none", "Charged hadron", "Neutral hadron", "HFEM", "HFHAD", "Photon", "Electron", "Muon", "Tau"],
    "delphes": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
    "clic": ["none", "Charged hadron", "Neutral hadron", "Photon", "Electron", "Muon"],
}

X_FEATURES = {
    "cms": [
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
    "delphes": [
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
    "clic": [
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
    "cms": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "delphes": [
        "PDG",
        "charge",
        "pt",
        "eta",
        "sin_phi",
        "cos_phi",
        "energy",
    ],
    "clic": ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"],
}


def save_mlpf(args, mlpf, model_kwargs):
    if not osp.isdir(args.model_prefix):
        os.system(f"mkdir -p {args.model_prefix}")

    else:  # if directory already exists
        assert args.overwrite, f"model {args.model_prefix} already exists, please delete it"

        print("model already exists, deleting it")
        os.system(f"rm -rf {args.model_prefix}")
        os.system(f"mkdir -p {args.model_prefix}")

    with open(f"{args.model_prefix}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    num_mlpf_parameters = sum(p.numel() for p in mlpf.parameters() if p.requires_grad)

    with open(f"{args.model_prefix}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump({**{"Num of mlpf parameters": num_mlpf_parameters}, **vars(args)}, fp)


class DataLoader(torch.utils.data.DataLoader):
    """
    Copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
    because we need to implement our own Collater class to load the tensorflow_datasets (see below).
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(follow_batch, exclude_keys),
            **kwargs,
        )


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, inputs):
        num_samples_in_batch = len(inputs)
        elem_keys = list(inputs[0].keys())

        batch = []
        for ev in range(num_samples_in_batch):
            batch.append(Data())
            for elem_key in elem_keys:
                batch[ev][elem_key] = Tensor(inputs[ev][elem_key])
            batch[ev]["batch"] = torch.tensor([ev] * len(inputs[ev][elem_key]))

        elem = batch[0]

        if isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)

        raise TypeError(f"DataLoader found invalid type: {type(elem)}")


class Dataset:
    """Will combine DataLoaders of different lengths and batch sizes."""

    def __init__(self, data_dir, name="clic_edm_ttbar_pf:1.5.0", split="train"):
        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = builder.as_data_source(split=split)

        # to prevent a warning from tfds about accessing sequences of indices
        self.ds.__class__.__getitems__ = my_getitem

    def get_sampler(self):
        sampler = torch.utils.data.RandomSampler(self.ds)
        return sampler

    def get_distributed_sampler(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.ds)
        return sampler

    def get_loader(self, batch_size, world_size, num_workers=2, prefetch_factor=4):
        if world_size > 1:
            return DataLoader(
                self.ds,
                batch_size=batch_size,
                collate_fn=Collater(),
                sampler=self.get_distributed_sampler(),
            )
        else:
            return DataLoader(
                self.ds,
                batch_size=batch_size,
                collate_fn=Collater(),
                sampler=self.get_sampler(),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
            )

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.ds.__repr__()


def my_getitem(self, vals):
    records = self.data_source.__getitems__(vals)
    return [self.dataset_info.features.deserialize_example_np(record, decoders=self.decoders) for record in records]


class InterleavedIterator(object):
    """Will combine DataLoaders of different lengths and batch sizes."""

    def __init__(self, data_loaders):
        self.idx = 0
        self.data_loaders_iter = [iter(dl) for dl in data_loaders]
        max_loader_size = max([len(dl) for dl in data_loaders])

        self.loader_ds_indices = []
        for i in range(max_loader_size):
            for iloader, loader in enumerate(data_loaders):
                if i < len(loader):
                    self.loader_ds_indices.append(iloader)

        self.cur_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        iloader = self.loader_ds_indices[self.cur_index]
        self.cur_index += 1
        return next(self.data_loaders_iter[iloader])
