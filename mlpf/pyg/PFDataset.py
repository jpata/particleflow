from types import SimpleNamespace
from typing import List, Optional

import tensorflow_datasets as tfds
import torch
import torch.utils.data
import torch_geometric
from torch import Tensor
from torch_geometric.data import Batch, Data

from pyg.logger import _logger

import numpy as np


class TFDSDataSource:
    def __init__(self, ds, sort):
        self.ds = ds
        tmp = self.ds.dataset_info
        self.ds.dataset_info = SimpleNamespace()
        self.ds.dataset_info.name = tmp.name
        self.ds.dataset_info.features = tmp.features
        self.sort = sort
        self.rep = self.ds.__repr__()

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        records = self.ds.data_source.__getitems__(item)
        ret = [self.ds.dataset_info.features.deserialize_example_np(record, decoders=self.ds.decoders) for record in records]
        if len(item) == 1:
            ret = ret[0]

        # sorting the elements in pT descending order for the Mamba-based model
        if self.sort:
            sortidx = np.argsort(ret["X"][:, 1])[::-1]
            ret["X"] = ret["X"][sortidx]
            ret["ycand"] = ret["ycand"][sortidx]
            ret["ygen"] = ret["ygen"][sortidx]

        return ret

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.rep


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, num_samples=None, sort=False):
        """
        Args
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test" (if "valid" then will use "test")
            keys_to_get: any selection of ["X", "ygen", "ycand"] to retrieve
        """
        if split == "valid":
            split = "test"

        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = TFDSDataSource(builder.as_data_source(split=split), sort=sort)

        if num_samples:
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def __len__(self):
        return len(self.ds)


class PFDataLoader(torch.utils.data.DataLoader):
    """
    Copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
    because we need to implement our own Collater class to load the tensorflow_datasets (see below).
    """

    def __init__(
        self,
        dataset: PFDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        collate_fn = kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, keys_to_get, follow_batch=None, exclude_keys=None, pad_3d=True, pad_power_of_two=True):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.keys_to_get = keys_to_get
        self.pad_3d = pad_3d
        self.pad_power_of_two = False

    def __call__(self, inputs):
        num_samples_in_batch = len(inputs)
        elem_keys = self.keys_to_get

        batch = []
        for ev in range(num_samples_in_batch):
            batch.append(Data())
            for elem_key in elem_keys:
                batch[ev][elem_key] = Tensor(inputs[ev][elem_key])
            batch[ev]["batch"] = torch.tensor([ev] * len(inputs[ev][elem_key]))

        ret = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)

        if not self.pad_3d:
            return ret
        else:
            # pad to closest power of two
            if self.pad_power_of_two:
                sizes = [next_power_of_2(len(b.X)) for b in batch]
                max_size = max(sizes)
            else:
                max_size = None
            ret = {
                k: torch_geometric.utils.to_dense_batch(getattr(ret, k), ret.batch, max_num_nodes=max_size)
                for k in elem_keys
            }

            ret["mask"] = ret["X"][1]

            # remove the mask from each element
            for k in elem_keys:
                ret[k] = ret[k][0]

            ret = Batch(**ret)
        return ret


class InterleavedIterator(object):
    """Will combine DataLoaders of different lengths and batch sizes."""

    def __init__(self, data_loaders):
        self.idx = 0
        self.data_loaders = data_loaders
        self.data_loaders_iter = [iter(dl) for dl in data_loaders]
        max_loader_size = max([len(dl) for dl in data_loaders])

        self.loader_ds_indices = []
        for i in range(max_loader_size):
            for iloader, loader in enumerate(data_loaders):
                if i < len(loader):
                    self.loader_ds_indices.append(iloader)

        self.cur_index = 0
        self._len = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            iloader = self.loader_ds_indices[self.cur_index]
        except IndexError:
            self.cur_index = 0  # reset the curser index
            self.data_loaders_iter = [iter(dl) for dl in self.data_loaders]  # reset the loader
            raise StopIteration

        self.cur_index += 1
        return next(self.data_loaders_iter[iloader])

    def __len__(self):
        if self._len:
            return self._len
        else:
            # compute and cache the length
            len_ = 0
            for iloader in range(len(self.data_loaders_iter)):
                len_ += len(self.data_loaders_iter[iloader])
            self._len = len_
            return len_


def get_interleaved_dataloaders(world_size, rank, config, use_cuda, pad_3d, pad_power_of_two, use_ray):
    loaders = {}
    for split in ["train", "valid"]:  # build train, valid dataset and dataloaders
        loaders[split] = []
        # build dataloader for physical and gun samples seperately
        for type_ in config[f"{split}_dataset"][config["dataset"]]:  # will be "physical", "gun", "multiparticlegun"
            dataset = []
            for sample in config[f"{split}_dataset"][config["dataset"]][type_]["samples"]:
                version = config[f"{split}_dataset"][config["dataset"]][type_]["samples"][sample]["version"]

                ds = PFDataset(
                    config["data_dir"],
                    f"{sample}:{version}",
                    split,
                    num_samples=config[f"n{split}"],
                    sort=config["sort_data"],
                ).ds

                if (rank == 0) or (rank == "cpu"):
                    _logger.info(f"{split}_dataset: {sample}, {len(ds)}", color="blue")

                dataset.append(ds)
            dataset = torch.utils.data.ConcatDataset(dataset)

            if world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            # build dataloaders
            batch_size = config[f"{split}_dataset"][config["dataset"]][type_]["batch_size"] * config["gpu_batch_multiplier"]
            loader = PFDataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=Collater(["X", "ygen"], pad_3d=pad_3d, pad_power_of_two=pad_power_of_two),
                sampler=sampler,
                num_workers=config["num_workers"],
                prefetch_factor=config["prefetch_factor"],
                pin_memory=use_cuda,
                pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
            )

            if use_ray:
                import ray

                # prepare loader for distributed training, adds distributed sampler
                loader = ray.train.torch.prepare_data_loader(loader)

            loaders[split].append(loader)

        loaders[split] = InterleavedIterator(loaders[split])  # will interleave maximum of three dataloaders
    return loaders
