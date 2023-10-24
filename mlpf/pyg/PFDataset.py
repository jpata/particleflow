from typing import List, Optional

import tensorflow_datasets as tfds
import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, keys_to_get, num_samples=None):
        """
        Args
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test
            keys_to_get: any selection of ["X", "ygen", "ycand"] to retrieve
        """

        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = builder.as_data_source(split=split)

        # to prevent a warning from tfds about accessing sequences of indices
        self.ds.__class__.__getitems__ = my_getitem

        # to make dataset_info pickable
        tmp = self.ds.dataset_info
        self.ds.dataset_info = None
        from types import SimpleNamespace

        self.ds.dataset_info = SimpleNamespace()
        self.ds.dataset_info.features = tmp.features

        # any selection of ["X", "ygen", "ycand"] to retrieve
        self.keys_to_get = keys_to_get

        if num_samples:
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def get_sampler(self):
        sampler = torch.utils.data.RandomSampler(self.ds)
        return sampler

    def get_distributed_sampler(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.ds)
        return sampler

    def get_loader(self, batch_size, world_size, num_workers=0, prefetch_factor=None):
        if (num_workers > 0) and (prefetch_factor is None):
            prefetch_factor = 2  # default prefetch_factor when num_workers>0

        if world_size > 1:
            sampler = self.get_distributed_sampler()
        else:
            sampler = self.get_sampler()

        return DataLoader(
            self.ds,
            batch_size=batch_size,
            collate_fn=Collater(self.keys_to_get),
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.ds.__repr__()


class DataLoader(torch.utils.data.DataLoader):
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


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, keys_to_get, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.keys_to_get = keys_to_get

    def __call__(self, inputs):
        num_samples_in_batch = len(inputs)
        elem_keys = self.keys_to_get

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


def my_getitem(self, vals):
    # print(
    #     "reading dataset {}:{} from disk in slice {}, total={}".format(self.dataset_info.name, self.split, vals, len(self))
    # )
    records = self.data_source.__getitems__(vals)
    return [self.dataset_info.features.deserialize_example_np(record, decoders=self.decoders) for record in records]


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
        len_ = 0
        for iloader in range(len(self.data_loaders_iter)):
            len_ += len(self.data_loaders_iter[iloader])

        return len_
