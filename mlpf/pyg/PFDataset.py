from types import SimpleNamespace
from typing import List, Optional

import tensorflow_datasets as tfds
import torch
import torch.utils.data
import torch_geometric
from torch import Tensor
from torch_geometric.data import Batch, Data


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, num_samples=None):
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

        self.ds = builder.as_data_source(split=split)

        # to prevent a warning from tfds about accessing sequences of indices
        self.ds.__class__.__getitems__ = my_getitem

        # to make dataset_info pickable
        tmp = self.ds.dataset_info

        self.ds.dataset_info = SimpleNamespace()
        self.ds.dataset_info.name = tmp.name
        self.ds.dataset_info.features = tmp.features
        self.rep = self.ds.__repr__()

        if num_samples:
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.rep


def my_getitem(self, vals):
    records = self.data_source.__getitems__(vals)
    return [self.dataset_info.features.deserialize_example_np(record, decoders=self.decoders) for record in records]


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


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, keys_to_get, follow_batch=None, exclude_keys=None, pad_bin_size=640, pad_3d=True):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.keys_to_get = keys_to_get
        self.pad_bin_size = pad_bin_size
        self.pad_3d = pad_3d

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
            ret = {k: torch_geometric.utils.to_dense_batch(getattr(ret, k), ret.batch) for k in elem_keys}

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
