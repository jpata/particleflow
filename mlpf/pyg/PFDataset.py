from typing import List, Optional

import tensorflow_datasets as tfds
import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, keys_to_get, num_samples=None):
        """
        Args
            dataset: "cms", "clic", or "delphes"
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test
        """

        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = builder.as_data_source(split=split)

        self.keys_to_get = keys_to_get

        if num_samples:
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def get_sampler(self):
        sampler = torch.utils.data.RandomSampler(self.ds)
        return sampler

    def get_distributed_sampler(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.ds)
        return sampler

    def get_loader(self, batch_size, world_size, is_distributed=False, num_workers=None, prefetch_factor=2, flag="train"):
        if (world_size > 1) and is_distributed:  # torch.nn.parallel.DistributedDataParallel
            if flag == "valid":
                sampler = self.get_sampler()  # validation is done a on single machine
            else:
                sampler = self.get_distributed_sampler()

            # TODO: add num_workers>0 for DDP
            return DataLoader(
                self.ds,
                batch_size=batch_size,
                collate_fn=Collater(self.keys_to_get),
                sampler=sampler,
            )

        elif (world_size > 1) and not is_distributed:  # torch_geometric.nn.data_parallel
            sampler = self.get_sampler()

            if num_workers is not None:
                return DataLoader(
                    self.ds,
                    batch_size=batch_size,
                    collate_fn=Collater(self.keys_to_get, return_lists=True),
                    sampler=sampler,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                )
            else:
                return DataLoader(
                    self.ds,
                    batch_size=batch_size,
                    collate_fn=Collater(self.keys_to_get, return_lists=True),
                    sampler=sampler,
                )
        else:  # single-gpu and cpu
            sampler = self.get_sampler()
            if num_workers is not None:
                return DataLoader(
                    self.ds,
                    batch_size=batch_size,
                    collate_fn=Collater(self.keys_to_get),
                    sampler=sampler,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                )
            else:
                return DataLoader(
                    self.ds,
                    batch_size=batch_size,
                    collate_fn=Collater(self.keys_to_get),
                    sampler=sampler,
                )

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.ds.__repr__()


class DataLoader(torch.utils.data.DataLoader):
    """
    Copied from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
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

        super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn, **kwargs)


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, keys_to_get, return_lists=False, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.keys_to_get = keys_to_get
        self.return_lists = return_lists

    def __call__(self, inputs):
        num_samples_in_batch = len(inputs)
        elem_keys = self.keys_to_get

        batch = []
        for ev in range(num_samples_in_batch):
            batch.append(Data())
            for elem_key in elem_keys:
                batch[ev][elem_key] = Tensor(inputs[ev][elem_key])
            batch[ev]["batch"] = torch.tensor([ev] * len(inputs[ev][elem_key]))

        if self.return_lists:
            return batch
        else:
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)


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
