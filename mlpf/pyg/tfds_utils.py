from typing import List, Optional, Sequence, Union

import tensorflow_datasets as tfds
import torch
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


class DataLoader(torch.utils.data.DataLoader):
    """
    Copied from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
    because we need to implement our own Collater class (see below)
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
    """Based on the Collater found on torch_geometric docs we build our own"""

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
    def __init__(self, name="clic_edm_ttbar_pf:1.5.0", split="train"):
        builder = tfds.builder(name, data_dir="/pfvol/tensorflow_datasets/clic/clusters/")

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
        # return DataLoader(
        #     self.ds,
        #     batch_size=batch_size,
        #     collate_fn=Collater(),
        #     sampler=self.get_sampler(),
        # )

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
    def __init__(self, data_loaders):
        self.idx = 0
        self.data_loaders_iter = [iter(dl) for dl in data_loaders]
        max_loader_size = max([len(dl) for dl in data_loaders])

        # interleave loaders of different length
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
