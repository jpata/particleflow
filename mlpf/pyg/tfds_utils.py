from typing import List, Optional, Sequence, Union

import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
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


class DataListLoader(torch.utils.data.DataLoader):
    r"""A data loader which batches data objects from a
    :class:`torch_geometric.data.dataset` to a Python list.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    .. note::

        This data loader should be used for multi-GPU support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
            :obj:`num_workers`.
    """

    def __init__(self, dataset: Union[Dataset, List[BaseData]], batch_size: int = 1, shuffle: bool = False, **kwargs):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)

    def collate_fn(data_list):
        return data_list


class Dataset:
    def __init__(self, name="clic_edm_ttbar_pf:1.5.0", split="train"):
        builder = tfds.builder(name, data_dir="/pfvol/tensorflow_datasets/clic/clusters/")

        self.ds = builder.as_data_source(split=split)

        # to prevent a warning from tfds about accessing sequences of indices
        self.ds.__class__.__getitems__ = my_getitem

    def get_sampler(self):
        sampler = torch.utils.data.SequentialSampler(self.ds)
        return sampler

    def get_distributed_sampler(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.ds)
        return sampler

    def get_loader(self, batch_size=20, num_workers=2, prefetch_factor=4, num_gpus=0):
        if num_gpus > 1:
            return DataLoader(
                self.ds,
                batch_size=batch_size,
                collate_fn=Collater(),
                sampler=self.get_distributed_sampler(),
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
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


def collate_padded_batch(inputs):
    # num_samples_in_batch = len(inputs)
    elem_keys = list(inputs[0].keys())
    ret = {}
    for elem_key in elem_keys:
        batch = [Tensor(i[elem_key]) for i in inputs]
        max_seq_len = max([x.shape[0] for x in batch])
        padded_batch_data = [nn.functional.pad(x, (0, 0, 0, max_seq_len - x.shape[0])) for x in batch]
        ret[elem_key] = torch.stack(padded_batch_data, axis=0)
    ret["mask"] = ret["X"][:, :, 0] == 0

    return ret


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
