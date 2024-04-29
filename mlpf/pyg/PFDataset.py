from types import SimpleNamespace

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.utils.data
from pyg.logger import _logger


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


class PFBatch:
    def __init__(self, X=None, ygen=None, ycand=None):
        self.X = X
        self.ygen = ygen
        self.ycand = ycand
        self.mask = X[:, :, 0] != 0

    def to(self, device, **kwargs):
        attrs = {}
        for attr in ["X", "ygen", "ycand"]:
            this_attr = getattr(self, attr)
            if not (this_attr is None):
                attrs[attr] = this_attr.to(device, **kwargs)
        return PFBatch(**attrs)


# pads items with variable lengths (seq_len1, seq_len2, ...) to [batch, max(seq_len), ...]
class Collater:
    def __init__(self, keys_to_get, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.keys_to_get = keys_to_get

    def __call__(self, inputs):
        ret = {}
        for key_to_get in self.keys_to_get:
            ret[key_to_get] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(inp[key_to_get]).to(torch.float32) for inp in inputs], batch_first=True
            )

        return PFBatch(**ret)


class InterleavedIterator(object):
    """Will combine DataLoaders of different lengths and batch sizes."""

    def __init__(self, data_loaders):
        self.idx = 0
        self.data_loaders = data_loaders
        self.data_loaders_iter = [iter(dl) for dl in data_loaders]
        max_loader_size = max([len(dl) for dl in data_loaders])

        self.loader_ds_indices = []

        # iterate loaders interleaved
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


def get_interleaved_dataloaders(world_size, rank, config, use_cuda, use_ray):
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
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=Collater(["X", "ygen", "ycand"]),
                sampler=sampler,
                num_workers=config["num_workers"],
                prefetch_factor=config["prefetch_factor"],
                pin_memory=use_cuda,
                pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                drop_last=True,
            )

            # This doesn't seem to be needed anymore. 2024.04.17
            # if use_ray:
            #     import ray

            #     # prepare loader for distributed training, adds distributed sampler
            #     loader = ray.train.torch.prepare_data_loader(loader)

            loaders[split].append(loader)

        loaders[split] = InterleavedIterator(loaders[split])  # will interleave maximum of three dataloaders
    return loaders
