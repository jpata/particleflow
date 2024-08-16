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

        # sort the elements in each event in pT descending order
        if self.sort:
            sortidx = np.argsort(ret["X"][:, 1])[::-1]
            ret["X"] = ret["X"][sortidx]
            ret["ycand"] = ret["ycand"][sortidx]
            ret["ygen"] = ret["ygen"][sortidx]

        if self.ds.dataset_info.name.startswith("cms_"):
            ret["ygen"][:, 0][(ret["X"][:, 0] == 1) & (ret["ygen"][:, 0] == 2)] = 1
            ret["ygen"][:, 0][(ret["X"][:, 0] == 1) & (ret["ygen"][:, 0] == 5)] = 1

            ret["ygen"][:, 0][(ret["X"][:, 0] == 4) & (ret["ygen"][:, 0] == 1)] = 5
            ret["ygen"][:, 0][(ret["X"][:, 0] == 5) & (ret["ygen"][:, 0] == 1)] = 2
            # ret["ygen"][:, 0][(ret["X"][:, 0]==4) & (ret["ygen"][:, 0] == 6)] = 5
            ret["ygen"][:, 0][(ret["X"][:, 0] == 5) & (ret["ygen"][:, 0] == 6)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 4) & (ret["ygen"][:, 0] == 7)] = 5
            ret["ygen"][:, 0][(ret["X"][:, 0] == 5) & (ret["ygen"][:, 0] == 7)] = 2

            ret["ygen"][:, 0][(ret["X"][:, 0] == 8) & (ret["ygen"][:, 0] == 1)] = 4
            ret["ygen"][:, 0][(ret["X"][:, 0] == 9) & (ret["ygen"][:, 0] == 1)] = 3
            ret["ygen"][:, 0][(ret["X"][:, 0] == 8) & (ret["ygen"][:, 0] == 2)] = 4
            ret["ygen"][:, 0][(ret["X"][:, 0] == 9) & (ret["ygen"][:, 0] == 2)] = 3
            ret["ygen"][:, 0][(ret["X"][:, 0] == 8) & (ret["ygen"][:, 0] == 6)] = 4
            ret["ygen"][:, 0][(ret["X"][:, 0] == 9) & (ret["ygen"][:, 0] == 6)] = 3
            ret["ygen"][:, 0][(ret["X"][:, 0] == 8) & (ret["ygen"][:, 0] == 7)] = 4
            ret["ygen"][:, 0][(ret["X"][:, 0] == 9) & (ret["ygen"][:, 0] == 7)] = 3

            ret["ygen"][:, 0][(ret["X"][:, 0] == 10) & (ret["ygen"][:, 0] == 1)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 11) & (ret["ygen"][:, 0] == 1)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 10) & (ret["ygen"][:, 0] == 6)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 11) & (ret["ygen"][:, 0] == 6)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 10) & (ret["ygen"][:, 0] == 7)] = 2
            ret["ygen"][:, 0][(ret["X"][:, 0] == 11) & (ret["ygen"][:, 0] == 7)] = 2

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
            keys_to_get: any keys in the TFDS to retrieve (e.g. X, ygen, ycand)
        """
        if split == "valid":
            split = "test"

        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = TFDSDataSource(builder.as_data_source(split=split), sort=sort)

        if num_samples and num_samples < len(self.ds):
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def __len__(self):
        return len(self.ds)


class PFBatch:
    def __init__(self, **kwargs):
        self.attrs = list(kwargs.keys())

        # write out the possible attributes here explicitly
        self.X = kwargs.get("X")
        self.ygen = kwargs.get("ygen")
        self.ycand = kwargs.get("ycand", None)
        self.genmet = kwargs.get("genmet", None)
        self.genjets = kwargs.get("genjets", None)
        self.mask = self.X[:, :, 0] != 0

    def to(self, device, **kwargs):
        attrs = {}
        for attr in self.attrs:
            this_attr = getattr(self, attr)
            attrs[attr] = this_attr.to(device, **kwargs)
        return PFBatch(**attrs)


# pads items with variable lengths (seq_len1, seq_len2, ...) to [batch, max(seq_len), ...]
class Collater:
    def __init__(self, per_particle_keys_to_get, per_event_keys_to_get, **kwargs):
        super(Collater, self).__init__(**kwargs)
        self.per_particle_keys_to_get = (
            per_particle_keys_to_get  # these quantities are a variable-length tensor per each event
        )
        self.per_event_keys_to_get = per_event_keys_to_get  # these quantities are one value (scalar) per event

    def __call__(self, inputs):
        ret = {}

        # per-particle quantities need to be padded across events of different size
        for key_to_get in self.per_particle_keys_to_get:
            ret[key_to_get] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(inp[key_to_get]).to(torch.float32) for inp in inputs], batch_first=True
            )

        # per-event quantities can be stacked across events
        for key_to_get in self.per_event_keys_to_get:
            ret[key_to_get] = torch.stack([torch.tensor(inp[key_to_get]) for inp in inputs])

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
        for type_ in config[f"{split}_dataset"][config["dataset"]]:
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
                sampler = torch.utils.data.SequentialSampler(dataset)

            # build dataloaders
            batch_size = config[f"{split}_dataset"][config["dataset"]][type_]["batch_size"] * config["gpu_batch_multiplier"]
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=Collater(["X", "ygen", "genjets"], ["genmet"]),
                sampler=sampler,
                num_workers=config["num_workers"],
                prefetch_factor=config["prefetch_factor"],
                # pin_memory=use_cuda,
                # pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                drop_last=True,
            )

            loaders[split].append(loader)

        loaders[split] = InterleavedIterator(loaders[split])  # will interleave maximum of three dataloaders
    return loaders
