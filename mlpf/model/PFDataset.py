import sys
from types import SimpleNamespace

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.utils.data

from mlpf.model.logger import _logger


# https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
SHARING_STRATEGY = "file_descriptor"


class TFDSDataSource:
    def __init__(self, ds, sort, pad_to_multiple=None):
        self.ds = ds
        tmp = self.ds.dataset_info
        self.ds.dataset_info = SimpleNamespace()
        self.ds.dataset_info.name = tmp.name
        self.ds.dataset_info.features = tmp.features
        self.sort = sort
        self.pad_to_multiple = pad_to_multiple
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
            ret["ytarget"] = ret["ytarget"][sortidx]

        # Pad elements to the nearest multiple if specified
        if self.pad_to_multiple and self.pad_to_multiple > 0:
            current_len = ret["X"].shape[0]
            if current_len % self.pad_to_multiple != 0:
                num_to_pad = self.pad_to_multiple - (current_len % self.pad_to_multiple)
                for key_to_pad in ["X", "ycand", "ytarget"]:
                    if key_to_pad in ret:  # Ensure key exists
                        array_to_pad = ret[key_to_pad]
                        pad_width = ((0, num_to_pad), (0, 0))  # Pad only the first axis
                        ret[key_to_pad] = np.pad(array_to_pad, pad_width, mode="constant", constant_values=0)

        if self.ds.dataset_info.name.startswith("cms_"):
            # track, target label neutral hadron -> reconstruct as charged hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 1) & (ret["ytarget"][:, 0] == 2)] = 1

            # track, target label photon -> reconstruct as charged hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 1) & (ret["ytarget"][:, 0] == 5)] = 1

            # ECAL cluster, target label charged hadron -> reconstruct as photon
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 4) & (ret["ytarget"][:, 0] == 1)] = 5

            # HCAL cluster, target label charged hadron -> reconstruct as neutral hadron
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 1)] = 2

            # ECAL cluster, target label electron -> reconstruct as photon
            # ret["ytarget"][:, 0][(ret["X"][:, 0]==4) & (ret["ytarget"][:, 0] == 6)] = 5

            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 4) & (ret["ytarget"][:, 0] == 7)] = 5
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 5) & (ret["ytarget"][:, 0] == 7)] = 2

            # HFEM cluster, reconstruct as HFEM
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 8) & (ret["ytarget"][:, 0] != 0)] = 4

            # HFHAD cluster, reconstruct as HFHAD
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 9) & (ret["ytarget"][:, 0] != 0)] = 3

            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 1)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 1)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 6)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 10) & (ret["ytarget"][:, 0] == 7)] = 2
            ret["ytarget"][:, 0][(ret["X"][:, 0] == 11) & (ret["ytarget"][:, 0] == 7)] = 2

            # set pt for HO which would otherwise be 0
            msk_ho = ret["X"][:, 0] == 10
            eta = ret["X"][:, 2][msk_ho]
            e = ret["X"][:, 5][msk_ho]
            ret["X"][:, 1][msk_ho] = np.sqrt(e**2 - (np.tanh(eta) * e) ** 2)

        # transform pt -> log(pt / elem pt), same for energy
        # where target does not exist, set to 0
        with np.errstate(divide="ignore"):
            target_pt = np.log(ret["ytarget"][:, 2] / ret["X"][:, 1])
        target_pt[np.isnan(target_pt)] = 0
        target_pt[np.isinf(target_pt)] = 0
        ret["ytarget_pt_orig"] = ret["ytarget"][:, 2].copy()
        ret["ytarget"][:, 2] = target_pt

        with np.errstate(divide="ignore"):
            target_e = np.log(ret["ytarget"][:, 6] / ret["X"][:, 5])
        target_e[ret["ytarget"][:, 0] == 0] = 0
        target_e[np.isnan(target_e)] = 0
        target_e[np.isinf(target_e)] = 0
        ret["ytarget_e_orig"] = ret["ytarget"][:, 6].copy()
        ret["ytarget"][:, 6] = target_e

        return ret

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return self.rep


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, num_samples=None, sort=False, pad_to_multiple=512):
        """
        Args
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test" (if "valid" then will use "test")
        """
        if split == "valid":
            split = "test"

        try:
            builder = tfds.builder(name, data_dir=data_dir)
        except Exception:
            _logger.error(
                "Could not find dataset {} in {}, please check that you have downloaded the correct version of the dataset".format(name, data_dir)
            )
            sys.exit(1)
        self.ds = TFDSDataSource(builder.as_data_source(split=split), sort=sort, pad_to_multiple=pad_to_multiple)

        if num_samples and num_samples < len(self.ds):
            self.ds = torch.utils.data.Subset(self.ds, range(num_samples))

    def __len__(self):
        return len(self.ds)


class PFBatch:
    def __init__(self, **kwargs):
        self.attrs = list(kwargs.keys())

        # write out the possible attributes here explicitly
        self.X = kwargs.get("X")
        self.ytarget = kwargs.get("ytarget")
        self.ytarget_pt_orig = kwargs.get("ytarget_pt_orig", None)
        self.ytarget_e_orig = kwargs.get("ytarget_e_orig", None)
        self.ycand = kwargs.get("ycand", None)
        self.genmet = kwargs.get("genmet", None)
        self.genjets = kwargs.get("genjets", None)
        self.targetjets = kwargs.get("targetjets", None)
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
        self.per_particle_keys_to_get = per_particle_keys_to_get  # these quantities are a variable-length tensor per each event
        self.per_event_keys_to_get = per_event_keys_to_get  # these quantities are one value (scalar) per event

    def __call__(self, inputs):
        ret = {}

        # per-particle quantities need to be padded across events of different size
        for key_to_get in self.per_particle_keys_to_get:
            ret[key_to_get] = torch.nn.utils.rnn.pad_sequence([torch.tensor(inp[key_to_get]).to(torch.float32) for inp in inputs], batch_first=True)

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


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def get_interleaved_dataloaders(world_size, rank, config, use_cuda, use_ray):
    loaders = {}
    for split in ["train", "valid"]:  # build train, valid dataset and dataloaders
        loaders[split] = []
        for type_ in config[f"{split}_dataset"][config["dataset"]]:
            dataset = []
            for sample in config[f"{split}_dataset"][config["dataset"]][type_]["samples"]:
                version = config[f"{split}_dataset"][config["dataset"]][type_]["samples"][sample]["version"]
                split_configs = config[f"{split}_dataset"][config["dataset"]][type_]["samples"][sample]["splits"]
                print("split_configs", split_configs)

                nevents = None
                if not (config[f"n{split}"] is None):
                    nevents = config[f"n{split}"] // len(split_configs)

                for split_config in split_configs:
                    ds = PFDataset(
                        config["data_dir"],
                        f"{sample}/{split_config}:{version}",
                        split,
                        num_samples=nevents,
                        sort=config["sort_data"],
                        pad_to_multiple=config.get("pad_to_multiple_elements", None),
                    ).ds

                    if (rank == 0) or (rank == "cpu"):
                        _logger.info(f"{split}_dataset: {sample}, {len(ds)}", color="blue")

                    dataset.append(ds)
            dataset = torch.utils.data.ConcatDataset(dataset)

            shuffle = split == "train"
            if world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            else:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)

            # build dataloaders
            batch_size = config[f"{split}_dataset"][config["dataset"]][type_]["batch_size"] * config["gpu_batch_multiplier"]
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=Collater(["X", "ytarget"], ["genmet"]),
                sampler=sampler,
                num_workers=config["num_workers"],
                prefetch_factor=config["prefetch_factor"],
                # pin_memory=use_cuda,
                # pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                drop_last=True,
                worker_init_fn=set_worker_sharing_strategy,
                # persistent_workers=True,
            )

            loaders[split].append(loader)

        loaders[split] = InterleavedIterator(loaders[split])
    return loaders
