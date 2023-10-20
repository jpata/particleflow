"""
Debugging error when using num_workers>0 for pytorch dataloader in a torch.distributed context.
"""

import argparse
import logging
import os
from typing import List, Optional

import tensorflow_datasets as tfds
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, default="0", help="to use CPU set to empty string; else e.g., `0,1`")
parser.add_argument("--data_dir", type=str, default="/pfvol/tensorflow_datasets_small/", help="path to tfds")
parser.add_argument("--batch-size", type=int, default=2, help="batch size for data loader")
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=2, help="will only be set if --num-workers>0")
parser.add_argument("--spawn-method", type=str, default="spawn", help="['spawn', 'fork', 'forkserver']")


class PFDataset:
    """Builds a DataSource from tensorflow datasets."""

    def __init__(self, data_dir, name, split, keys_to_get):
        """
        Args
            data_dir: path to tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
            name: sample and version (e.g. `clic_edm_ttbar_pf:1.5.0`)
            split: "train" or "test
        """

        builder = tfds.builder(name, data_dir=data_dir)

        self.ds = builder.as_data_source(split=split)

        self.keys_to_get = keys_to_get

    def get_sampler(self):
        sampler = torch.utils.data.RandomSampler(self.ds)
        return sampler

    def get_distributed_sampler(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.ds)
        return sampler

    def get_loader(self, world_size, batch_size, num_workers=None, prefetch_factor=2):
        if world_size > 1:
            sampler = self.get_distributed_sampler()
        else:
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


def main_worker(rank, world_size, args):
    """Demo function that will be passed to each gpu if (world_size > 1) else will run normally on the given device."""

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    print("Defining dataset")
    ds = PFDataset(args.data_dir, "cms_pf_ttbar:1.6.0", "train", ["X", "ygen"])

    print("Defining dataloader")
    train_loader = ds.get_loader(world_size, args.batch_size, args.num_workers, args.prefetch_factor)

    print("Looping over dataloader")
    for i, batch in enumerate(train_loader):
        print("batch", batch.to(rank))
        if i > 9:
            break

    if world_size > 1:
        dist.destroy_process_group()


def main():
    # e.g.
    # on cpu: python3 ddp_error.py --gpus "" --num-workers 2
    # on single-gpu: python3 ddp_error.py --gpus "0" --num-workers 2
    # on multi-gpu: python3 ddp_error.py --gpus "0,1" --num-workers 2

    args = parser.parse_args()
    world_size = len(args.gpus.split(","))  # will be 1 for both cpu ("") and single-gpu ("0")

    if args.gpus:
        assert (
            world_size <= torch.cuda.device_count()
        ), f"--gpus is too high (specefied {world_size} gpus but only {torch.cuda.device_count()} gpus are available)"

        if world_size > 1:
            print(f"Will use torch.nn.parallel.DistributedDataParallel() and {world_size} gpus")
            for rank in range(world_size):
                print(torch.cuda.get_device_name(rank))

            mp.start_processes(
                main_worker,
                args=(world_size, args),
                nprocs=world_size,
                join=True,
                start_method=args.spawn_method,
            )

        elif world_size == 1:
            rank = 0
            print(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}")
            main_worker(rank, world_size, args)

    else:
        rank = "cpu"
        print("Will use cpu")
        main_worker(rank, world_size, args)


if __name__ == "__main__":
    main()
