"""
Debugging error when using num_workers>0 for pytorch dataloader in a torch.distributed context.
"""

import argparse
import logging
import os

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


class Collater:
    """Based on the Collater found on torch_geometric docs we build our own."""

    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, inputs):
        num_samples_in_batch = len(inputs)
        elem_keys = ["X"]

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


def main_worker(rank, world_size, args, ds):
    """Demo function that will be passed to each gpu if (world_size > 1) else will run normally on the given device."""

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

        sampler = torch.utils.data.distributed.DistributedSampler(ds)
    else:
        sampler = torch.utils.data.RandomSampler(ds)

    print(args.num_workers)
    if args.num_workers is not None:
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=Collater(),
            sampler=sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        print("here")
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            collate_fn=Collater(),
            sampler=sampler,
        )

    print("Looping over dataloader from inside the worker")
    for i, batch in enumerate(train_loader):
        print("batch", batch.to(rank))
        if i > 9:
            break

    if world_size > 1:
        dist.destroy_process_group()


def main():
    # e.g.
    # on cpu: python3 mlpf/ddp_error.py --gpus "" --num-workers 2
    # on single-gpu: python3 mlpf/ddp_error.py --gpus "0" --num-workers 2
    # on multi-gpu: python3 mlpf/ddp_error.py --gpus "0,1" --num-workers 2

    args = parser.parse_args()
    world_size = len(args.gpus.split(","))  # will be 1 for both cpu ("") and single-gpu ("0")

    print("Defining dataset")
    builder = tfds.builder("cms_pf_ttbar:1.6.0", data_dir=args.data_dir)
    ds = builder.as_data_source(split="train")

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
                args=(world_size, args, ds),
                nprocs=world_size,
                join=True,
                start_method=args.spawn_method,
            )

        elif world_size == 1:
            rank = 0
            print(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}")
            main_worker(rank, world_size, args, ds)

    else:
        rank = "cpu"
        print("Will use cpu")
        main_worker(rank, world_size, args, ds)


if __name__ == "__main__":
    main()
