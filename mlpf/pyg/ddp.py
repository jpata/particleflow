import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_geometric
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    """
    Necessary setup function that sets up environment variables and initializes the process group
    to perform training & inference using DistributedDataParallel (DDP). DDP relies on c10d ProcessGroup
    for communications, hence, applications must create ProcessGroup instances before constructing DDP.

    Args:
    rank: the process id (or equivalently the gpu index)
    world_size: number of gpus available
    """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )  # nccl should be faster than gloo for DistributedDataParallel on gpus


def cleanup():
    """Necessary function that destroys the spawned process group at the end."""

    dist.destroy_process_group()


def run_demo(demo_fn, world_size, args, model):
    """
    Necessary function that spawns a process group of size=world_size processes to run demo_fn()
    on each gpu device that will be indexed by 'rank'.

    Args:
    demo_fn: function you wish to run on each gpu.
    world_size: number of gpus available.
    """

    mp.spawn(
        demo_fn,
        args=(world_size, args, model),
        nprocs=world_size,
        join=True,
    )


def train(rank, world_size, args, data, model, outpath):
    """
    A function that may be passed as a demo_fn to run_demo() to perform training over
    multiple gpus using DDP in case there are multiple gpus available (world_size > 1).

        . It divides and distributes the training dataset appropriately.
        . Copies the model on each gpu.
        . Wraps the model with DDP on each gpu to allow synching of gradients.
        . Invokes the training_loop() to run synchronized training among gpus.

    If there are NO multiple gpus available, the function should run fine
    and use the available device for training.
    """

    if world_size > 1:
        setup(rank, world_size)
    else:  # hack in case there's no multigpu
        rank = 0
        world_size = 1

    # give each gpu a subset of the data
    hyper_train = int(args.n_train / world_size)
    hyper_valid = int(args.n_valid / world_size)

    train_dataset = torch.utils.data.Subset(data, np.arange(start=rank * hyper_train, stop=(rank + 1) * hyper_train))
    valid_dataset = torch.utils.data.Subset(
        data, np.arange(start=args.n_train + rank * hyper_valid, stop=args.n_train + (rank + 1) * hyper_valid)
    )
    print("train_dataset={}".format(len(train_dataset)))
    print("valid_dataset={}".format(len(valid_dataset)))

    train_data = []
    for file in train_dataset:
        train_data += file
    file_loader_train = [torch_geometric.loader.DataLoader(train_data, args.bs)]

    valid_data = []
    for file in valid_dataset:
        valid_data += file
    file_loader_valid = [torch_geometric.loader.DataLoader(valid_data, args.bs)]

    print("-----------------------------")
    if world_size > 1:
        print(f"Running training on rank {rank}: {torch.cuda.get_device_name(rank)}")
        print(f"Copying the model on rank {rank}..")
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        if torch.cuda.device_count():
            rank = torch.device("cuda:0")
        else:
            rank = "cpu"
        print(f"Running training on {rank}")
        model = model.to(rank)
    model.train()

    from training import train_mlpf

    train_mlpf(
        rank,
        model,
        file_loader_train,
        file_loader_valid,
        args.bs,
        args.n_epochs,
        args.patience,
        args.lr,
        outpath,
    )

    if world_size > 1:
        cleanup()


def inference(rank, world_size, args, data, model, PATH):
    """
    A function that may be passed as a demo_fn to run_demo() to perform inference over
    multiple gpus using DDP in case there are multiple gpus available (world_size > 1).

        . It divides and distributes the testing dataset appropriately.
        . Copies the model on each gpu.
        . Wraps the model with DDP on each gpu to allow synching of gradients.
        . Runs inference

    If there are NO multiple gpus available, the function should run fine
    and use the available device for inference.
    """

    if world_size > 1:
        setup(rank, world_size)
    else:  # hack in case there's no multigpu
        rank = 0
        world_size = 1

    # give each gpu a subset of the data
    hyper_test = int(args.n_test / world_size)

    test_dataset = torch.utils.data.Subset(data, np.arange(start=rank * hyper_test, stop=(rank + 1) * hyper_test))

    file_loader_test = torch_geometric.loader.DataLoader(test_dataset, args.bs)

    if world_size > 1:
        print(f"Running inference on rank {rank}: {torch.cuda.get_device_name(rank)}")
        print(f"Copying the model on rank {rank}..")
        model = model.to(rank)
        model = DDP(model, device_ids=[rank])
    else:
        if torch.cuda.device_count():
            rank = torch.device("cuda:0")
        else:
            rank = "cpu"
        print(f"Running inference on {rank}")
        model = model.to(rank)
    model.eval()

    from evaluate import make_predictions

    make_predictions(rank, args.dataset, model, file_loader_test, args.bs, PATH)

    if world_size > 1:
        cleanup()
