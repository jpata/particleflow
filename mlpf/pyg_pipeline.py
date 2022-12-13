import json
import os
import os.path as osp

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyg import features_cms
from pyg import features_delphes
from pyg import load_model
from pyg import make_file_loaders
from pyg import make_plots_cms
from pyg import make_predictions
from pyg import MLPF
from pyg import parse_args
from pyg import PFGraphDataset
from pyg import postprocess_predictions
from pyg import save_model
from pyg import target_p4
from pyg import training_loop
from torch.nn.parallel import DistributedDataParallel as DDP

matplotlib.use("Agg")


"""
Developing a PyTorch Geometric MLPF pipeline using DistributedDataParallel.

Author: Farouk Mokhtar
"""

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# define the global base device
if torch.cuda.device_count():
    device = torch.device("cuda:0")
else:
    device = "cpu"


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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # should be faster for DistributedDataParallel on gpus


def cleanup():
    """
    Necessary function that destroys the spawned process group at the end.
    """

    dist.destroy_process_group()


def run_demo(demo_fn, world_size, args, dataset, model, num_classes, outpath):
    """
    Necessary function that spawns a process group of size=world_size processes to run demo_fn()
    on each gpu device that will be indexed by 'rank'.

    Args:
    demo_fn: function you wish to run on each gpu
    world_size: number of gpus available
    mode: 'train' or 'inference'
    """

    # mp.set_start_method('forkserver')

    mp.spawn(
        demo_fn,
        args=(world_size, args, dataset, model, num_classes, outpath),
        nprocs=world_size,
        join=True,
    )


def train_ddp(rank, world_size, args, dataset, model, num_classes, outpath):
    """
    A train_ddp() function that will be passed as a demo_fn to run_demo() to
    perform training over multiple gpus using DDP.

    It divides and distributes the training dataset appropriately, copies the model,
    wraps the model with DDP on each device to allow synching of gradients,
    and finally, invokes the training_loop() to run synchronized training among devices.
    """

    setup(rank, world_size)

    print(f"Running training on rank {rank}: {torch.cuda.get_device_name(rank)}")

    # give each gpu a subset of the data
    hyper_train = int(args.n_train / world_size)
    hyper_valid = int(args.n_valid / world_size)

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=rank * hyper_train, stop=(rank + 1) * hyper_train))
    valid_dataset = torch.utils.data.Subset(
        dataset, np.arange(start=args.n_train + rank * hyper_valid, stop=args.n_train + (rank + 1) * hyper_valid)
    )

    # construct file loaders
    file_loader_train = make_file_loaders(
        world_size, train_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )
    file_loader_valid = make_file_loaders(
        world_size, valid_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )

    # copy the model to the GPU with id=rank
    print(f"Copying the model on rank {rank}..")
    model = model.to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)

    training_loop(
        rank,
        args.data,
        ddp_model,
        file_loader_train,
        file_loader_valid,
        args.batch_size,
        args.n_epochs,
        args.patience,
        optimizer,
        args.alpha,
        args.target,
        num_classes,
        outpath,
    )

    cleanup()


def inference_ddp(rank, world_size, args, dataset, model, num_classes, PATH):
    """
    An inference_ddp() function that will be passed as a demo_fn to run_demo()
    to perform inference over multiple gpus using DDP.

    It divides and distributes the testing dataset appropriately, copies the model,
    and wraps the model with DDP on each device.
    """

    setup(rank, world_size)

    print(f"Running inference on rank {rank}: {torch.cuda.get_device_name(rank)}")

    # give each gpu a subset of the data
    hyper_test = int(args.n_test / world_size)

    test_dataset = torch.utils.data.Subset(dataset, np.arange(start=rank * hyper_test, stop=(rank + 1) * hyper_test))

    # construct data loaders
    file_loader_test = make_file_loaders(
        world_size, test_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )

    # copy the model to the GPU with id=rank
    print(f"Copying the model on rank {rank}..")
    model = model.to(rank)
    model.eval()
    ddp_model = DDP(model, device_ids=[rank])

    make_predictions(rank, ddp_model, file_loader_test, args.batch_size, num_classes, PATH)

    cleanup()


def train(device, world_size, args, dataset, model, num_classes, outpath):
    """
    A train() function that will load the training dataset and start a training_loop
    on a single device (cuda or cpu).
    """

    if device == "cpu":
        print("Running training on cpu")
    else:
        print(f"Running training on: {torch.cuda.get_device_name(device)}")
        device = device.index

    train_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=args.n_train))
    valid_dataset = torch.utils.data.Subset(dataset, np.arange(start=args.n_train, stop=args.n_train + args.n_valid))

    # construct file loaders
    file_loader_train = make_file_loaders(
        world_size, train_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )
    file_loader_valid = make_file_loaders(
        world_size, valid_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )

    # move the model to the device (cuda or cpu)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    training_loop(
        device,
        args.data,
        model,
        file_loader_train,
        file_loader_valid,
        args.batch_size,
        args.n_epochs,
        args.patience,
        optimizer,
        args.alpha,
        args.target,
        num_classes,
        outpath,
    )


def inference(device, world_size, args, dataset, model, num_classes, PATH):
    """
    An inference() function that will load the testing dataset and start running inference
    on a single device (cuda or cpu).
    """

    if device == "cpu":
        print("Running inference on cpu")
    else:
        print(f"Running inference on: {torch.cuda.get_device_name(device)}")
        device = device.index

    test_dataset = torch.utils.data.Subset(dataset, np.arange(start=0, stop=args.n_test))

    # construct data loaders
    file_loader_test = make_file_loaders(
        world_size, test_dataset, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )

    # copy the model to the GPU with id=rank
    model = model.to(device)
    model.eval()

    make_predictions(device, model, file_loader_test, args.batch_size, num_classes, PATH)


if __name__ == "__main__":

    args = parse_args()

    world_size = torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True

    # retrieve the dimensions of the PF-elements & PF-candidates
    # to set the input/output dimension of the model
    if args.data == "delphes":
        input_dim = len(features_delphes)
        num_classes = 6  # we have 6 classes/pids for delphes
    elif args.data == "cms":
        input_dim = len(features_cms)
        num_classes = 9  # we have 9 classes/pids for cms (including taus)
    output_dim_p4 = len(target_p4)

    outpath = osp.join(args.outpath, args.model_prefix)

    # load a pre-trained specified model, otherwise, instantiate and train a new model
    if args.load:
        state_dict, model_kwargs, outpath = load_model(device, outpath, args.model_prefix, args.load_epoch)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

    else:
        model_kwargs = {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "output_dim_p4": output_dim_p4,
            "embedding_dim": args.embedding_dim,
            "hidden_dim1": args.hidden_dim1,
            "hidden_dim2": args.hidden_dim2,
            "num_convs": args.num_convs,
            "space_dim": args.space_dim,
            "propagate_dim": args.propagate_dim,
            "k": args.nearest,
        }

        model = MLPF(**model_kwargs)

        # save model_kwargs and hyperparameters
        save_model(args, args.model_prefix, outpath, model_kwargs)

        print(model)
        print(args.model_prefix)

        print("Training over {} epochs".format(args.n_epochs))

        # run the training using DDP if more than one gpu is available
        dataset = PFGraphDataset(args.dataset, args.data)

        if world_size >= 2:
            run_demo(train_ddp, world_size, args, dataset, model, num_classes, outpath)
        else:
            train(device, world_size, args, dataset, model, num_classes, outpath)

        # load the best epoch state
        state_dict = torch.load(outpath + "/best_epoch_weights.pth", map_location=device)
        model.load_state_dict(state_dict)

    # specify which epoch/state to load to run the inference and make plots
    if args.load and args.load_epoch != -1:
        epoch_to_load = args.load_epoch
    else:
        epoch_to_load = json.load(open(f"{outpath}/best_epoch.json"))["best_epoch"]

    PATH = f"{outpath}/testing_epoch_{epoch_to_load}_{args.sample}/"
    pred_path = f"{PATH}/predictions/"
    plot_path = f"{PATH}/plots/"

    # run the inference
    if args.make_predictions:

        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if not os.path.exists(f"{PATH}/predictions/"):
            os.makedirs(f"{PATH}/predictions/")
        if not os.path.exists(f"{PATH}/plots/"):
            os.makedirs(f"{PATH}/plots/")

        # run the inference using DDP if more than one gpu is available
        dataset_test = PFGraphDataset(args.dataset_test, args.data)

        if world_size >= 2:
            run_demo(inference_ddp, world_size, args, dataset_test, model, num_classes, PATH)
        else:
            inference(device, world_size, args, dataset_test, model, num_classes, PATH)

        postprocess_predictions(pred_path)

    # load the predictions and make plots (must have ran make_predictions before)
    if args.make_plots:

        if not osp.isdir(plot_path):
            os.makedirs(plot_path)

        if args.data == "cms":
            make_plots_cms(pred_path, plot_path, args.sample)
