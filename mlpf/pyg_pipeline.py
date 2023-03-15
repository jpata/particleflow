import json
import os
import os.path as osp

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_geometric
from pyg.args import parse_args
from pyg.evaluate import make_predictions_awk
from pyg.mlpf import MLPF
from pyg.PFGraphDataset import PFGraphDataset
from pyg.plotting import make_plots
from pyg.training import training_loop
from pyg.utils import CLASS_LABELS, X_FEATURES, load_mlpf, make_file_loaders, save_mlpf
from torch.nn.parallel import DistributedDataParallel as DDP

matplotlib.use("Agg")


"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

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
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )  # nccl should be faster than gloo for DistributedDataParallel on gpus


def cleanup():
    """Necessary function that destroys the spawned process group at the end."""

    dist.destroy_process_group()


def run_demo(demo_fn, world_size, args, dataset, model, outpath):
    """
    Necessary function that spawns a process group of size=world_size processes to run demo_fn()
    on each gpu device that will be indexed by 'rank'.

    Args:
    demo_fn: function you wish to run on each gpu.
    world_size: number of gpus available.
    """

    mp.spawn(
        demo_fn,
        args=(world_size, args, dataset, model, outpath),
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

    if args.dataset == "CMS":  # construct file loaders first because we need to set num_workers>0 and pre_fetch factors>2
        file_loader_train = make_file_loaders(world_size, train_dataset)
        file_loader_valid = make_file_loaders(world_size, valid_dataset)
    else:  # construct pyg DataLoaders directly
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

    training_loop(
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

    if args.dataset == "CMS":  # construct "file loaders" first because we need to set num_workers>0 and prefetch_factor>2
        file_loader_test = make_file_loaders(world_size, test_dataset)
    else:  # construct pyg DataLoaders directly because "file loaders" are not needed
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

    make_predictions_awk(rank, args.dataset, model, file_loader_test, args.bs, PATH)

    if world_size > 1:
        cleanup()


def load_data(data_path, dataset, sample):
    """Loads the appropriate sample for a given dataset."""
    dict_ = {
        "CMS": {
            "TTbar": f"{data_path}/TTbar_14TeV_TuneCUETP8M1_cfi/",
            "QCD": f"{data_path}/QCDForPF_14TeV_TuneCUETP8M1_cfi/",
        },
        "DELPHES": {"TTbar": f"{data_path}/pythia8_ttbar/", "QCD": f"{data_path}/pythia8_qcd/"},
        "CLIC": {"TTbar": f"{data_path}/p8_ee_tt_ecm380//", "QCD": f"{data_path}/p8_ee_qq_ecm380//"},
    }
    return PFGraphDataset(dict_[dataset][sample], dataset)


if __name__ == "__main__":

    args = parse_args()

    world_size = torch.cuda.device_count()

    torch.backends.cudnn.benchmark = True

    outpath = osp.join(args.outpath, args.prefix)

    # load a pre-trained specified model, otherwise, instantiate and train a new model
    if args.load:
        state_dict, model_kwargs = load_mlpf(device, outpath)

        model = MLPF(**model_kwargs)
        model.load_state_dict(state_dict)

    else:
        model_kwargs = {
            "input_dim": len(X_FEATURES[args.dataset]),
            "NUM_CLASSES": len(CLASS_LABELS[args.dataset]),
            "embedding_dim": args.embedding_dim,
            "width": args.width,
            "num_convs": args.num_convs,
            "k": args.nearest,
            "propagate_dimensions": args.propagate_dim,
            "space_dimensions": args.space_dim,
            "dropout": args.dropout,
        }

        model = MLPF(**model_kwargs)

        # save model_kwargs and hyperparameters
        save_mlpf(args, outpath, model, model_kwargs)

        print(model)
        print(args.prefix)

        print(f"Training over {args.n_epochs} epochs on the {args.dataset} dataset.")

        # load the ttbar data for training/validation
        data = load_data(args.data_path, args.dataset, "TTbar")
        print("loaded data={}".format(len(data)))

        # run the training using DDP if more than one gpu is available
        if world_size > 1:
            run_demo(
                train,
                world_size,
                args,
                data,
                model,
                outpath,
            )
        else:
            train(device, world_size, args, data, model, outpath)

    # load the best epoch state
    best_epoch = json.load(open(f"{outpath}/best_epoch.json"))["best_epoch"]
    state_dict = torch.load(outpath + "/best_epoch_weights.pth", map_location=device)
    model.load_state_dict(state_dict)

    # prepare for inference and plotting
    PATH = f"{outpath}/testing_epoch_{best_epoch}_{args.sample}/"
    pred_path = f"{PATH}/predictions/"
    plot_path = f"{PATH}/plots/"

    if args.make_predictions:
        print(f"Will run inference on the {args.dataset} {args.sample} sample.")

        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        # load the qcd data for testing
        data = load_data(args.data_path, args.dataset, args.sample)
        print("loaded data={}".format(len(data)))

        # run the inference using DDP if more than one gpu is available
        if world_size > 1:
            run_demo(
                inference,
                world_size,
                args,
                data,
                model,
                PATH,
            )
        else:
            inference(
                device,
                world_size,
                args,
                data,
                model,
                PATH,
            )

    # load the predictions and make plots (must have ran make_predictions() beforehand)
    if args.make_plots:
        print(f"Will make plots of the {args.dataset} {args.sample} sample.")
        make_plots(pred_path, plot_path, args.dataset, args.sample)
