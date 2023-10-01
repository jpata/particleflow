import argparse
import os
import os.path as osp
import sys

sys.path.append("pyg/")

import matplotlib
import numpy as np

# import ray
# import ray.data
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# import torch_geometric
# from pyg.evaluate import make_predictions_awk
from pyg.logger import _logger
from pyg.mlpf import MLPF
from pyg.PFGraphDataset import PFGraphDataset
from pyg.training import train_mlpf
from pyg.utils import CLASS_LABELS, X_FEATURES, load_mlpf, save_mlpf

# from ray import train
# from ray.air import Checkpoint, session
# from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
# from ray.train.torch import TorchConfig, TorchTrainer
# from torch.nn.parallel import DistributedDataParallel as DDP

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


matplotlib.use("Agg")

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Author: Farouk Mokhtar
"""


parser = argparse.ArgumentParser()

parser.add_argument(
    "--backend",
    type=str,
    choices=["gloo", "nccl"],
    default=None,
    help="backend for distributed training (nccl should be faster)",
)
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help='to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `0,1`',
)
parser.add_argument("--outpath", type=str, default="../experiments/", help="output folder")
parser.add_argument("--prefix", type=str, default="MLPF_model", help="directory to hold the model and all plots")
parser.add_argument("--dataset", type=str, required=True, help="CLIC, CMS or DELPHES")
parser.add_argument("--data_path", type=str, default="../data/", help="path which contains the samples")
parser.add_argument("--sample", type=str, default="QCD", help="sample to test on")
parser.add_argument("--n_train", type=int, default=2, help="number of files to use for training")
parser.add_argument("--n_valid", type=int, default=2, help="number of data files to use for validation")
parser.add_argument("--n_test", type=int, default=2, help="number of data files to use for testing")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrites the model if True")
parser.add_argument("--load", dest="load", action="store_true", help="Load the model (no training)")
parser.add_argument("--train", dest="train", action="store_true", help="Initiates a training")
parser.add_argument("--n_epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=10, help="training minibatch size in number of events")
parser.add_argument("--patience", type=int, default=50, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--width", type=int, default=256, help="hidden dimension of mlpf")
parser.add_argument("--embedding_dim", type=int, default=256, help="first embedding of mlpf")
parser.add_argument("--num_convs", type=int, default=3, help="number of graph layers for mlpf")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout for MLPF model")
parser.add_argument("--space_dim", type=int, default=4, help="Gravnet hyperparameter")
parser.add_argument("--propagate_dim", type=int, default=22, help="Gravnet hyperparameter")
parser.add_argument("--nearest", type=int, default=32, help="k nearest neighbors in gravnet layer")
parser.add_argument("--conv_type", type=str, default="gnn-lsh", help="choices are ['gnn-lsh', 'gravnet', 'attention']")
parser.add_argument(
    "--make_predictions", dest="make_predictions", action="store_true", help="run inference on the test data"
)
parser.add_argument("--make_plots", dest="make_plots", action="store_true", help="makes plots of the test predictions")


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


# def inference(rank, world_size, args, data, model, PATH):
#     """
#     A function that may be passed as a demo_fn to run_demo() to perform inference over
#     multiple gpus using DDP in case there are multiple gpus available (world_size > 1).

#         . It divides and distributes the testing dataset appropriately.
#         . Copies the model on each gpu.
#         . Wraps the model with DDP on each gpu to allow synching of gradients.
#         . Runs inference

#     If there are NO multiple gpus available, the function should run fine
#     and use the available device for inference.
#     """

#     if world_size > 1:
#         setup(rank, world_size)
#     else:  # hack in case there's no multigpu
#         rank = 0
#         world_size = 1

#     # give each gpu a subset of the data
#     hyper_test = int(args.n_test / world_size)

#     test_dataset = torch.utils.data.Subset(data, np.arange(start=rank * hyper_test, stop=(rank + 1) * hyper_test))

#     if args.dataset == "CMS":  # construct "file loaders" first because we need to set num_workers>0 and prefetch_factor>2
#         file_loader_test = make_file_loaders(world_size, test_dataset)
#     else:  # construct pyg DataLoaders directly because "file loaders" are not needed
#         file_loader_test = torch_geometric.loader.DataLoader(test_dataset, args.batch_size)

#     if world_size > 1:
#         print(f"Running inference on rank {rank}: {torch.cuda.get_device_name(rank)}")
#         print(f"Copying the model on rank {rank}..")
#         model = model.to(rank)
#         model = DDP(model, device_ids=[rank])
#     else:
#         if torch.cuda.device_count():
#             rank = torch.device("cuda:0")
#         else:
#             rank = "cpu"
#         print(f"Running inference on {rank}")
#         model = model.to(rank)
#     model.eval()

#     make_predictions_awk(rank, args.dataset, model, file_loader_test, args.batch_size, PATH)

#     if world_size > 1:
#         cleanup()


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


def main():
    args = parser.parse_args()

    if args.gpus:
        if args.backend is not None:  # distributed training
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            device = torch.device(local_rank)
            import datetime

            torch.distributed.init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=5400))
            _logger.info(f"Using distributed PyTorch with {args.backend} backend")
        else:
            gpus = [int(i) for i in args.gpus.split(",")]
            device = torch.device(gpus[0])
    else:
        gpus = None
        device = torch.device("cpu")

    assert (
        len(gpus) <= torch.cuda.device_count()
    ), f"--gpus must match availability (specefied {len(gpus)} gpus but only {torch.cuda.device_count()} gpus are available)"

    outpath = osp.join(args.outpath, args.prefix)

    # get dataset
    from pyg import tfds_utils
    from pyg.utils import InterleavedIterator

    ds_train = [
        tfds_utils.Dataset("clic_edm_ttbar_pf:1.5.0", "train"),
        tfds_utils.Dataset("clic_edm_qq_pf:1.5.0", "train"),
        tfds_utils.Dataset("clic_edm_ww_fullhad_pf:1.5.0", "train"),
        tfds_utils.Dataset("clic_edm_zh_tautau_pf:1.5.0", "train"),
    ]
    ds_valid = [
        tfds_utils.Dataset("clic_edm_ttbar_pf:1.5.0", "test"),
        tfds_utils.Dataset("clic_edm_qq_pf:1.5.0", "test"),
        tfds_utils.Dataset("clic_edm_ww_fullhad_pf:1.5.0", "test"),
        tfds_utils.Dataset("clic_edm_zh_tautau_pf:1.5.0", "test"),
    ]

    for ds in ds_train:
        print("train_dataset: {}, {}".format(ds, len(ds)))
    for ds in ds_valid:
        print("test_dataset: {}, {}".format(ds, len(ds)))

    train_loaders = [ds.get_loader(batch_size=args.batch_size, num_workers=2, prefetch_factor=4) for ds in ds_train]
    valid_loaders = [ds.get_loader(batch_size=args.batch_size, num_workers=2, prefetch_factor=4) for ds in ds_valid]

    train_loader = InterleavedIterator(train_loaders)
    valid_loader = InterleavedIterator(valid_loaders)

    # load a pre-trained specified model, otherwise, instantiate and train a new model
    if args.load:
        model_state, model_kwargs = load_mlpf(device, outpath)

        model = MLPF(**model_kwargs).to(device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        model.load_state_dict(model_state)

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
            "conv_type": args.conv_type,
        }

        model = MLPF(**model_kwargs).to(device)

        # save model_kwargs and hyperparameters
        save_mlpf(args, outpath, model, model_kwargs)

        print(model)
        print(args.prefix)

        print(f"Training over {args.n_epochs} epochs on the {args.dataset} dataset.")

    # DistributedDataParallel
    if args.backend is not None:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=local_rank)

    # DataParallel
    if args.backend is None:
        print(gpus, len(gpus))
        if gpus is not None and len(gpus) > 1:
            print("DataParallel", gpus)
            model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    if args.train:
        # model = ray.train.torch.prepare_model(model)

        train_mlpf(
            device,
            model,
            train_loader,
            valid_loader,
            args.n_epochs,
            args.patience,
            args.lr,
            outpath,
        )

    if args.backend:
        dist.destroy_process_group()

    #     # load the best epoch state
    #     best_epoch = json.load(open(f"{outpath}/best_epoch.json"))["best_epoch"]
    #     model_state = torch.load(outpath + "/best_epoch_weights.pth", map_location=device)
    #     model.load_state_dict(model_state)

    #     try:
    #         dummy_features = torch.randn(256, model_kwargs["input_dim"], device=device)
    #         dummy_batch = torch.zeros(256, dtype=torch.int64, device=device)
    #         torch.onnx.export(
    #             model,
    #             (dummy_features, dummy_batch),
    #             "test.onnx",
    #             verbose=True,
    #             input_names=["features", "batch"],
    #             output_names=["id", "momentum", "charge"],
    #             dynamic_axes={
    #                 "features": {0: "num_elements"},
    #                 "batch": [0],
    #                 "id": [0],
    #                 "momentum": [0],
    #                 "charge": [0],
    #             },
    #         )
    #     except Exception as e:
    #         print("ONNX export failed: {}".format(e))

    # # prepare for inference and plotting
    # PATH = f"{outpath}/testing_epoch_{best_epoch}_{args.sample}/"
    # pred_path = f"{PATH}/predictions/"
    # plot_path = f"{PATH}/plots/"

    # if args.make_predictions:
    #     print(f"Will run inference on the {args.dataset} {args.sample} sample.")

    #     if not os.path.exists(PATH):
    #         os.makedirs(PATH)
    #     if not os.path.exists(pred_path):
    #         os.makedirs(pred_path)

    #     # load the qcd data for testing
    #     data = load_data(args.data_path, args.dataset, args.sample)
    #     print("loaded data={}".format(len(data)))

    #     # run the inference using DDP if more than one gpu is available
    #     if world_size > 1:
    #         run_demo(
    #             inference,
    #             world_size,
    #             args,
    #             data,
    #             model,
    #             PATH,
    #         )
    #     else:
    #         inference(
    #             device,
    #             world_size,
    #             args,
    #             data,
    #             model,
    #             PATH,
    #         )

    # # load the predictions and make plots (must have ran make_predictions() beforehand)
    # if args.make_plots:
    #     print(f"Will make plots of the {args.dataset} {args.sample} sample.")
    #     make_plots(pred_path, plot_path, args.dataset, args.sample)


if __name__ == "__main__":
    main()
