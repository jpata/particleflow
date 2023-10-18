"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Author: Farouk Mokhtar
"""

import argparse
import logging
import os
import os.path as osp
import pickle as pkl
import sys
from pathlib import Path

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import fastjet
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyg.inference import make_plots, run_predictions
from pyg.logger import _configLogger, _logger
from pyg.mlpf import MLPF
from pyg.training import train_mlpf
from pyg.utils import CLASS_LABELS, X_FEATURES, InterleavedIterator, PFDataset, save_HPs
from utils import create_experiment_dir

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default="parameters/pyg-cms.yaml", help="yaml config")
parser.add_argument("--prefix", type=str, default="test_", help="prefix appended to result dir name")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="overwrites the model if True")
parser.add_argument("--data_dir", type=str, default="/pfvol/tensorflow_datasets/", help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=str, default="0", help="to use CPU set to empty string; else e.g., `0,1`")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=1, help="Increase batch size per GPU by this constant factor"
)
parser.add_argument("--dataset", type=str, choices=["clic", "cms", "delphes"], required=True, help="which dataset?")
parser.add_argument("--load", type=str, default=None, help="dir from which to load a saved model")
parser.add_argument("--train", action="store_true", help="initiates a training")
parser.add_argument("--test", action="store_true", help="tests the model")
parser.add_argument("--num-epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--patience", type=int, default=20, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--conv-type", type=str, default="gravnet", help="choices are ['gnn-lsh', 'gravnet', 'attention']")
parser.add_argument("--make-plots", action="store_true", help="make plots of the test predictions")
parser.add_argument("--export-onnx", action="store_true", help="exports the model to onnx")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="testing samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=500, help="validation samples to use, default will use 500 events")
parser.add_argument("--log-file", type=str, default="log.log", help="path to the log file")


def run(rank, world_size, args, outdir):
    """Demo function that will be passed to each gpu if (world_size > 1) else will run normally on the given device."""

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    if (rank == 0) or (rank == "cpu"):  # write the logs
        _configLogger("mlpf", stdout=sys.stdout, filename=f"{outdir}/{args.log_file}")
        # logger = logging.getLogger("mlpf")
        # logfile = logging.FileHandler(f"{outdir}/{args.log_file}")
        # logger.addHandler(logfile)

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    if args.load:  # load a pre-trained model
        with open(f"{outdir}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)

        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        checkpoint = torch.load(f"{outdir}/best_weights.pth", map_location=torch.device(rank))

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (rank == 0) or (rank == "cpu"):
            _logger.info(f"Loaded model weights from {outdir}/best_weights.pth")

    else:  # instantiate a new model in the outdir created
        model_kwargs = {
            "input_dim": len(X_FEATURES[args.dataset]),
            "num_classes": len(CLASS_LABELS[args.dataset]),
            **config["model"][args.conv_type],
        }
        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.to(rank)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)

    if args.train:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info(f"Creating experiment dir {outdir}")
            _logger.info(f"Model directory {outdir}", color="bold")

        train_loaders = []
        for sample in config["train_dataset"][args.dataset]:
            version = config["train_dataset"][args.dataset][sample]["version"]
            batch_size = config["train_dataset"][args.dataset][sample]["batch_size"] * args.gpu_batch_multiplier

            ds = PFDataset(args.data_dir, f"{sample}:{version}", "train", ["X", "ygen"], num_samples=args.ntrain)
            _logger.info(f"train_dataset: {ds}, {len(ds)}", color="blue")

            train_loaders.append(ds.get_loader(batch_size=batch_size, world_size=world_size))
        train_loader = InterleavedIterator(train_loaders)

        if (rank == 0) or (rank == "cpu"):  # quick validation only on a single machine
            valid_loaders = []
            for sample in config["valid_dataset"][args.dataset]:
                version = config["valid_dataset"][args.dataset][sample]["version"]
                batch_size = config["valid_dataset"][args.dataset][sample]["batch_size"] * args.gpu_batch_multiplier

                ds = PFDataset(args.data_dir, f"{sample}:{version}", "test", ["X", "ygen", "ycand"], num_samples=args.nvalid)
                _logger.info(f"valid_dataset: {ds}, {len(ds)}", color="blue")

                valid_loaders.append(ds.get_loader(batch_size=batch_size, world_size=1))
            valid_loader = InterleavedIterator(valid_loaders)
        else:
            valid_loader = None

        train_mlpf(
            rank,
            world_size,
            model,
            optimizer,
            train_loader,
            valid_loader,
            args.num_epochs,
            args.patience,
            outdir,
        )

    if args.test:
        if args.load is None:
            # if we don't load, we must have a newly trained model
            assert args.train, "Please train a model before testing, or load a model with --load"
            assert outdir is not None, "Error: no outdir to evaluate model from"
        else:
            outdir = args.load

        test_loaders = {}
        for sample in config["test_dataset"][args.dataset]:
            version = config["test_dataset"][args.dataset][sample]["version"]
            batch_size = config["test_dataset"][args.dataset][sample]["batch_size"] * args.gpu_batch_multiplier

            ds = PFDataset(args.data_dir, f"{sample}:{version}", "test", ["X", "ygen", "ycand"], num_samples=args.ntest)
            _logger.info(f"test_dataset: {ds}, {len(ds)}", color="blue")

            test_loaders[sample] = InterleavedIterator([ds.get_loader(batch_size=batch_size, world_size=world_size)])

            if not osp.isdir(f"{outdir}/preds/{sample}"):
                if (rank == 0) or (rank == "cpu"):
                    os.system(f"mkdir -p {outdir}/preds/{sample}")

        checkpoint = torch.load(f"{outdir}/best_weights.pth", map_location=torch.device(rank))

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for sample in test_loaders:
            _logger.info(f"Running predictions on {sample}")
            torch.cuda.empty_cache()

            if args.dataset == "clic":
                jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
            else:
                jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

            run_predictions(rank, model, test_loaders[sample], sample, outdir, jetdef, jet_ptcut=5.0, jet_match_dr=0.1)

    if (rank == 0) or (rank == "cpu"):  # make plots and export to onnx only on a single machine
        if args.make_plots:
            for sample in config["test_dataset"][args.dataset]:
                _logger.info(f"Plotting distributions for {sample}")

                make_plots(outdir, sample, args.dataset)

        if args.export_onnx:
            try:
                dummy_features = torch.randn(256, model_kwargs["input_dim"], rank=rank)
                dummy_batch = torch.zeros(256, dtype=torch.int64, rank=rank)
                torch.onnx.export(
                    model,
                    (dummy_features, dummy_batch),
                    "test.onnx",
                    verbose=True,
                    input_names=["features", "batch"],
                    output_names=["id", "momentum", "charge"],
                    dynamic_axes={
                        "features": {0: "num_elements"},
                        "batch": [0],
                        "id": [0],
                        "momentum": [0],
                        "charge": [0],
                    },
                )
            except Exception as e:
                print("ONNX export failed: {}".format(e))

    if world_size > 1:
        dist.destroy_process_group()


def main():
    args = parser.parse_args()
    world_size = len(args.gpus.split(","))  # will be 1 for both cpu ("") and single-gpu ("0")

    if args.train:  # create a new outdir when training a model to never overwrite
        outdir = create_experiment_dir(prefix=args.prefix + Path(args.config).stem + "_")
        _configLogger("mlpf", stdout=sys.stdout, filename=f"{outdir}/train.log")
        os.system(f"cp {args.config} {outdir}/train-config.yaml")
    else:
        outdir = args.load
        _configLogger("mlpf", stdout=sys.stdout, filename=f"{outdir}/test.log")
        os.system(f"cp {args.config} {outdir}/test-config.yaml")

    if args.gpus:
        assert (
            world_size <= torch.cuda.device_count()
        ), f"--gpus is too high (specefied {world_size} gpus but only {torch.cuda.device_count()} gpus are available)"

        torch.cuda.empty_cache()
        if world_size > 1:
            _logger.info(f"Will use torch.nn.parallel.DistributedDataParallel() and {world_size} gpus", color="purple")
            for rank in range(world_size):
                _logger.info(torch.cuda.get_device_name(rank), color="purple")

            mp.spawn(
                run,
                args=(world_size, args, outdir),
                nprocs=world_size,
                join=True,
            )
        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
            run(rank, world_size, args, outdir)

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")
        run(rank, world_size, args, outdir)


if __name__ == "__main__":
    main()
