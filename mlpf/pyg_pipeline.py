"""
Developing a PyTorch Geometric supervised training of MLPF.

Author: Farouk Mokhtar
"""

import argparse
import logging
import os
import sys

sys.path.append("pyg/")

import torch
import torch.distributed as dist
import yaml
from pyg import tfds_utils
from pyg.evaluate import make_plots, make_predictions
from pyg.logger import _logger
from pyg.mlpf import MLPF
from pyg.training import train_mlpf
from pyg.utils import CLASS_LABELS, X_FEATURES, load_mlpf, save_mlpf

# import ray
# import ray.data
# # from ray import train
# from ray.air import Checkpoint, session
# from ray.air.config import CheckpointConfig, RunConfig, ScalingConfig
# from ray.train.torch import TorchConfig, TorchTrainer

logging.basicConfig(level=logging.INFO)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"


parser = argparse.ArgumentParser()

parser.add_argument("--model-prefix", type=str, default="MLPF_model", help="directory to hold the model and all plots")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrites the model if True")
parser.add_argument("--backend", type=str, choices=["gloo", "nccl"], default=None, help="backend for distributed training")
parser.add_argument("--gpus", type=str, default="0", help="to use CPU set to empty string; else e.g., `0,1`")
parser.add_argument("--dataset", type=str, choices=["clic", "cms", "delphes"], required=True, help="which dataset")
parser.add_argument("--load", action="store_true", help="Load the model (no training)")
parser.add_argument("--train", action="store_true", help="Initiates a training")
parser.add_argument("--test", action="store_true", help="Tests the model")
parser.add_argument("--num-epochs", type=int, default=3, help="number of training epochs")
parser.add_argument("--patience", type=int, default=50, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--conv-type", type=str, default="gnn-lsh", help="choices are ['gnn-lsh', 'gravnet', 'attention']")
parser.add_argument("--make-predictions", action="store_true", help="run inference on the test data")
parser.add_argument("--make-plots", action="store_true", help="makes plots of the test predictions")


def main():
    args = parser.parse_args()

    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        assert (
            len(gpus) <= torch.cuda.device_count()
        ), f"--gpus is too high (specefied {len(gpus)} gpus but only {torch.cuda.device_count()} gpus are available)"

        if args.backend is not None:  # TODO: distributed training
            torch.distributed.init_process_group(backend=args.backend, world_size=len(gpus))
        else:
            device = torch.device(gpus[0])
    else:
        gpus = None
        device = torch.device("cpu")

    # load config from yaml
    with open("../parameters/pyg.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    if args.load:  # load a pre-trained model
        model_state, model_kwargs = load_mlpf(device, args.model_prefix)

        model = MLPF(**model_kwargs).to(device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        model.load_state_dict(model_state)

    else:  # instantiate a new model
        model_kwargs = {
            "input_dim": len(X_FEATURES[args.dataset]),
            "num_classes": len(CLASS_LABELS[args.dataset]),
            **config["model"][args.conv_type],
        }
        model = MLPF(**model_kwargs).to(device)

        # save model_kwargs and hyperparameters
        save_mlpf(args, model, model_kwargs)

    _logger.info(model)
    _logger.info(f"Saving the model at {args.model_prefix}", color="bold")

    # DistributedDataParallel
    if args.backend is not None:
        _logger.info(f"Will use torch.nn.parallel.DistributedDataParallel() and {len(gpus)} gpus", color="purple")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=local_rank)

    if args.backend is None:
        # DataParallel
        if gpus is not None and len(gpus) > 1:
            _logger.info(f"Will use torch.nn.DataParallel() and {len(gpus)} gpus", color="purple")
            model = torch.nn.DataParallel(model, device_ids=gpus)

        # Single GPU
        if gpus is not None and len(gpus) == 1:
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(0)}", color="purple")

        # CPU
        if device == torch.device("cpu"):
            _logger.info("Will use cpu", color="purple")

    if args.train:
        # model = ray.train.torch.prepare_model(model)
        train_loaders = []
        for sample in config["train_dataset"][args.dataset]:
            ds = tfds_utils.Dataset(f"{sample}:{config['train_dataset'][args.dataset][sample]['version']}", "train")
            _logger.info(f"train_dataset: {ds}, {len(ds)}", color="blue")

            train_loaders.append(
                ds.get_loader(
                    batch_size=config["train_dataset"][args.dataset][sample]["batch_size"], num_workers=2, prefetch_factor=4
                )
            )

        train_loader = tfds_utils.InterleavedIterator(train_loaders)
        valid_loader = train_loader  # TODO: fix

        _logger.info(f"Training over {args.num_epochs} epochs on the {args.dataset} dataset")

        train_mlpf(
            device,
            model,
            train_loader,
            valid_loader,
            args.num_epochs,
            args.patience,
            args.lr,
            args.model_prefix,
        )

    if args.backend:
        dist.destroy_process_group()

    if args.test:
        test_loaders = {}
        for sample in config["test_dataset"][args.dataset]:
            ds = tfds_utils.Dataset(f"{sample}:{config['test_dataset'][args.dataset][sample]['version']}", "test")
            _logger.info(f"test_dataset: {ds}, {len(ds)}", color="blue")

            test_loaders[sample] = tfds_utils.InterleavedIterator(
                [
                    ds.get_loader(
                        batch_size=config["test_dataset"][args.dataset][sample]["batch_size"],
                        num_workers=2,
                        prefetch_factor=4,
                    )
                ]
            )

        # load the best epoch state
        # best_epoch = json.load(open(f"{args.model_prefix}/best_epoch.json"))["best_epoch"]
        model_state = torch.load(args.model_prefix + "/best_epoch_weights.pth", map_location=device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        model.eval()

        for sample in test_loaders:
            _logger.info(f"Running predictions on {sample}")
            make_predictions(device, model, test_loaders[sample], args.model_prefix, sample)

    # load the predictions and make plots (must have ran make_predictions() beforehand)
    if args.make_plots:
        for sample in config["test_dataset"][args.dataset]:
            _logger.info(f"Plotting distributions for {sample}")

            make_plots(args.model_prefix, sample)

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


if __name__ == "__main__":
    main()
