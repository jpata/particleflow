"""
PyTorch supervised training of MLPF on CMS or EDM4HEP data.
- Supports single-node, multi-GPU training using DistributedDataParallel.
- Supports single-node, multi-GPU and multi-node training using Ray.
- Supports AMD and Nvidia GPUs.

Main authors: Joosep Pata, Farouk Mokhtar, Eric Wulff, Javier Duarte
Full list of authors: https://github.com/jpata/particleflow/graphs/contributors
"""

import argparse
import logging
import os
from pathlib import Path

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import yaml
from mlpf.model.training import device_agnostic_run, override_config
from mlpf.model.distributed_ray import run_hpo, run_ray_training
from mlpf.model.PFDataset import SHARING_STRATEGY
from utils import create_experiment_dir

# import habana if available
try:
    import habana_frameworks.torch.core as htcore
except ImportError:
    pass


def get_parser():
    """Create and return the ArgumentParser object."""
    parser = argparse.ArgumentParser()

    # --- Define top-level, global arguments ---
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file")
    parser.add_argument("--experiment-dir", type=str, help="The directory where to save the weights and configs. If None, create a new one.")
    parser.add_argument("--prefix", type=str, help="Prefix prepended to the experiment dir name")
    parser.add_argument("--data-dir", type=str, help="Path to the `tensorflow_datasets/` directory")
    parser.add_argument("--experiments-dir", type=str, help="Base directory within which trainings are stored")
    parser.add_argument("--pipeline", action="store_true", help="Flag to indicate the script is running in a CI/CD pipeline")

    # --- Create subparsers for each command ---
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- 'train' command parser ---
    parser_train = subparsers.add_parser("train", help="Run standard training on a single node (CPU, single-GPU, or DDP)")
    parser_train.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use. Set to 0 for CPU.")
    parser_train.add_argument("--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor")
    parser_train.add_argument("--num-workers", type=int, default=None, help="Number of processes to load data")
    parser_train.add_argument("--prefetch-factor", type=int, default=None, help="Number of samples to fetch & prefetch per worker")
    parser_train.add_argument("--load", type=str, default=None, help="Load a checkpoint and continue training")
    parser_train.add_argument("--relaxed-load", action="store_true", help="Loosely load model parameters, ignoring missing keys")
    parser_train.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser_train.add_argument("--start-epoch", type=int, default=None, help="The initial epoch counter")
    parser_train.add_argument("--patience", type=int, default=None, help="Patience before early stopping")
    parser_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser_train.add_argument("--lr-schedule", type=str, default=None, choices=["constant", "cosinedecay", "onecycle", "reduce_lr_on_plateau"])
    parser_train.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd", "lamb"])
    parser_train.add_argument("--weight-decay", type=float, default=None, help="Weight decay for the optimizer")
    parser_train.add_argument("--conv-type", type=str, default=None, choices=["attention", "gnn_lsh"])
    parser_train.add_argument("--num-convs", type=int, default=None, help="Number of GNN/Attention layers")
    parser_train.add_argument("--ntrain", type=int, default=None, help="Number of training samples to use")
    parser_train.add_argument("--nvalid", type=int, default=None, help="Number of validation samples to use")
    parser_train.add_argument("--ntest", type=int, default=None, help="Number of test samples to use")
    parser_train.add_argument("--val-freq", type=int, default=None, help="Run validation every N training steps")
    parser_train.add_argument("--checkpoint-freq", type=int, default=None, help="Epoch frequency for checkpointing")
    parser_train.add_argument("--comet", action="store_true", help="Use comet.ml logging")
    parser_train.add_argument("--comet-offline", action="store_true", help="Save comet logs locally")
    parser_train.add_argument("--comet-step-freq", type=int, default=None, help="Step frequency for comet logging")
    parser_train.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    parser_train.add_argument("--attention-type", type=str, default=None, choices=["math", "efficient", "flash"])
    parser_train.add_argument("--test-datasets", nargs="+", default=[], help="Test samples to process after training")
    parser_train.add_argument("--make-plots", action="store_true", help="Generate plots of test predictions")

    # --- 'test' command parser ---
    parser_test = subparsers.add_parser("test", help="Run evaluation on a trained model")
    parser_test.add_argument("--load", type=str, required=True, help="Path to a saved model checkpoint to test")
    parser_test.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use. Set to 0 for CPU.")
    parser_test.add_argument("--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor")
    parser_test.add_argument("--num-workers", type=int, default=None, help="Number of processes to load data")
    parser_test.add_argument("--prefetch-factor", type=int, default=None, help="Number of samples to fetch & prefetch per worker")
    parser_test.add_argument("--ntest", type=int, default=None, help="Number of test samples to use")
    parser_test.add_argument("--comet", action="store_true", help="Use comet.ml logging")
    parser_test.add_argument("--comet-offline", action="store_true", help="Save comet logs locally")
    parser_test.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    parser_test.add_argument("--attention-type", type=str, default=None, choices=["math", "efficient", "flash"])
    parser_test.add_argument("--test-datasets", nargs="+", default=[], help="Test samples to process")
    parser_test.add_argument("--make-plots", action="store_true", help="Generate plots of test predictions")
    parser_test.add_argument("--num-convs", type=int, default=None, help="Number of GNN/Attention layers")

    # --- 'ray-train' command parser ---
    parser_ray = subparsers.add_parser("ray-train", help="Run distributed training with Ray Train")
    parser_ray.add_argument("--ray-gpus", type=int, default=0, help="GPUs per worker for Ray Train")
    parser_ray.add_argument("--ray-cpus", type=int, help="CPUs per worker for Ray Train")
    parser_ray.add_argument("--ray-local", action="store_true", help="Run Ray Train cluster locally")
    parser_ray.add_argument("--num-epochs", type=int, default=None, help="Number of training epochs")
    parser_ray.add_argument("--load", type=str, default=None, help="Load a checkpoint and continue training")
    parser_ray.add_argument("--comet", action="store_true", help="Use comet.ml logging")
    parser_ray.add_argument("--comet-offline", action="store_true", help="Save comet logs locally")
    parser_ray.add_argument("--comet-step-freq", type=int, default=None, help="Step frequency for comet logging")
    parser_ray.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    parser_ray.add_argument("--attention-type", type=str, default=None, choices=["math", "efficient", "flash"])
    parser_ray.add_argument("--conv-type", type=str, default=None, choices=["attention", "gnn_lsh"])
    parser_ray.add_argument("--num-convs", type=int, default=None, help="Number of GNN/Attention layers")
    parser_ray.add_argument("--test-datasets", nargs="+", default=[], help="Test samples to process after training")
    parser_ray.add_argument("--make-plots", action="store_true", help="Generate plots of test predictions")
    parser_ray.add_argument("--start-epoch", type=int, default=None, help="The initial epoch counter")

    # --- 'ray-hpo' command parser ---
    parser_hpo = subparsers.add_parser("ray-hpo", help="Run hyperparameter optimization with Ray Tune")
    parser_hpo.add_argument("--name", type=str, required=True, help="Name of the HPO experiment")
    parser_hpo.add_argument("--ray-gpus", type=int, default=0, help="GPUs per trial for Ray Tune")
    parser_hpo.add_argument("--ray-cpus", type=int, help="CPUs per trial for Ray Tune")
    parser_hpo.add_argument("--ray-local", action="store_true", help="Run Ray Tune locally")
    parser_hpo.add_argument("--raytune-num-samples", type=int, help="Number of samples to draw from the search space")
    parser_hpo.add_argument("--comet", action="store_true", help="Use comet.ml logging")

    # option for habana training
    parser_train.add_argument("--habana", action="store_true", default=None, help="use Habana Gaudi device")
    parser_test.add_argument("--habana", action="store_true", default=None, help="use Habana Gaudi device")
    return parser


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # --- Manually set action flags based on the command, for override_config ---
    if args.command == "train":
        args.train = True
        args.test = True  # By default, run testing after training
        args.hpo = None
        args.ray_train = False
    elif args.command == "test":
        args.train = False
        args.test = True
        args.hpo = None
        args.ray_train = False
    elif args.command == "ray-train":
        args.train = True
        args.test = True
        args.hpo = None
        args.ray_train = True
        args.gpus = args.ray_gpus
    elif args.command == "ray-hpo":
        args.train = True
        args.test = False
        args.hpo = args.name  # Set hpo to the experiment name
        args.ray_train = False
        args.gpus = args.ray_gpus

    # override some options for the pipeline test
    if args.pipeline:
        config["model"]["gnn_lsh"]["num_convs"] = 1
        config["model"]["gnn_lsh"]["width"] = 32
        config["model"]["gnn_lsh"]["embedding_dim"] = 32

        config["model"]["attention"]["num_convs"] = 1
        config["model"]["attention"]["num_heads"] = 2
        config["model"]["attention"]["head_dim"] = 2

        if config["dataset"] == "cms":
            for ds in ["train_dataset", "valid_dataset"]:
                config[ds]["cms"] = {
                    "physical_pu": {
                        "batch_size": config[ds]["cms"]["physical_pu"]["batch_size"],
                        "samples": {"cms_pf_ttbar": config[ds]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]},
                    }
                }
                # load only the first config split
                config[ds]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"] = ["1"]
            config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]}
            config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["1"]

    # override loaded config with values from command line args
    config = override_config(config, args)

    # --- Main logic based on sub-command ---
    if args.command == "ray-hpo":
        run_hpo(config, args)
    else:
        experiment_dir = args.experiment_dir
        if experiment_dir is None:
            experiment_dir = create_experiment_dir(
                prefix=(args.prefix or "") + Path(args.config).stem + "_",
                experiments_dir=args.experiments_dir if args.experiments_dir else "experiments",
            )

        # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
        config_filename = f"{args.command}-config.yaml"
        with open((Path(experiment_dir) / config_filename), "w") as file:
            yaml.dump(config, file)

        if args.command == "ray-train":
            run_ray_training(config, args, experiment_dir)
        elif args.command in ["train", "test"]:
            world_size = args.gpus if args.gpus > 0 else 1
            device_agnostic_run(config, world_size, experiment_dir, args.habana)


if __name__ == "__main__":
    main()
