"""
PyTorch supervised training of MLPF using DistributedDataParallel or Ray Train.
Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff
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

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument(
    "--experiment-dir", type=str, default=None, help="The directory where to save the weights and configs. if None, create a new one."
)
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument("--prefix", type=str, default=None, help="prefix prepended to the experiment dir name")
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=int, default=None, help="to use CPU set to 0; else e.g., 4")
parser.add_argument("--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor")
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument("--load", type=str, default=None, help="load checkpoint and continue training from previous epoch")
parser.add_argument(
    "--relaxed-load",
    action="store_true",
    default=None,
    help="load parameters from the checkpoint model with the same name as the existing model, ignoring any missing parameters",
)
parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--start-epoch", type=None, default=None, help="the initial epoch counter for LR decay and logging")
parser.add_argument("--patience", type=int, default=None, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--lr-schedule",
    type=str,
    default=None,
    help="learning rate schedule to use",
    choices=["constant", "cosinedecay", "onecycle", "reduce_lr_on_plateau"],
)
parser.add_argument(
    "--optimizer",
    type=str,
    default=None,
    help="optimizer to use for training",
    choices=["adamw", "sgd", "lamb"],
)
parser.add_argument("--weight-decay", type=float, default=None, help="weight decay for the optimizer")
parser.add_argument(
    "--conv-type",
    type=str,
    default=None,
    help="which graph layer to use",
    choices=["attention", "gnn_lsh", "mamba"],
)
parser.add_argument("--num-convs", type=int, default=None, help="number of cross-particle convolution (GNN, attention, Mamba) layers")
parser.add_argument("--make-plots", action="store_true", default=None, help="make plots of the test predictions")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=None, help="validation samples to use")
parser.add_argument("--val-freq", type=int, default=None, help="run extra validation every val_freq training steps")
parser.add_argument("--checkpoint-freq", type=int, default=None, help="epoch frequency for checkpointing")
parser.add_argument("--hpo", type=str, default=None, help="perform hyperparameter optimization, name of HPO experiment")
parser.add_argument("--comet", action="store_true", help="use comet ml logging")
parser.add_argument("--comet-offline", action="store_true", help="save comet logs locally")
parser.add_argument("--comet-step-freq", type=int, default=None, help="step frequency for saving comet metrics")
parser.add_argument("--experiments-dir", type=str, default=None, help="base directory within which trainings are stored")
parser.add_argument("--pipeline", action="store_true", default=None, help="test is running in pipeline")
parser.add_argument(
    "--dtype",
    type=str,
    default=None,
    help="data type for training",
    choices=["float32", "float16", "bfloat16"],
)
parser.add_argument(
    "--attention-type",
    type=str,
    default=None,
    help="attention type for self-attention layer",
    choices=["math", "efficient", "flash", "flash_external"],
)
parser.add_argument("--test-datasets", nargs="+", default=[], help="test samples to process")

# options only used for the ray-based training
parser.add_argument("--ray-train", action="store_true", help="run training using Ray Train")
parser.add_argument("--ray-local", action="store_true", default=None, help="run ray-train locally")
parser.add_argument("--ray-cpus", type=int, default=None, help="CPUs for ray-train")
parser.add_argument("--ray-gpus", type=int, default=None, help="GPUs for ray-train")
parser.add_argument("--raytune-num-samples", type=int, default=None, help="Number of samples to draw from search space")


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    # plt.rcParams['text.usetex'] = True
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    world_size = args.gpus if args.gpus > 0 else 1  # will be 1 for both cpu (args.gpu < 1) and single-gpu (1)

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override some options for the pipeline test
    if args.pipeline:
        config["model"]["gnn_lsh"]["num_convs"] = 1
        config["model"]["gnn_lsh"]["width"] = 64
        config["model"]["gnn_lsh"]["embedding_dim"] = 64

        config["model"]["attention"]["num_convs"] = 1
        config["model"]["attention"]["num_heads"] = 8
        config["model"]["attention"]["head_dim"] = 8

        if config["dataset"] == "cms":
            for ds in ["train_dataset", "valid_dataset"]:
                config[ds]["cms"] = {
                    "physical_pu": {
                        "batch_size": config[ds]["cms"]["physical_pu"]["batch_size"],
                        "samples": {"cms_pf_ttbar": config[ds]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]},
                    }
                }
                # load only the last config split
                config[ds]["cms"]["physical_pu"]["samples"]["cms_pf_ttbar"]["splits"] = ["10"]
            config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]}
            config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]

    # override loaded config with values from command line args
    config = override_config(config, args)

    if args.hpo:
        run_hpo(config, args)
    else:
        experiment_dir = args.experiment_dir
        if experiment_dir is None:
            experiment_dir = create_experiment_dir(
                prefix=(args.prefix or "") + Path(args.config).stem + "_",
                experiments_dir=args.experiments_dir if args.experiments_dir else "experiments",
            )

        # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
        config_filename = "train-config.yaml" if args.train else "test-config.yaml"
        with open((Path(experiment_dir) / config_filename), "w") as file:
            yaml.dump(config, file)

        if args.ray_train:
            run_ray_training(config, args, experiment_dir)
        else:
            device_agnostic_run(config, world_size, experiment_dir)


if __name__ == "__main__":
    main()
