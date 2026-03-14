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
os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

import yaml
from mlpf.conf import MLPFConfig
from mlpf.model.training import device_agnostic_run
from mlpf.model.distributed_ray import run_hpo, run_ray_training
from mlpf.model.PFDataset import SHARING_STRATEGY
from mlpf.utils import create_experiment_dir, load_spec
from enum import Enum


class Command(Enum):
    TRAIN = "train"
    TEST = "test"
    RAY_TRAIN = "ray-train"
    RAY_HPO = "ray-hpo"


def get_parser():
    """Create and return the ArgumentParser object."""
    parser = argparse.ArgumentParser()

    # --- Define top-level, global arguments ---
    parser.add_argument("--spec-file", type=str, required=True, help="Path to the yaml spec file (particleflow_spec.yaml)")
    parser.add_argument("--model-name", type=str, required=True, help="Model name from spec file to train")
    parser.add_argument("--production-name", type=str, required=True, help="Production name from spec file")

    parser.add_argument("--experiment-dir", type=str, help="The directory where to save the weights and configs. If None, create a new one.")
    parser.add_argument("--prefix", type=str, help="Prefix prepended to the experiment dir name")
    parser.add_argument("--data-dir", type=str, help="Path to the `tensorflow_datasets/` directory")
    parser.add_argument("--experiments-dir", type=str, help="Base directory within which trainings are stored")
    parser.add_argument("--pipeline", action="store_true", help="Flag to indicate the script is running in a CI/CD pipeline")

    # --- Create subparsers for each command ---
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Common arguments for most commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use. Set to 0 for CPU.")
    common_parser.add_argument("--load", type=str, default=None, help="Load a checkpoint")
    common_parser.add_argument("--compile", action="store_true", help="Compile the model (Torch 2.0+)")
    common_parser.add_argument("--comet", action="store_true", help="Use comet.ml logging")
    common_parser.add_argument("--comet-offline", action="store_true", help="Save comet logs locally")
    common_parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    common_parser.add_argument("--test-datasets", nargs="+", default=[], help="Test samples to process")
    common_parser.add_argument("--make-plots", action="store_true", help="Generate plots")

    # --- 'train' command parser ---
    parser_train = subparsers.add_parser("train", parents=[common_parser], help="Run standard training")
    parser_train.add_argument("--num-steps", type=int, default=None, help="Number of training steps")
    parser_train.add_argument("--lr", type=float, default=None, help="Learning rate")

    # --- 'test' command parser ---
    subparsers.add_parser("test", parents=[common_parser], help="Run evaluation")

    # --- 'ray-train' command parser ---
    parser_ray = subparsers.add_parser("ray-train", parents=[common_parser], help="Run distributed training with Ray Train")
    parser_ray.add_argument("--ray-gpus", type=int, default=0, help="GPUs per worker")
    parser_ray.add_argument("--ray-cpus", type=int, help="CPUs per worker")
    parser_ray.add_argument("--ray-local", action="store_true", help="Run Ray Train locally")

    # --- 'ray-hpo' command parser ---
    parser_hpo = subparsers.add_parser("ray-hpo", parents=[common_parser], help="Run hyperparameter optimization")
    parser_hpo.add_argument("--name", type=str, required=True, help="HPO experiment name")
    parser_hpo.add_argument("--ray-gpus", type=int, default=0, help="GPUs per trial")
    parser_hpo.add_argument("--ray-cpus", type=int, help="CPUs per trial")
    parser_hpo.add_argument("--ray-local", action="store_true", help="Run Ray Tune locally")
    parser_hpo.add_argument("--raytune-num-samples", type=int, help="Number of HPO samples")

    return parser


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    parser = get_parser()
    args, extra_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    # --- Manually set action flags based on the command, for MLPFConfig.from_spec ---
    cmd = Command(args.command)
    if cmd == Command.TRAIN:
        args.train = True
        args.test = True  # By default, run testing after training
        args.hpo = None
        args.ray_train = False
    elif cmd == Command.TEST:
        args.train = False
        args.test = True
        args.hpo = None
        args.ray_train = False
    elif cmd == Command.RAY_TRAIN:
        args.train = True
        args.test = True
        args.hpo = None
        args.ray_train = True
        args.gpus = args.ray_gpus
    elif cmd == Command.RAY_HPO:
        args.train = True
        args.test = False
        args.hpo = args.name  # Set hpo to the experiment name
        args.ray_train = False
        args.gpus = args.ray_gpus

    # Load Spec and Build Config using the new Pydantic-based system
    config_obj = MLPFConfig.from_spec(args.spec_file, args.model_name, args.production_name, args, extra_args)
    config = config_obj.model_dump()

    print("Final configuration (dot-notation):")
    flat_config = config_obj.flatten_config()
    for k in sorted(flat_config.keys()):
        print(f"  --{k} {flat_config[k]}")

    # --- Main logic based on sub-command ---
    if cmd == Command.RAY_HPO:
        run_hpo(config, args)
    else:
        experiment_dir = args.experiment_dir
        if experiment_dir is None:
            # Use model_name and production_name for prefix if available
            prefix = (args.prefix or "") + f"{args.model_name}_{args.production_name}_"
            experiment_dir = create_experiment_dir(
                prefix=prefix,
                experiments_dir=args.experiments_dir if args.experiments_dir else "experiments",
            )

        # Save config for later reference.
        config_filename = f"{cmd.value}-config.yaml"
        with open((Path(experiment_dir) / config_filename), "w") as file:
            yaml.dump(config_obj.model_dump(mode="json"), file)

        # Also save the spec file for reproducibility
        spec = load_spec(args.spec_file)
        with open((Path(experiment_dir) / "particleflow_spec.yaml"), "w") as file:
            yaml.dump(spec, file)

        if cmd == Command.RAY_TRAIN:
            run_ray_training(config, args, experiment_dir)
        elif cmd in [Command.TRAIN, Command.TEST]:
            world_size = config_obj.gpus if config_obj.gpus > 0 else 1
            device_agnostic_run(config_obj, world_size, experiment_dir)


if __name__ == "__main__":
    main()
