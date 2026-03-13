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
from mlpf.utils import create_experiment_dir, load_spec, resolve_path


def flatten_dict(d, prefix=""):
    """Flatten a nested dictionary into a dot-separated path dictionary."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


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


def build_config_from_spec(spec, model_name, production_name):
    if model_name not in spec["models"]:
        raise ValueError(f"Model {model_name} not found in spec")
    if production_name not in spec["productions"]:
        raise ValueError(f"Production {production_name} not found in spec")

    model_config = spec["models"][model_name]
    prod_config = spec["productions"][production_name]

    # Initialize config with model parameters
    config = {}

    # Merge with model defaults if present
    if "defaults" in spec["models"]:
        for k, v in spec["models"]["defaults"].items():
            if isinstance(v, str):
                v = resolve_path(v, spec)
            config[k] = v

    # Copy hyperparameters and other top-level settings
    for k, v in model_config.items():
        if k not in ["architecture", "train_datasets", "validation_datasets", "test_datasets"]:
            if isinstance(v, str):
                v = resolve_path(v, spec)
            config[k] = v

    # Handle hyperparameters specifically if they are nested
    if "hyperparameters" in model_config:
        for k, v in model_config["hyperparameters"].items():
            config[k] = v

    # Model Architecture
    config["model"] = model_config["architecture"]
    config["conv_type"] = config["model"]["type"]

    if "gnn_lsh" in config["model"]:
        config["model"]["gnn_lsh"]["conv_type"] = "gnn_lsh"
    if "attention" in config["model"]:
        config["model"]["attention"]["conv_type"] = "attention"
    config["model"]["trainable"] = "all"
    config["model"]["learned_representation_mode"] = "last"
    config["model"]["input_encoding"] = "split"
    config["model"]["pt_mode"] = "direct-elemtype-split"
    config["model"]["eta_mode"] = "linear"
    config["model"]["sin_phi_mode"] = "linear"
    config["model"]["cos_phi_mode"] = "linear"
    config["model"]["energy_mode"] = "direct-elemtype-split"

    # Dataset and Production
    config["dataset"] = model_config.get("dataset", prod_config.get("type"))

    workspace_dir = resolve_path(prod_config["workspace_dir"], spec)
    config["data_dir"] = os.path.join(workspace_dir, "tfds")

    def build_dataset_config(dataset_input):
        ds_config = {}
        ds_config[config["dataset"]] = {}
        dataset_groups = dataset_input

        for phys_key, phys_val in dataset_groups.items():
            ds_config[config["dataset"]][phys_key] = {
                "batch_size": phys_val.get("batch_size", config.get("batch_size", 1)),
                "samples": {},
            }
            target_dict = ds_config[config["dataset"]][phys_key]["samples"]

            for ds_item in phys_val["samples"]:
                name = ds_item["name"]

                entry = {}
                if "version" in ds_item:
                    entry["version"] = ds_item["version"]

                if "splits" in ds_item:
                    entry["splits"] = ds_item["splits"]

                # Copy batch size if specific
                if "batch_size" in ds_item:
                    entry["batch_size"] = ds_item["batch_size"]

                target_dict[name] = entry

        return ds_config

    if "train_datasets" in model_config:
        config["train_dataset"] = build_dataset_config(model_config["train_datasets"])

    if "validation_datasets" in model_config:
        config["valid_dataset"] = build_dataset_config(model_config["validation_datasets"])

    if "test_datasets" in model_config:
        config["test_dataset"] = {}
        for ds_item in model_config.get("test_datasets", []):
            name = ds_item["name"]
            entry = {}
            entry["version"] = ds_item.get("version")
            entry["splits"] = ds_item.get("splits", ["test"])
            entry["batch_size"] = ds_item.get("batch_size", 1)
            config["test_dataset"][name] = entry

    # Ensure some defaults for testing/validation if not present
    if "test_dataset" not in config:
        config["test_dataset"] = {}

    # Default fields expected by pipeline/training
    if "comet_name" not in config:
        config["comet_name"] = "particleflow"

    return config


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    parser = get_parser()
    args, extra_args = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)

    # Load Spec and Build Config
    spec = load_spec(args.spec_file)
    config = build_config_from_spec(spec, args.model_name, args.production_name)

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
        if "gnn_lsh" not in config["model"]:
            config["model"]["gnn_lsh"] = {}
        config["model"]["gnn_lsh"]["num_convs"] = 1
        config["model"]["gnn_lsh"]["width"] = 32
        config["model"]["gnn_lsh"]["embedding_dim"] = 32

        if "attention" not in config["model"]:
            config["model"]["attention"] = {}
        config["model"]["attention"]["num_convs"] = 1
        config["model"]["attention"]["num_heads"] = 2
        config["model"]["attention"]["head_dim"] = 2

        if config["dataset"] == "cms":
            for ds in ["train_dataset", "valid_dataset"]:
                if ds in config:
                    config[ds]["cms"] = {
                        "physical_pu": {
                            "batch_size": config[ds]["cms"]["physical_pu"]["batch_size"],
                            "samples": {"cms_pf_ttbar": {"splits": ["10"], "version": "3.0.0"}},
                        }
                    }
            if "cms_pf_ttbar" in config["test_dataset"]:
                config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]}
                config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]

    # override loaded config with values from command line args
    config = override_config(config, args, extra_args)

    print("Final configuration (dot-notation):")
    flat_config = flatten_dict(config)
    for k in sorted(flat_config.keys()):
        print(f"  --{k} {flat_config[k]}")

    # --- Main logic based on sub-command ---
    if args.command == "ray-hpo":
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
        config_filename = f"{args.command}-config.yaml"
        with open((Path(experiment_dir) / config_filename), "w") as file:
            yaml.dump(config, file)

        # Also save the spec file for reproducibility
        with open((Path(experiment_dir) / "particleflow_spec.yaml"), "w") as file:
            yaml.dump(spec, file)

        if args.command == "ray-train":
            run_ray_training(config, args, experiment_dir)
        elif args.command in ["train", "test"]:
            if args.gpus is not None:
                config["gpus"] = args.gpus
            if "gpus" not in config:
                config["gpus"] = 0
            gpus = config["gpus"]
            world_size = gpus if gpus > 0 else 1
            device_agnostic_run(config, world_size, experiment_dir)


if __name__ == "__main__":
    main()
