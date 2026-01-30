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

    # --- 'train' command parser ---
    parser_train = subparsers.add_parser("train", help="Run standard training on a single node (CPU, single-GPU, or DDP)")
    parser_train.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use. Set to 0 for CPU.")
    parser_train.add_argument("--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor")
    parser_train.add_argument("--num-workers", type=int, default=None, help="Number of processes to load data")
    parser_train.add_argument("--prefetch-factor", type=int, default=None, help="Number of samples to fetch & prefetch per worker")
    parser_train.add_argument("--load", type=str, default=None, help="Load a checkpoint and continue training")
    parser_train.add_argument("--relaxed-load", action="store_true", help="Loosely load model parameters, ignoring missing keys")
    parser_train.add_argument("--num-steps", type=int, default=None, help="Number of training steps")
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
    parser_test.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use. Set to 0 for CPU.")
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
    parser_ray.add_argument("--num-steps", type=int, default=None, help="Number of training steps")
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

    # --- 'ray-hpo' command parser ---
    parser_hpo = subparsers.add_parser("ray-hpo", help="Run hyperparameter optimization with Ray Tune")
    parser_hpo.add_argument("--name", type=str, required=True, help="Name of the HPO experiment")
    parser_hpo.add_argument("--ray-gpus", type=int, default=0, help="GPUs per trial for Ray Tune")
    parser_hpo.add_argument("--ray-cpus", type=int, help="CPUs per trial for Ray Tune")
    parser_hpo.add_argument("--ray-local", action="store_true", help="Run Ray Tune locally")
    parser_hpo.add_argument("--raytune-num-samples", type=int, help="Number of samples to draw from the search space")
    parser_hpo.add_argument("--comet", action="store_true", help="Use comet.ml logging")

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
    config["load"] = None
    config["num_steps"] = 100000
    config["comet"] = False
    config["comet_step_freq"] = 1000
    config["ntrain"] = None
    config["ntest"] = None
    config["nvalid"] = None
    config["sort_data"] = False
    config["num_workers"] = 1
    config["prefetch_factor"] = 1
    config["patience"] = 1000
    config["checkpoint_freq"] = 1000
    config["val_freq"] = 1000

    # Copy hyperparameters and other top-level settings
    for k, v in model_config.items():
        if k not in ["architecture", "train_datasets", "validation_datasets", "test_datasets"]:
            print(k, v)
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

    def build_dataset_config(dataset_list):
        ds_config = {}

        if config["dataset"] == "cms":
            phys_key = "physical_pu"
        else:
            phys_key = "physical"

        ds_config[config["dataset"]] = {
            phys_key: {
                "batch_size": config.get("batch_size", 1),
                "samples": {},
            }
        }
        target_dict = ds_config[config["dataset"]][phys_key]["samples"]

        for ds_item in dataset_list:
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
            entry["version"] = ds_item.get("version", "2.8.0")
            entry["splits"] = ds_item.get("splits", ["test"])
            entry["batch_size"] = ds_item.get("batch_size", 1)
            config["test_dataset"][name] = entry

    # Ensure some defaults for testing/validation if not present
    if "test_dataset" not in config:
        config["test_dataset"] = {}

    # Default fields expected by pipeline/training
    if "comet_name" not in config:
        config["comet_name"] = "particleflow"
    print(config)
    return config


def main():
    # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
    import torch

    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

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
                            "samples": {"cms_pf_ttbar": {"splits": ["10"], "version": "2.8.0"}},
                        }
                    }
        # config["test_dataset"] = {"cms_pf_ttbar": config["test_dataset"]["cms_pf_ttbar"]} # This line in original code might fail if key missing
        # config["test_dataset"]["cms_pf_ttbar"]["splits"] = ["10"]

    # override loaded config with values from command line args
    config = override_config(config, args)

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
            gpus = config.get("gpus", 0)
            world_size = gpus if gpus > 0 else 1
            device_agnostic_run(config, world_size, experiment_dir)


if __name__ == "__main__":
    main()
