"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Author: Farouk Mokhtar
"""

import argparse
import logging
import os
import os.path as osp
import pickle as pkl
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import fastjet
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from pyg.inference import make_plots, run_predictions
from pyg.logger import _configLogger, _logger
from pyg.mlpf import MLPF
from pyg.PFDataset import Collater, InterleavedIterator, PFDataLoader, PFDataset
from pyg.training import train_mlpf
from pyg.utils import CLASS_LABELS, X_FEATURES, save_HPs
from utils import create_experiment_dir

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument("--prefix", type=str, default=None, help="prefix appended to result dir name")
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=str, default=None, help="to use CPU set to empty string; else e.g., `0,1`")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor"
)
parser.add_argument(
    "--dataset", type=str, default=None, choices=["clic", "cms", "delphes"], required=False, help="which dataset?"
)
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument("--load", type=str, default=None, help="dir from which to load a saved model")
parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--patience", type=int, default=None, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--conv-type", type=str, default="gravnet", help="which graph layer to use", choices=["gravnet", "attention", "gnn_lsh"]
)
parser.add_argument("--make-plots", action="store_true", default=None, help="make plots of the test predictions")
parser.add_argument("--export-onnx", action="store_true", default=None, help="exports the model to onnx")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=500, help="validation samples to use, default will use 500 events")
parser.add_argument("--hpo", type=str, default=None, help="perform hyperparameter optimization, name of HPO experiment")
parser.add_argument("--local", action="store_true", default=None, help="perform HPO locally, without a Ray cluster")
parser.add_argument("--ray-cpus", type=int, default=None, help="CPUs per trial for HPO")
parser.add_argument("--ray-gpus", type=int, default=None, help="GPUs per trial for HPO")


def run(rank, world_size, config, args, outdir, logfile):
    """Demo function that will be passed to each gpu if (world_size > 1) else will run normally on the given device."""

    pad_3d = args.conv_type != "gravnet"
    use_cuda = rank != "cpu"

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    if (rank == 0) or (rank == "cpu"):  # keep writing the logs
        _configLogger("mlpf", filename=logfile)

    if config["load"]:  # load a pre-trained model
        outdir = config["load"]  # in case both --load and --train are provided

        with open(f"{outdir}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)

        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

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
            "input_dim": len(X_FEATURES[config["dataset"]]),
            "num_classes": len(CLASS_LABELS[config["dataset"]]),
            **config["model"][config["conv_type"]],
        }
        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    model.to(rank)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)

    if args.train:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info("Creating experiment dir {}".format(outdir))
            _logger.info(f"Model directory {outdir}", color="bold")

        loaders = {}
        for split in ["train", "valid"]:  # build train, valid dataset and dataloaders
            loaders[split] = []
            # build dataloader for physical and gun samples seperately
            for type_ in config[f"{split}_dataset"][config["dataset"]]:  # will be "physical", "gun"
                dataset = []
                for sample in config[f"{split}_dataset"][config["dataset"]][type_]["samples"]:
                    version = config[f"{split}_dataset"][config["dataset"]][type_]["samples"][sample]["version"]

                    ds = PFDataset(config["data_dir"], f"{sample}:{version}", split, num_samples=config[f"n{split}"]).ds

                    if (rank == 0) or (rank == "cpu"):
                        _logger.info(f"{split}_dataset: {sample}, {len(ds)}", color="blue")

                    dataset.append(ds)
                dataset = torch.utils.data.ConcatDataset(dataset)

                # build dataloaders
                batch_size = (
                    config[f"{split}_dataset"][config["dataset"]][type_]["batch_size"] * config["gpu_batch_multiplier"]
                )

                if world_size > 1:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                else:
                    sampler = torch.utils.data.RandomSampler(dataset)

                loaders[split].append(
                    PFDataLoader(
                        dataset,
                        batch_size=batch_size,
                        collate_fn=Collater(["X", "ygen"], pad_3d=pad_3d),
                        sampler=sampler,
                        num_workers=config["num_workers"],
                        prefetch_factor=config["prefetch_factor"],
                        pin_memory=use_cuda,
                        pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                    )
                )

            loaders[split] = InterleavedIterator(loaders[split])  # will interleave just two dataloaders

        train_mlpf(
            rank,
            world_size,
            model,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_epochs"],
            config["patience"],
            outdir,
            hpo=True if args.hpo is not None else False,
        )

    if args.test:
        if config["load"] is None:
            # if we don't load, we must have a newly trained model
            assert args.train, "Please train a model before testing, or load a model with --load"
            assert outdir is not None, "Error: no outdir to evaluate model from"
        else:
            outdir = config["load"]

        test_loaders = {}
        for type_ in config["test_dataset"][config["dataset"]]:  # will be "physical", "gun"
            batch_size = config["test_dataset"][config["dataset"]][type_]["batch_size"] * config["gpu_batch_multiplier"]
            for sample in config["test_dataset"][config["dataset"]][type_]["samples"]:
                version = config["test_dataset"][config["dataset"]][type_]["samples"][sample]["version"]

                ds = PFDataset(config["data_dir"], f"{sample}:{version}", "test", num_samples=config["ntest"]).ds

                if (rank == 0) or (rank == "cpu"):
                    _logger.info(f"test_dataset: {sample}, {len(ds)}", color="blue")

                if world_size > 1:
                    sampler = torch.utils.data.distributed.DistributedSampler(ds)
                else:
                    sampler = torch.utils.data.RandomSampler(ds)

                test_loaders[sample] = PFDataLoader(
                    ds,
                    batch_size=batch_size,
                    collate_fn=Collater(["X", "ygen", "ycand"], pad_3d=False),  # in inference, use sparse dataset
                    sampler=sampler,
                    num_workers=config["num_workers"],
                    prefetch_factor=config["prefetch_factor"],
                    pin_memory=use_cuda,
                    pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                )

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
            for sample in config["test_dataset"][config["dataset"]]:
                _logger.info(f"Plotting distributions for {sample}")

                make_plots(outdir, sample, config["dataset"])

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


def override_config(config, args):
    """override config with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            config[arg] = arg_value
    return config


def device_agnostic_run(config, args, world_size, outdir):
    if args.train:  # create a new outdir when training a model to never overwrite
        logfile = f"{outdir}/train.log"
        _configLogger("mlpf", filename=logfile)

        os.system(f"cp {args.config} {outdir}/train-config.yaml")
    else:
        outdir = args.load
        logfile = f"{outdir}/test.log"
        _configLogger("mlpf", filename=logfile)

        os.system(f"cp {args.config} {outdir}/test-config.yaml")

    if config["gpus"]:
        assert (
            world_size <= torch.cuda.device_count()
        ), f"--gpus is too high (specified {world_size} gpus but only {torch.cuda.device_count()} gpus are available)"

        torch.cuda.empty_cache()
        if world_size > 1:
            _logger.info(f"Will use torch.nn.parallel.DistributedDataParallel() and {world_size} gpus", color="purple")
            for rank in range(world_size):
                _logger.info(torch.cuda.get_device_name(rank), color="purple")

            mp.spawn(
                run,
                args=(world_size, config, args, outdir, logfile),
                nprocs=world_size,
                join=True,
            )
        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
            run(rank, world_size, config, args, outdir, logfile)

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")
        run(rank, world_size, config, args, outdir, logfile)


def main():
    # torch.multiprocessing.set_start_method('spawn')

    args = parser.parse_args()
    world_size = len(args.gpus.split(","))  # will be 1 for both cpu ("") and single-gpu ("0")

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    if args.hpo:
        import ray
        from ray import train as ray_train
        from ray import tune

        # from ray.tune.logger import TBXLoggerCallback
        from raytune.pt_search_space import raytune_num_samples, search_space, set_hps_from_search_space
        from raytune.utils import get_raytune_schedule, get_raytune_search_alg

        name = args.hpo  # name of Ray Tune experiment directory

        os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"  # don't crash if a metric is missing
        if isinstance(config["raytune"]["local_dir"], type(None)):
            raise TypeError("Please specify a local_dir in the raytune section of the config file.")
        trd = config["raytune"]["local_dir"] + "/tune_result_dir"
        os.environ["TUNE_RESULT_DIR"] = trd

        expdir = Path(config["raytune"]["local_dir"]) / name
        expdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            "mlpf/raytune/search_space.py",
            str(Path(config["raytune"]["local_dir"]) / name / "search_space.py"),
        )  # Copy the config file to the train dir for later reference
        shutil.copy(
            args.config,
            str(Path(config["raytune"]["local_dir"]) / name / "config.yaml"),
        )  # Copy the config file to the train dir for later reference

        if not args.local:
            ray.init(address="auto")

        sched = get_raytune_schedule(config["raytune"])
        search_alg = get_raytune_search_alg(config["raytune"])

        def hpo(search_space, config, args, world_size):
            config = set_hps_from_search_space(search_space, config)
            outdir = ray_train.get_context().get_trial_dir()
            device_agnostic_run(config, args, world_size, outdir)

        start = datetime.now()
        analysis = tune.run(
            partial(
                hpo,
                config=config,
                args=args,
                world_size=world_size,
            ),
            config=search_space,
            resources_per_trial={"cpu": args.ray_cpus, "gpu": args.ray_gpus},
            name=name,
            scheduler=sched,
            search_alg=search_alg,
            num_samples=raytune_num_samples,
            local_dir=config["raytune"]["local_dir"],
            # callbacks=[TBXLoggerCallback()],
            log_to_file=True,
            resume=False,  # TODO: make this configurable
            max_failures=2,
            # sync_config=sync_config,
        )
        end = datetime.now()
        logging.info("Total time of tune.run(...): {}".format(end - start))

        logging.info(
            "Best hyperparameters found according to {} were: ".format(config["raytune"]["default_metric"]),
            analysis.get_best_config(
                metric=config["raytune"]["default_metric"],
                mode=config["raytune"]["default_mode"],
                scope="all",
            ),
        )

    else:
        outdir = create_experiment_dir(prefix=(args.prefix or "") + Path(args.config).stem + "_")
        device_agnostic_run(config, args, world_size, outdir)


if __name__ == "__main__":
    main()
