"""
Developing a finetuning script of MLPF on a downstream task of MET regression.

Authors: Farouk Mokhtar
"""

import argparse
import logging
import os
import pickle as pkl
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from pyg.logger import _configLogger, _logger
from pyg.mlpf import MLPF
from pyg.PFDataset import get_interleaved_dataloaders
from pyg.utils import count_parameters, load_checkpoint, save_HPs
from utils import create_experiment_dir

from mlpf.pyg.finetuning import (
    configure_model_trainable,
    finetune_mlpf,
    override_config,
)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

# add default=None to all arparse arguments to ensure they do not override
# values loaded from the config file given by --config unless explicitly given
parser.add_argument("--config", type=str, default=None, help="yaml config")
parser.add_argument("--prefix", type=str, default=None, help="prefix appended to result dir name")
parser.add_argument("--data-dir", type=str, default=None, help="path to `tensorflow_datasets/`")
parser.add_argument("--gpus", type=int, default=None, help="to use CPU set to 0; else e.g., 4")
parser.add_argument(
    "--gpu-batch-multiplier", type=int, default=None, help="Increase batch size per GPU by this constant factor"
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["clic", "cms", "delphes", "clic_hits"],
    required=False,
    help="which dataset?",
)
parser.add_argument("--num-workers", type=int, default=None, help="number of processes to load the data")
parser.add_argument("--prefetch-factor", type=int, default=None, help="number of samples to fetch & prefetch at every call")
parser.add_argument("--load", type=str, default=None, help="load checkpoint and start new training from epoch 1")
parser.add_argument("--train", action="store_true", default=None, help="initiates a training")
parser.add_argument("--test", action="store_true", default=None, help="tests the model")
parser.add_argument("--num-epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--patience", type=int, default=None, help="patience before early stopping")
parser.add_argument("--lr", type=float, default=None, help="learning rate")
parser.add_argument(
    "--conv-type",
    type=str,
    default=None,
    help="which graph layer to use",
    choices=["gravnet", "attention", "gnn_lsh", "mamba"],
)
parser.add_argument("--num-convs", type=int, default=None, help="number of convlution (GNN, attention, Mamba) layers")
parser.add_argument("--make-plots", action="store_true", default=None, help="make plots of the test predictions")
parser.add_argument("--export-onnx", action="store_true", default=None, help="exports the model to onnx")
parser.add_argument("--ntrain", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--ntest", type=int, default=None, help="training samples to use, if None use entire dataset")
parser.add_argument("--nvalid", type=int, default=None, help="validation samples to use")
parser.add_argument("--val-freq", type=int, default=None, help="run extra validation every val_freq training steps")
parser.add_argument("--checkpoint-freq", type=int, default=None, help="epoch frequency for checkpointing")
parser.add_argument("--in-memory", action="store_true", default=None, help="if True will load the data into memory first")
parser.add_argument("--numtrain", type=int, default=10000, help="training samples to use")
parser.add_argument("--numvalid", type=int, default=1000, help="validation samples to use")

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

# finetuning args
parser.add_argument(
    "--downstream-input",
    type=str,
    required=True,
    choices=["pfcands", "mlpfcands", "latents"],
    help="input to the downstream",
)

parser.add_argument(
    "--backbone-mode",
    type=str,
    required=True,
    choices=["freeze", "float"],
    help="if freeze: will freeze the MLPF backbone before the downstream training, else float",
)

parser.add_argument(
    "--reinitialize-backbone",
    action="store_true",
    default=None,
    help="if True will reinitialize the MLPF backbone before the downstream training",
)


def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        act(),
        torch.nn.LayerNorm(width),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


class DeepMET(nn.Module):
    def __init__(
        self,
        input_dim=11,
        output_dim=2,
        width=256,
        dropout=0,
    ):
        super(DeepMET, self).__init__()

        """
        Takes as input either (1) the MLPF candidates OR (2) the latent representations of the MLPF candidates,
        and runs an MLP to predict two outputs per candidate: "w_xi" and "w_yi"; which will enter the loss as follows:
            pred_met_x = sum(w_xi * pxi)
            pred_met_y = sum(w_yi * pyi)

            LOSS = Huber(true_met_x, pred_met_x) + Huber(true_met_y, pred_met_y)

        Note: default `input_dim` is 9 which stands for "clf_nodes (6) + regression_nodes (5)"
        """

        self.act = nn.ELU
        self.nn = ffn(input_dim, output_dim, width, self.act, dropout)

    # @torch.compile
    def forward(self, X):

        MET = self.nn(X)

        return MET[:, :, 0], MET[:, :, 1]


def main():
    args = parser.parse_args()

    world_size = args.gpus if args.gpus > 0 else 1  # will be 1 for both cpu (args.gpu < 1) and single-gpu (1)

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    assert config["load"], "Must pass an MLPF model to --load"

    if "best_weights" in Path(config["load"]).name:
        backbone_dir = str(Path(config["load"]).parent)
    else:
        # the checkpoint is provided directly
        backbone_dir = str(Path(config["load"]).parent.parent)

    # outdir is the directory of the finetuned model
    outdir = create_experiment_dir(
        prefix=(args.prefix or "") + "_",
        experiments_dir=backbone_dir,
    )

    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    config_filename = "train-config.yaml" if args.train else "test-config.yaml"
    with open((Path(outdir) / config_filename), "w") as file:
        yaml.dump(config, file)

    logfile = f"{outdir}/train.log"
    _configLogger("mlpf", filename=logfile)

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
                args=(world_size, config, args, backbone_dir, outdir, logfile),
                nprocs=world_size,
                join=True,
            )
        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
            run(rank, world_size, config, args, backbone_dir, outdir, logfile)

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")
        run(rank, world_size, config, args, backbone_dir, outdir, logfile)


def run(rank, world_size, config, args, backbone_dir, outdir, logfile):

    if (rank == 0) or (rank == "cpu"):  # keep writing the logs
        _configLogger("mlpf", filename=logfile)

    use_cuda = rank != "cpu"

    dtype = getattr(torch, config["dtype"])
    _logger.info("using dtype={}".format(dtype))

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    # ----------------------- Backbone model -----------------------

    with open(f"{backbone_dir}/model_kwargs.pkl", "rb") as f:
        mlpf_kwargs = pkl.load(f)

    if config["conv_type"] == "attention":
        mlpf_kwargs["attention_type"] = config["model"]["attention"]["attention_type"]

    mlpf = MLPF(**mlpf_kwargs).to(torch.device(rank))

    if not args.reinitialize_backbone:
        checkpoint = torch.load(config["load"], map_location=torch.device(rank))

        for k in mlpf.state_dict().keys():
            shp0 = mlpf.state_dict()[k].shape
            shp1 = checkpoint["model_state_dict"][k].shape
            if shp0 != shp1:
                raise Exception("shape mismatch in {}, {}!={}".format(k, shp0, shp1))

        if (rank == 0) or (rank == "cpu"):
            _logger.info("mlpf_kwargs: {}".format(mlpf_kwargs))
            _logger.info("Loaded model weights from {}".format(config["load"]), color="bold")

        mlpf = load_checkpoint(checkpoint, mlpf)

    mlpf.to(rank)

    if world_size > 1:
        mlpf = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mlpf)
        mlpf = torch.nn.parallel.DistributedDataParallel(mlpf, device_ids=[rank])

    configure_model_trainable(mlpf, [] if args.backbone_mode == "freeze" else "all", True)
    trainable_params, nontrainable_params, table = count_parameters(mlpf)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(mlpf)
        _logger.info(f"Backbone Trainable parameters: {trainable_params}")
        _logger.info(f"Backbone Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Backbone Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table.to_string(index=False))

    configure_model_trainable(mlpf, "all", True)
    trainable_params, nontrainable_params, table = count_parameters(mlpf)

    # ----------------------- Finetuned model -----------------------

    if (
        args.downstream_input == "latents"
    ):  # the dimension will be the same as the input to one of the regression MLPs (e.g. pt)

        deepmet_input_dim = (
            mlpf.module.nn_pt.nn[0].in_features
            if isinstance(mlpf, torch.nn.parallel.distributed.DistributedDataParallel)
            else mlpf.nn_pt.nn[0].in_features
        )
    else:
        deepmet_input_dim = 5 + 6  # p4 + PID

    deepmet = DeepMET(input_dim=deepmet_input_dim)
    optimizer = (
        torch.optim.AdamW(deepmet.parameters(), lr=args.lr)
        if args.backbone_mode == "freeze"
        else torch.optim.AdamW(list(deepmet.parameters()) + list(mlpf.parameters()), lr=args.lr)
    )
    deepmet.to(rank)

    if world_size > 1:
        deepmet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(deepmet)
        deepmet = torch.nn.parallel.DistributedDataParallel(deepmet, device_ids=[rank])

    configure_model_trainable(deepmet, "all", True)
    trainable_params, nontrainable_params, table = count_parameters(deepmet)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(deepmet)
        _logger.info(f"DeepMET Trainable parameters: {trainable_params}")
        _logger.info(f"DeepMET Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"DeepMET Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table.to_string(index=False))

    # ----------------------- Training -----------------------

    if args.train:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(args, deepmet, mlpf_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info("Creating experiment dir {}".format(outdir))
            _logger.info(f"Model directory {outdir}", color="bold")

        loaders = get_interleaved_dataloaders(
            world_size,
            rank,
            config,
            use_cuda,
            use_ray=False,
        )

        finetune_mlpf(
            rank,
            world_size,
            deepmet,
            mlpf,
            args.args.backbone_mode,
            args.downstream_input,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_epochs"],
            config["patience"],
            outdir,
            trainable="all",
            dtype=dtype,
            checkpoint_freq=config["checkpoint_freq"],
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":

    # e.g.
    # noqa: python mlpf/met_finetuning_pipeline.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic-ttbar.yaml  --gpus 2 --num-epochs 200 --patience 30 --lr 1e-4 --train --load /pfvol/experiments/MLPF_clic_backbone_pyg-clic_20240429_101112_971749/best_weights.pth --checkpoint-freq 1 --num-workers 2 --prefetch-factor 2 --prefix mlpf --use-latentX  --conv-type attention --attention-type math --dtype float32 --gpu-batch-multiplier 20
    main()
