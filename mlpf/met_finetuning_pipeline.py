"""
Developing a PyTorch Geometric supervised training of MLPF using DistributedDataParallel.

Authors: Farouk Mokhtar, Joosep Pata, Eric Wulff
"""

import argparse
import logging
import pickle as pkl
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from pyg.logger import _configLogger, _logger
from pyg.mlpf import MLPF
from pyg.PFDataset import get_interleaved_dataloaders
from pyg.training_met import override_config, train_mlpf
from pyg.utils import load_checkpoint, save_HPs
from utils import create_experiment_dir

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


parser.add_argument(
    "--use-latentX", action="store_true", default=None, help="if True will use the latent representations of MLPF"
)

parser.add_argument(
    "--freeze-backbone",
    action="store_true",
    default=None,
    help="if True will freeze the MLPF backbone before the downstream training",
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

    with open(args.config, "r") as stream:  # load config (includes: which physics samples, model params)
        config = yaml.safe_load(stream)

    # override loaded config with values from command line args
    config = override_config(config, args)

    assert config["load"], "Must pass an MLPF model to --load"

    if "best_weights" in Path(config["load"]).name:
        loaddir = str(Path(config["load"]).parent)
    else:
        # the checkpoint is provided directly
        loaddir = str(Path(config["load"]).parent.parent)

    outdir = create_experiment_dir(
        prefix=(args.prefix or "") + "_",
        experiments_dir=loaddir,
    )

    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    config_filename = "train-config.yaml" if args.train else "test-config.yaml"
    with open((Path(outdir) / config_filename), "w") as file:
        yaml.dump(config, file)

    logfile = f"{outdir}/train.log"
    _configLogger("mlpf", filename=logfile)

    if config["gpus"]:
        assert torch.cuda.device_count() > 0, "--No gpu available"

        torch.cuda.empty_cache()

        rank = 0
        _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")

    _configLogger("mlpf", filename=logfile)

    _logger.info("Initializing an MLPF backbone model", color="orange")

    with open(f"{loaddir}/model_kwargs.pkl", "rb") as f:
        mlpf_kwargs = pkl.load(f)
    _logger.info("mlpf_kwargs: {}".format(mlpf_kwargs))

    mlpf_kwargs["attention_type"] = config["model"]["attention"]["attention_type"]

    mlpf = MLPF(**mlpf_kwargs).to(torch.device(rank))

    if not args.reinitialize_backbone:
        _logger.info("Loading the weights from a checkpoint", color="orange")
        checkpoint = torch.load(config["load"], map_location=torch.device(rank))
        mlpf = load_checkpoint(checkpoint, mlpf)

    mlpf.eval()
    _logger.info(mlpf)

    if args.use_latentX:  # the dimension will be the same as the input to one of the regression MLPs (e.g. pt)
        deepmet_input_dim = mlpf.nn_pt.nn[0].in_features
    else:
        deepmet_input_dim = 5 + 6  # p4 + PID

    # define the deepmet model
    deepmet = DeepMET(input_dim=deepmet_input_dim).to(torch.device(rank))
    optimizer = torch.optim.AdamW(deepmet.parameters(), lr=args.lr)
    _logger.info(deepmet)

    if args.train:
        save_HPs(args, deepmet, mlpf_kwargs, outdir)  # save model_kwargs and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

        loaders = get_interleaved_dataloaders(
            1,
            rank,
            config,
            use_cuda=rank != "cpu",
            use_ray=False,
        )

        train_mlpf(
            rank,
            deepmet,
            mlpf,
            args.freeze_backbone,
            args.use_latentX,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_epochs"],
            config["patience"],
            outdir,
            trainable=config["model"]["trainable"],
            checkpoint_freq=config["checkpoint_freq"],
        )


if __name__ == "__main__":

    # e.g.
    # noqa: python mlpf/met_finetuning_pipeline.py --dataset clic --data-dir tensorflow_datasets --config parameters/pytorch/pyg-clic-ttbar.yaml --gpus 1 --prefix MLPF_test1 --num-epochs 10 --train --load /pfvol/experiments/MLPF_clic_backbone_pyg-clic_20240429_101112_971749/best_weights.pth --gpu-batch-multiplier 100 --num-workers 2 --prefetch-factor 2 --checkpoint-freq 1 --lr 1e-4 --use-latentX
    main()
