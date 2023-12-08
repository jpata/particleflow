import os
import os.path as osp
import pickle as pkl
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import logging
import shutil
from datetime import datetime
import tqdm
import yaml

import matplotlib.pyplot as plt
import numpy as np

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.tensorboard import SummaryWriter

from pyg.logger import _logger, _configLogger
from pyg.utils import (
    unpack_predictions,
    unpack_target,
    get_model_state_dict,
    load_checkpoint,
    save_checkpoint,
    CLASS_LABELS,
    X_FEATURES,
    save_HPs,
)


import fastjet
from pyg.inference import make_plots, run_predictions
from pyg.mlpf import MLPF
from pyg.PFDataset import Collater, PFDataLoader, PFDataset, get_interleaved_dataloaders
from utils import create_comet_experiment

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

ISTEP_GLOBAL = 0


def mlpf_loss(y, ypred):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge"
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0)

    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), axis=-1)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    ypred["charge"] = ypred["charge"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle
    y["charge"] = y["charge"] * msk_true_particle[..., 0]

    # pytorch expects (N, C, ...)
    if ypred["cls_id_onehot"].ndim > 2:
        ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))
        ypred["charge"] = ypred["charge"].permute((0, 2, 1))

    loss["Classification"] = 100 * loss_obj_id(ypred["cls_id_onehot"], y["cls_id"])

    loss["Regression"] = 10 * torch.nn.functional.huber_loss(ypred["momentum"], y["momentum"])
    loss["Charge"] = torch.nn.functional.cross_entropy(ypred["charge"], y["charge"].to(dtype=torch.int64))

    loss["Total"] = loss["Classification"] + loss["Regression"] + loss["Charge"]
    return loss


# from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self, alpha: Optional[Tensor] = None, gamma: float = 0.0, reduction: str = "mean", ignore_index: int = -100
    ):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        # this is slow due to indexing
        # all_rows = torch.arange(len(x))
        # log_pt = log_p[all_rows, y]
        log_pt = torch.gather(log_p, 1, y.unsqueeze(axis=-1)).squeeze(axis=-1)

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def train_and_valid(
    rank, world_size, model, optimizer, data_loader, is_train, comet_experiment=None, comet_step_freq=None, epoch=None
):
    """
    Performs training over a given epoch. Will run a validation step every N_STEPS and after the last training batch.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}

    if is_train:
        model.train()
    else:
        model.eval()

    # only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(data_loader)
    else:
        iterator = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
        )

    for itrain, batch in iterator:
        batch = batch.to(rank, non_blocking=True)

        ygen = unpack_target(batch.ygen)

        if world_size > 1:
            conv_type = model.module.conv_type
        else:
            conv_type = model.conv_type

        batchidx_or_mask = batch.batch if conv_type == "gravnet" else batch.mask
        ypred = model(batch.X, batchidx_or_mask)
        ypred = unpack_predictions(ypred)

        if is_train:
            loss = mlpf_loss(ygen, ypred)
            for param in model.parameters():
                param.grad = None
        else:
            with torch.no_grad():
                loss = mlpf_loss(ygen, ypred)

        if is_train:
            loss["Total"].backward()
            optimizer.step()

        for loss_ in epoch_loss:
            epoch_loss[loss_] += loss[loss_].detach()

        if comet_experiment and is_train:
            if itrain % comet_step_freq == 0:
                # this loss is not normalized to batch size
                comet_experiment.log_metrics(loss, prefix=f"{train_or_valid}", step=(epoch - 1) * len(data_loader) + itrain)

    num_data = torch.tensor(len(data_loader), device=rank)
    # sum up the number of steps from all workers
    if world_size > 1:
        torch.distributed.all_reduce(num_data)

    for loss_ in epoch_loss:
        # sum up the losses from all workers
        if world_size > 1:
            torch.distributed.all_reduce(epoch_loss[loss_])
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / num_data.cpu().item()

    if world_size > 1:
        dist.barrier()

    return epoch_loss


def train_mlpf(
    rank,
    world_size,
    model,
    optimizer,
    train_loader,
    valid_loader,
    num_epochs,
    patience,
    outdir,
    use_ray=False,
    checkpoint_freq=None,
    comet_experiment=None,
    comet_step_freq=None,
):
    """
    Will run a full training by calling train().

    Args:
        rank: 'cpu' or int representing the gpu device id
        model: a pytorch model (may be wrapped by DistributedDataParallel)
        train_loader: a pytorch geometric Dataloader that loads the training data in the form ~ DataBatch(X, ygen, ycands)
        valid_loader: a pytorch geometric Dataloader that loads the validation data in the form ~ DataBatch(X, ygen, ycands)
        patience: number of stale epochs before stopping the training
        outdir: path to store the model weights and training plots
    """

    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer = SummaryWriter(f"{outdir}/runs/")
    else:
        tensorboard_writer = None

    t0_initial = time.time()

    losses_of_interest = ["Total", "Classification", "Regression"]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), float("inf")
    start_epoch = 1

    if use_ray:
        import ray
        from ray.train import Checkpoint

        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                with checkpoint.as_directory() as checkpoint_dir:
                    checkpoint = torch.load(Path(checkpoint_dir) / "checkpoint.pth", map_location=torch.device(rank))
                    model, optimizer = load_checkpoint(checkpoint, model, optimizer)
                    start_epoch = checkpoint["extra_state"]["epoch"] + 1

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        # training step
        if epoch == -1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True
            ) as prof:
                with record_function("model_train"):
                    losses_t = train_and_valid(rank, world_size, model, optimizer, train_loader, True)
            prof.export_chrome_trace("trace.json")
        else:
            losses_t = train_and_valid(
                rank, world_size, model, optimizer, train_loader, True, comet_experiment, comet_step_freq, epoch
            )

        losses_v = train_and_valid(
            rank, world_size, model, optimizer, valid_loader, False, comet_experiment, comet_step_freq, epoch
        )

        if comet_experiment:
            comet_experiment.log_metrics(losses_t, prefix="epoch_train_loss", epoch=epoch)
            comet_experiment.log_metrics(losses_v, prefix="epoch_valid_loss", epoch=epoch)
            comet_experiment.log_epoch_end(epoch)

        if (rank == 0) or (rank == "cpu"):
            extra_state = {"epoch": epoch}
            if losses_v["Total"] < best_val_loss:
                best_val_loss = losses_v["Total"]
                stale_epochs = 0
                torch.save(
                    {"model_state_dict": get_model_state_dict(model), "optimizer_state_dict": optimizer.state_dict()},
                    f"{outdir}/best_weights.pth",
                )
                save_checkpoint(f"{outdir}/best_weights.pth", model, optimizer, extra_state)
            else:
                stale_epochs += 1

            if checkpoint_freq and (epoch != 0) and (epoch % checkpoint_freq == 0):
                checkpoint_dir = Path(outdir) / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(checkpoint_dir, epoch, losses_v["Total"])
                save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        if use_ray:
            # save model, optimizer and epoch number for HPO-supported checkpointing
            # Ray automatically syncs the checkpoint to persistent storage
            metrics = dict(
                loss=losses_t["Total"],
                reg_loss=losses_t["Regression"],
                cls_loss=losses_t["Classification"],
                charge_loss=losses_t["Charge"],
                val_loss=losses_v["Total"],
                val_reg_loss=losses_v["Regression"],
                val_cls_loss=losses_v["Classification"],
                val_charge_loss=losses_v["Charge"],
                epoch=epoch,
            )
            if (rank == 0) or (rank == "cpu"):
                # only save checkpoint on first worker
                with TemporaryDirectory() as temp_checkpoint_dir:
                    temp_checkpoint_dir = Path(temp_checkpoint_dir)
                    save_checkpoint(temp_checkpoint_dir / "checkpoint.pth", model, optimizer, extra_state)

                    # report metrics and checkpoint to Ray
                    ray.train.report(
                        metrics,
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir) if rank == 0 else None,
                    )
            else:
                # ray requires all workers to report metrics
                ray.train.report(metrics)

        if stale_epochs > patience:
            break

        if (rank == 0) or (rank == "cpu"):
            for k, v in losses_t.items():
                tensorboard_writer.add_scalar("epoch/train_loss_" + k, v, epoch)

            for loss in losses_of_interest:
                losses["train"][loss].append(losses_t[loss])
                losses["valid"][loss].append(losses_v[loss])

            for k, v in losses_v.items():
                tensorboard_writer.add_scalar("epoch/valid_loss_" + k, v, epoch)

            t1 = time.time()

            epochs_remaining = num_epochs - epoch
            time_per_epoch = (t1 - t0_initial) / epoch
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch} / {num_epochs} "
                + f"train_loss={losses_t['Total']:.4f} "
                + f"valid_loss={losses_v['Total']:.4f} "
                + f"stale={stale_epochs} "
                + f"time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m",
                color="bold",
            )

            # for loss in losses_of_interest:
            #     fig, ax = plt.subplots()

            #     ax.plot(
            #         range(len(losses["train"][loss])),
            #         losses["train"][loss],
            #         label="training",
            #     )
            #     ax.plot(
            #         range(len(losses["valid"][loss])),
            #         losses["valid"][loss],
            #         label=f"validation ({best_val_loss:.3f})",
            #     )

            #     ax.set_xlabel("Epochs")
            #     ax.set_ylabel(f"{loss} Loss")
            #     ax.set_ylim(0.8 * losses["train"][loss][-1], 1.2 * losses["train"][loss][-1])
            #     ax.legend(title="MLPF", loc="best", title_fontsize=20, fontsize=15)
            #     plt.tight_layout()
            #     plt.savefig(f"{outdir}/mlpf_loss_{loss}.pdf")
            #     plt.close()

            with open(f"{outdir}/mlpf_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

            if tensorboard_writer:
                tensorboard_writer.flush()

    if world_size > 1:
        dist.barrier()

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")


def run(rank, world_size, config, args, outdir, logfile):
    """Demo function that will be passed to each gpu if (world_size > 1) else will run normally on the given device."""

    pad_3d = config["conv_type"] != "gravnet"
    use_cuda = rank != "cpu"

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    if (rank == 0) or (rank == "cpu"):  # keep writing the logs
        _configLogger("mlpf", filename=logfile)

    if config["load"]:  # load a pre-trained model
        loaddir = config["load"]  # in case both --load and --train are provided

        with open(f"{loaddir}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)

        model = MLPF(**model_kwargs).to(torch.device(rank))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        if args.load_checkpoint:
            checkpoint = torch.load(f"{args.load_checkpoint}", map_location=torch.device(rank))
            if (rank == 0) or (rank == "cpu"):
                _logger.info(f"Loaded model weights from {loaddir}/checkpoints/{args.load_checkpoint}")
        else:
            checkpoint = torch.load(f"{loaddir}/best_weights.pth", map_location=torch.device(rank))
            if (rank == 0) or (rank == "cpu"):
                _logger.info(f"Loaded model weights from {loaddir}/best_weights.pth")

        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

        if args.load_checkpoint:
            testdir_name = f"_{args.load_checkpoint[:13]}"
        else:
            testdir_name = "_bestweights"

    else:  # instantiate a new model in the outdir created
        testdir_name = "_bestweights"

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

        if args.comet:
            comet_experiment = create_comet_experiment(
                config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir
            )
            comet_experiment.set_name(f"rank_{rank}")
            comet_experiment.log_parameter("run_id", Path(outdir).name)
            comet_experiment.log_parameter("world_size", world_size)
            comet_experiment.log_parameter("rank", rank)
            comet_experiment.log_parameters(config, prefix="config:")
            comet_experiment.set_model_graph(model)
            comet_experiment.log_code("mlpf/pyg/training.py")
            comet_experiment.log_code("mlpf/pyg_pipeline.py")
            # save overridden config then log to comet
            config_filename = "overridden_config.yaml"
            with open((Path(outdir) / config_filename), "w") as file:
                yaml.dump(config, file)
            comet_experiment.log_code(str(Path(outdir) / config_filename))
        else:
            comet_experiment = None

        loaders = get_interleaved_dataloaders(
            world_size,
            rank,
            config,
            use_cuda,
            pad_3d,
            use_ray=False,
        )

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
            use_ray=False,
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
        )

        checkpoint = torch.load(f"{outdir}/best_weights.pth", map_location=torch.device(rank))
        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

    if args.test:
        if config["load"] is None:
            # if we don't load, we must have a newly trained model
            assert args.train, "Please train a model before testing, or load a model with --load"
            assert outdir is not None, "Error: no outdir to evaluate model from"
        else:
            outdir = config["load"]

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

                test_loader = PFDataLoader(
                    ds,
                    batch_size=batch_size,
                    collate_fn=Collater(["X", "ygen", "ycand"], pad_3d=False),  # in inference, use sparse dataset
                    sampler=sampler,
                    num_workers=config["num_workers"],
                    prefetch_factor=config["prefetch_factor"],
                    pin_memory=use_cuda,
                    pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
                )

                if not osp.isdir(f"{outdir}/preds{testdir_name}/{sample}"):
                    if (rank == 0) or (rank == "cpu"):
                        os.system(f"mkdir -p {outdir}/preds{testdir_name}/{sample}")

                _logger.info(f"Running predictions on {sample}")
                torch.cuda.empty_cache()

                if args.dataset == "clic":
                    jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
                else:
                    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

                run_predictions(
                    world_size,
                    rank,
                    model,
                    test_loader,
                    sample,
                    outdir,
                    jetdef,
                    jet_ptcut=5.0,
                    jet_match_dr=0.1,
                    dir_name=testdir_name,
                )

    if (rank == 0) or (rank == "cpu"):  # make plots and export to onnx only on a single machine
        if args.make_plots:
            for type_ in config["test_dataset"][config["dataset"]]:  # will be "physical", "gun"
                for sample in config["test_dataset"][config["dataset"]][type_]["samples"]:
                    _logger.info(f"Plotting distributions for {sample}")

                    make_plots(outdir, sample, config["dataset"], testdir_name)

        if args.export_onnx:
            try:
                dummy_features = torch.randn(1, 640, model_kwargs["input_dim"], device=rank)
                dummy_mask = torch.zeros(1, 640, dtype=torch.bool, device=rank)
                torch.onnx.export(
                    model,
                    (dummy_features, dummy_mask),
                    "test.onnx",
                    verbose=True,
                    input_names=["features", "mask"],
                    output_names=["id", "momentum", "charge"],
                    dynamic_axes={
                        "features": {0: "num_batch", 1: "num_elements"},
                        "mask": [0, 1],
                        "id": [0, 1],
                        "momentum": [0, 1],
                        "charge": [0, 1],
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
    if args.train:
        logfile = f"{outdir}/train.log"
        _configLogger("mlpf", filename=logfile)
    else:
        outdir = args.load
        logfile = f"{outdir}/test.log"
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


def train_ray_trial(config, args, outdir=None):
    import ray

    if outdir is None:
        outdir = ray.train.get_context().get_trial_dir()

    pad_3d = config["conv_type"] != "gravnet"
    use_cuda = True

    rank = ray.train.get_context().get_local_rank()
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # keep writing the logs
    _configLogger("mlpf", filename=f"{outdir}/train.log")

    model_kwargs = {
        "input_dim": len(X_FEATURES[config["dataset"]]),
        "num_classes": len(CLASS_LABELS[config["dataset"]]),
        **config["model"][config["conv_type"]],
    }
    model = MLPF(**model_kwargs)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # optimizer should be created after distributing the model to devices with ray.train.torch.prepare_model(model)
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)

    if (rank == 0) or (rank == "cpu"):
        save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

    loaders = get_interleaved_dataloaders(world_size, rank, config, use_cuda, pad_3d, use_ray=True)

    if args.comet:
        comet_experiment = create_comet_experiment(
            config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir
        )
        comet_experiment.set_name(f"world_rank_{world_rank}")
        comet_experiment.log_parameter("run_id", Path(outdir).name)
        comet_experiment.log_parameter("world_size", world_size)
        comet_experiment.log_parameter("rank", rank)
        comet_experiment.log_parameter("world_rank", world_rank)
        comet_experiment.log_parameters(config, prefix="config:")
        comet_experiment.set_model_graph(model)
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/pyg/training.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/pyg_pipeline.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/raytune/pt_search_space.py"))
        # save overridden config then log to comet
        config_filename = "overridden_config.yaml"
        with open((Path(outdir) / config_filename), "w") as file:
            yaml.dump(config, file)
        comet_experiment.log_code(str(Path(outdir) / config_filename))
    else:
        comet_experiment = None

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
        use_ray=True,
        checkpoint_freq=config["checkpoint_freq"],
        comet_experiment=comet_experiment,
        comet_step_freq=config["comet_step_freq"],
    )


def run_ray_training(config, args, outdir):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    # create ray cache for intermediate storage of trials
    tmp_ray_cache = TemporaryDirectory()
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = tmp_ray_cache.name
    _logger.info(f"RAY_AIR_LOCAL_CACHE_DIR: {os.environ['RAY_AIR_LOCAL_CACHE_DIR']}")

    if not args.local:
        ray.init(address="auto")

    _configLogger("mlpf", filename=f"{outdir}/train.log")

    num_workers = args.gpus
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"CPU": max(1, args.ray_cpus // num_workers - 1), "GPU": 1},  # -1 to avoid blocking
    )
    storage_path = Path(args.experiments_dir if args.experiments_dir else "experiments").resolve()
    run_config = ray.train.RunConfig(
        name=Path(outdir).name,
        storage_path=storage_path,
        log_to_file=False,
        failure_config=ray.train.FailureConfig(max_failures=2),
        checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),  # keep only latest checkpoint
        sync_config=ray.train.SyncConfig(sync_artifacts=True),
    )
    trainable = tune.with_parameters(train_ray_trial, args=args, outdir=outdir)
    trainer = TorchTrainer(
        train_loop_per_worker=trainable, train_loop_config=config, scaling_config=scaling_config, run_config=run_config
    )
    result = trainer.fit()

    _logger.info("Final loss: {}".format(result.metrics["loss"]), color="bold")
    _logger.info("Final cls_loss: {}".format(result.metrics["cls_loss"]), color="bold")
    _logger.info("Final reg_loss: {}".format(result.metrics["reg_loss"]), color="bold")
    _logger.info("Final charge_loss: {}".format(result.metrics["charge_loss"]), color="bold")

    _logger.info("Final val_loss: {}".format(result.metrics["val_loss"]), color="bold")
    _logger.info("Final val_cls_loss: {}".format(result.metrics["val_cls_loss"]), color="bold")
    _logger.info("Final val_reg_loss: {}".format(result.metrics["val_reg_loss"]), color="bold")
    _logger.info("Final val_charge_loss: {}".format(result.metrics["val_charge_loss"]), color="bold")

    # clean up ray cache
    tmp_ray_cache.cleanup()


def set_searchspace_and_run_trial(search_space, config, args):
    from raytune.pt_search_space import set_hps_from_search_space

    config = set_hps_from_search_space(search_space, config)
    train_ray_trial(config, args, outdir=None)  # outdir will be taken from the TrainContext in each trial


def run_hpo(config, args):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    from raytune.pt_search_space import raytune_num_samples, search_space
    from raytune.utils import get_raytune_schedule, get_raytune_search_alg

    # create ray cache for intermediate storage of trials
    tmp_ray_cache = TemporaryDirectory()
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = tmp_ray_cache.name
    _logger.info(f"RAY_AIR_LOCAL_CACHE_DIR: {os.environ['RAY_AIR_LOCAL_CACHE_DIR']}")

    name = args.hpo  # name of Ray Tune experiment directory

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"  # don't crash if a metric is missing
    if isinstance(config["raytune"]["local_dir"], type(None)):
        raise TypeError("Please specify a local_dir in the raytune section of the config file.")
    trd = config["raytune"]["local_dir"] + "/tune_result_dir"
    os.environ["TUNE_RESULT_DIR"] = trd

    expdir = Path(config["raytune"]["local_dir"]) / name
    expdir.mkdir(parents=True, exist_ok=True)
    dirname = Path(config["raytune"]["local_dir"]) / name
    shutil.copy(
        "mlpf/raytune/search_space.py",
        str(dirname / "search_space.py"),
    )  # Copy the search space definition file to the train dir for later reference
    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    with open((dirname / "config.yaml"), "w") as file:
        yaml.dump(config, file)

    if not args.local:
        ray.init(address="auto")

    sched = get_raytune_schedule(config["raytune"])
    search_alg = get_raytune_search_alg(config["raytune"])

    scaling_config = ray.train.ScalingConfig(
        num_workers=args.gpus,
        use_gpu=True,
        resources_per_worker={"CPU": args.ray_cpus // (args.gpus) - 1, "GPU": 1},  # -1 to avoid blocking
    )
    trainable = tune.with_parameters(set_searchspace_and_run_trial, config=config, args=args)
    trainer = TorchTrainer(train_loop_per_worker=trainable, scaling_config=scaling_config)

    search_space = {"train_loop_config": search_space}  # the ray TorchTrainer only takes a single arg: train_loop_config
    tuner = tune.Tuner(
        trainer,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=raytune_num_samples,
            metric=config["raytune"]["default_metric"] if (search_alg is None and sched is None) else None,
            mode=config["raytune"]["default_mode"] if (search_alg is None and sched is None) else None,
            search_alg=search_alg,
            scheduler=sched,
        ),
        run_config=ray.train.RunConfig(
            name=name,
            storage_path=config["raytune"]["local_dir"],
            log_to_file=False,
            failure_config=ray.train.FailureConfig(max_failures=2),
            checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),  # keep only latest checkpoint
            sync_config=ray.train.SyncConfig(sync_artifacts=True),
        ),
    )
    start = datetime.now()
    result_grid = tuner.fit()
    end = datetime.now()

    print("Number of errored trials: {}".format(result_grid.num_errors))
    print("Number of terminated (not errored) trials: {}".format(result_grid.num_terminated))
    print("Ray Tune experiment path: {}".format(result_grid.experiment_path))

    best_result = result_grid.get_best_result(scope="last-10-avg", metric="val_loss", mode="min")
    best_config = best_result.config
    print("Best trial path: {}".format(best_result.path))

    result_df = result_grid.get_dataframe()
    print(result_df)
    print(result_df.columns)

    print("Number of errored trials: {}".format(result_grid.num_errors))
    print("Number of terminated (not errored) trials: {}".format(result_grid.num_terminated))
    print("Ray Tune experiment path: {}".format(result_grid.experiment_path))

    logging.info("Total time of Tuner.fit(): {}".format(end - start))
    logging.info(
        "Best hyperparameters found according to {} were: {}".format(config["raytune"]["default_metric"], best_config)
    )

    # clean up ray cache
    tmp_ray_cache.cleanup()
