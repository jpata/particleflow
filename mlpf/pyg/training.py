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
import csv
import json
import sklearn
import sklearn.metrics
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import glob

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
    ELEM_TYPES_NONZERO,
    save_HPs,
    get_lr_schedule,
    count_parameters,
)


import fastjet
from pyg.inference import make_plots, run_predictions
from pyg.mlpf import MLPF
from pyg.PFDataset import Collater, PFDataset, get_interleaved_dataloaders
from utils import create_comet_experiment

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def sliced_wasserstein_loss(y_pred, y_true, num_projections=200):
    # create normalized random basis vectors
    theta = torch.randn(num_projections, y_true.shape[-1]).to(device=y_true.device)
    theta = theta / torch.sqrt(torch.sum(theta**2, axis=1, keepdims=True))

    # project the features with the random basis
    A = torch.matmul(y_true, torch.transpose(theta, -1, -2))
    B = torch.matmul(y_pred, torch.transpose(theta, -1, -2))

    A_sorted = torch.sort(A, axis=-2).values
    B_sorted = torch.sort(B, axis=-2).values

    ret = torch.sqrt(torch.sum(torch.pow(A_sorted - B_sorted, 2), axis=[-1, -2]))
    return ret


def mlpf_loss(y, ypred, batch):
    """
    Args
        y [dict]: relevant keys are "cls_id, momentum, charge"
        ypred [dict]: relevant keys are "cls_id_onehot, momentum, charge"
        batch [PFBatch]: the MLPF inputs
    """
    loss = {}
    loss_obj_id = FocalLoss(gamma=2.0, reduction="none")

    msk_pred_particle = torch.unsqueeze((ypred["cls_id"] != 0).to(dtype=torch.float32), axis=-1)
    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), axis=-1)
    nelem = torch.sum(batch.mask)
    npart = torch.sum(y["cls_id"] != 0)

    ypred["momentum"] = ypred["momentum"] * msk_true_particle
    y["momentum"] = y["momentum"] * msk_true_particle

    # in case of the 3D-padded mode, pytorch expects (batch, num_classes, ...)
    ypred["cls_binary"] = ypred["cls_binary"].permute((0, 2, 1))
    ypred["cls_id_onehot"] = ypred["cls_id_onehot"].permute((0, 2, 1))

    # binary loss for particle / no-particle classification
    # loss_binary_classification = loss_obj_id(ypred["cls_binary"], (y["cls_id"] != 0).long()).reshape(y["cls_id"].shape)
    loss_binary_classification = 10 * torch.nn.functional.cross_entropy(ypred["cls_binary"], (y["cls_id"] != 0).long(), reduction="none")

    # compare the particle type, only for cases where there was a true particle
    loss_pid_classification = loss_obj_id(ypred["cls_id_onehot"], y["cls_id"]).reshape(y["cls_id"].shape)
    loss_pid_classification[y["cls_id"] == 0] *= 0

    # compare particle momentum, only for cases where there was a true particle
    loss_regression = torch.nn.functional.mse_loss(ypred["momentum"], y["momentum"], reduction="none")
    loss_regression[y["cls_id"] == 0] *= 0

    # set the loss to 0 on padded elements in the batch
    loss_binary_classification[batch.mask == 0] *= 0
    loss_pid_classification[batch.mask == 0] *= 0
    loss_regression[batch.mask == 0] *= 0

    # average over all elements that were not padded
    loss["Classification_binary"] = loss_binary_classification.sum() / nelem
    loss["Classification"] = loss_pid_classification.sum() / nelem

    # normalize loss with stddev to stabilize across batches with very different pt, E distributions
    reg_losses = loss_regression[y["cls_id"] != 0]

    # average over all true particles
    loss["Regression"] = reg_losses.sum() / npart

    # in case we are using the 3D-padded mode, we can compute a few additional event-level monitoring losses
    # compute predicted pt from model output
    pred_pt = torch.exp(ypred["pt"].detach()) * batch.X[..., 1]

    # compute MET
    px = torch.unsqueeze(pred_pt * ypred["cos_phi"].detach(), axis=-1) * msk_pred_particle
    py = torch.unsqueeze(pred_pt * ypred["sin_phi"].detach(), axis=-1) * msk_pred_particle
    # sum across particle axis in event
    pred_met = torch.sqrt(torch.sum(px, axis=-2) ** 2 + torch.sum(py, axis=-2) ** 2)
    loss["MET"] = torch.nn.functional.huber_loss(pred_met.squeeze(dim=-1), batch.genmet).mean()

    # compute the classification loss between bin indices, for the cases where there was a true particle
    loss_bins = 10 * torch.nn.functional.cross_entropy(ypred["energy_bins"].transpose(1, 2), y["energy_bins"], reduction="none").reshape(
        y["cls_id"].shape
    )
    loss_bins[y["cls_id"] == 0] *= 0
    # print("pred bins", torch.argmax(ypred["energy_bins"], axis=-1))
    loss["Bins_energy"] = loss_bins.mean()

    was_input_pred = torch.concat([torch.softmax(ypred["cls_binary"].transpose(1, 2), axis=-1), ypred["momentum"]], axis=-1) * batch.mask.unsqueeze(
        axis=-1
    )
    was_input_true = torch.concat([torch.nn.functional.one_hot((y["cls_id"] != 0).to(torch.long)), y["momentum"]], axis=-1) * batch.mask.unsqueeze(
        axis=-1
    )

    # standardize Wasserstein loss
    std = was_input_true[batch.mask].std(axis=0)
    loss["Sliced_Wasserstein_Loss"] = sliced_wasserstein_loss(was_input_pred / std, was_input_true / std).mean()

    # this is the final loss to be optimized
    loss["Total"] = loss["Classification_binary"] + loss["Classification"] + loss["Bins_energy"] + loss["Regression"]

    # store these separately but detached
    loss["Classification_binary"] = loss["Classification_binary"].detach()
    loss["Classification"] = loss["Classification"].detach()
    loss["Regression"] = loss["Regression"].detach()
    loss["Sliced_Wasserstein_Loss"] = loss["Sliced_Wasserstein_Loss"].detach()

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

    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0.0, reduction: str = "mean", ignore_index: int = -100):
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


def configure_model_trainable(model, trainable, is_training):
    if is_training:
        model.train()
        if trainable != "all":
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            for layer in trainable:
                layer = getattr(model, layer)
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        model.eval()


def validation_plots(batch, ypred_raw, ygen, ypred, tensorboard_writer, epoch, outdir):
    X = batch.X[batch.mask].cpu()
    ygen_flat = batch.ygen[batch.mask].cpu()
    ypred_binary = ypred_raw[0][batch.mask].detach().cpu()
    ypred_binary_cls = torch.argmax(ypred_binary, axis=-1)
    ypred_cls = ypred_raw[1][batch.mask].detach().cpu()
    ypred_p4 = ypred_raw[2][batch.mask].detach().cpu()

    arr = torch.concatenate(
        [X, ygen_flat, ypred_binary, ypred_cls, ypred_p4],
        axis=-1,
    ).numpy()
    df = pandas.DataFrame(arr)
    df.to_parquet(f"{outdir}/batch0_epoch{epoch}.parquet")

    if tensorboard_writer:
        sig_prob = torch.softmax(ypred_binary, axis=-1)[:, 1].to(torch.float32)
        for xcls in np.unique(X[:, 0]):
            fig = plt.figure()
            msk = X[:, 0] == xcls
            egen = ygen_flat[msk & (ygen_flat[:, 0] != 0), 6]
            epred = ypred_p4[msk & (ypred_binary_cls != 0), 4]
            b = np.linspace(-8, 8, 100)
            plt.hist(egen, bins=b, histtype="step")
            plt.hist(epred, bins=b, histtype="step")
            plt.xlabel("log [E/E_elem]")
            tensorboard_writer.add_figure("energy_elemtype{}".format(int(xcls)), fig, global_step=epoch)

            fig = plt.figure()
            e_bin_pred = torch.argmax(ypred["energy_bins"][batch.mask].detach().cpu(), axis=-1)[
                msk & (ygen_flat[:, 0] != 0) & (ypred_binary_cls != 0)
            ]
            e_bin_true = ygen["energy_bins"][batch.mask].cpu()[msk & (ygen_flat[:, 0] != 0) & (ypred_binary_cls != 0)]
            if len(e_bin_pred) > 0 and len(e_bin_true) > 0:
                cm = sklearn.metrics.confusion_matrix(e_bin_true, e_bin_pred, labels=range(32))
                plt.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())
                tensorboard_writer.add_figure("energy_cm_elemtype{}".format(int(xcls)), fig, global_step=epoch)

            fig = plt.figure()
            b = np.linspace(0, 1, 100)
            plt.hist(sig_prob[msk & (ygen_flat[:, 0] == 0)], bins=b, histtype="step")
            plt.hist(sig_prob[msk & (ygen_flat[:, 0] != 0)], bins=b, histtype="step")
            plt.xlabel("particle proba")
            tensorboard_writer.add_figure("sig_proba_elemtype{}".format(int(xcls)), fig, global_step=epoch)

        tensorboard_writer.add_histogram("pt_target", torch.clamp(batch.ygen[batch.mask][:, 2], -10, 10), global_step=epoch)
        tensorboard_writer.add_histogram("pt_pred", torch.clamp(ypred_raw[2][batch.mask][:, 0], -10, 10), global_step=epoch)
        ratio = (ypred_raw[2][batch.mask][:, 0] / batch.ygen[batch.mask][:, 2])[batch.ygen[batch.mask][:, 0] != 0]
        tensorboard_writer.add_histogram("pt_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

        tensorboard_writer.add_histogram("eta_target", torch.clamp(batch.ygen[batch.mask][:, 3], -10, 10), global_step=epoch)
        tensorboard_writer.add_histogram("eta_pred", torch.clamp(ypred_raw[2][batch.mask][:, 1], -10, 10), global_step=epoch)
        ratio = (ypred_raw[2][batch.mask][:, 1] / batch.ygen[batch.mask][:, 3])[batch.ygen[batch.mask][:, 0] != 0]
        tensorboard_writer.add_histogram("eta_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

        tensorboard_writer.add_histogram("sphi_target", torch.clamp(batch.ygen[batch.mask][:, 4], -10, 10), global_step=epoch)
        tensorboard_writer.add_histogram("sphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 2], -10, 10), global_step=epoch)
        ratio = (ypred_raw[2][batch.mask][:, 2] / batch.ygen[batch.mask][:, 4])[batch.ygen[batch.mask][:, 0] != 0]
        tensorboard_writer.add_histogram("sphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

        tensorboard_writer.add_histogram("cphi_target", torch.clamp(batch.ygen[batch.mask][:, 5], -10, 10), global_step=epoch)
        tensorboard_writer.add_histogram("cphi_pred", torch.clamp(ypred_raw[2][batch.mask][:, 3], -10, 10), global_step=epoch)
        ratio = (ypred_raw[2][batch.mask][:, 3] / batch.ygen[batch.mask][:, 5])[batch.ygen[batch.mask][:, 0] != 0]
        tensorboard_writer.add_histogram("cphi_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

        tensorboard_writer.add_histogram("energy_target", torch.clamp(batch.ygen[batch.mask][:, 6], -10, 10), global_step=epoch)
        tensorboard_writer.add_histogram("energy_pred", torch.clamp(ypred_raw[2][batch.mask][:, 4], -10, 10), global_step=epoch)
        ratio = (ypred_raw[2][batch.mask][:, 4] / batch.ygen[batch.mask][:, 6])[batch.ygen[batch.mask][:, 0] != 0]
        tensorboard_writer.add_histogram("energy_ratio", torch.clamp(ratio, -10, 10), global_step=epoch)

        for attn in sorted(list(glob.glob(f"{outdir}/attn_conv_*.npz"))):
            attn_name = os.path.basename(attn).split(".")[0]
            attn_matrix = np.load(attn)["att"]
            batch_size = min(attn_matrix.shape[0], 8)
            fig, axes = plt.subplots(1,batch_size, figsize=((batch_size*3, 1*3)))
            for ibatch in range(batch_size):
                plt.sca(axes[ibatch])
                #plot the attention matrix of the first event in the batch
                plt.imshow(attn_matrix[ibatch].T, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=1e-5, vmax=1))
                plt.xticks([])
                plt.yticks([])
                plt.colorbar(fraction=0.04, pad=0.0)
                plt.title("event {}".format(ibatch))
            plt.suptitle(attn_name)
            tensorboard_writer.add_figure(attn_name, fig, global_step=epoch)


def train_and_valid(
    rank,
    world_size,
    outdir,
    model,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    lr_schedule=None,
    comet_experiment=None,
    comet_step_freq=None,
    epoch=None,
    val_freq=None,
    dtype=torch.float32,
    tensorboard_writer=None,
):
    """
    Performs training over a given epoch. Will run a validation step every N_STEPS and after the last training batch.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {}

    configure_model_trainable(model, trainable, is_train)
    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(data_loader)
    else:
        iterator = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}")

    device_type = "cuda" if isinstance(rank, int) else "cpu"

    loss_accum = 0.0
    val_freq_time_0 = time.time()

    if not is_train:
        cm_X_gen = np.zeros((13, 13))
        cm_X_pred = np.zeros((13, 13))
        cm_id = np.zeros((13, 13))

    for itrain, batch in iterator:
        model.set_save_attention(outdir, False)
        batch = batch.to(rank, non_blocking=True)

        ygen = unpack_target(batch.ygen, model)

        num_elems = batch.X[batch.mask].shape[0]
        num_batch = batch.X.shape[0]

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                ypred_raw = model(batch.X, batch.mask)
            else:
                with torch.no_grad():
                    if rank == 0 and itrain == 0:
                        model.set_save_attention(outdir, True)
                    ypred_raw = model(batch.X, batch.mask)

        ypred = unpack_predictions(ypred_raw)

        if not is_train:
            cm_X_gen += sklearn.metrics.confusion_matrix(
                batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ygen["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            cm_X_pred += sklearn.metrics.confusion_matrix(
                batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            cm_id += sklearn.metrics.confusion_matrix(
                ygen["cls_id"][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            # save the events of the first validation batch for quick checks
            if rank == 0 and itrain == 0:
                validation_plots(batch, ypred_raw, ygen, ypred, tensorboard_writer, epoch, outdir)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                loss = mlpf_loss(ygen, ypred, batch)
                for param in model.parameters():
                    param.grad = None
            else:
                with torch.no_grad():
                    loss = mlpf_loss(ygen, ypred, batch)

        if is_train:
            loss["Total"].backward()
            loss_accum += loss["Total"].detach().cpu().item()
            optimizer.step()
            if lr_schedule:
                lr_schedule.step()

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

        if is_train:
            step = (epoch - 1) * len(data_loader) + itrain
            if not (tensorboard_writer is None):
                if step % 100 == 0:
                    tensorboard_writer.add_scalar("step/loss", loss_accum / num_elems, step)
                    tensorboard_writer.add_scalar("step/num_elems", num_elems, step)
                    tensorboard_writer.add_scalar("step/num_batch", num_batch, step)
                    tensorboard_writer.add_scalar("step/learning_rate", lr_schedule.get_last_lr()[0], step)
                    tensorboard_writer.flush()
                    loss_accum = 0.0

                    extra_state = {"step": step, "lr_schedule_state_dict": lr_schedule.state_dict()}
                    save_checkpoint(f"{outdir}/step_weights.pth", model, optimizer, extra_state)

            if not (comet_experiment is None) and (itrain % comet_step_freq == 0):
                # this loss is not normalized to batch size
                comet_experiment.log_metrics(loss, prefix=f"{train_or_valid}", step=step)
                comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), step=step)

        if val_freq is not None and is_train:
            if itrain != 0 and itrain % val_freq == 0:
                # time since last intermediate validation run
                val_freq_time = torch.tensor(time.time() - val_freq_time_0, device=rank)
                if world_size > 1:
                    torch.distributed.all_reduce(val_freq_time)
                # compute intermediate training loss
                intermediate_losses_t = {key: epoch_loss[key] for key in epoch_loss}
                for loss_ in epoch_loss:
                    # sum up the losses from all workers and dicide by
                    if world_size > 1:
                        torch.distributed.all_reduce(intermediate_losses_t[loss_])
                    intermediate_losses_t[loss_] = intermediate_losses_t[loss_].cpu().item() / itrain

                # compute intermediate validation loss
                intermediate_losses_v = train_and_valid(
                    rank,
                    world_size,
                    outdir,
                    model,
                    optimizer,
                    train_loader,
                    valid_loader,
                    is_train=False,
                    epoch=epoch,
                    dtype=dtype,
                )
                intermediate_metrics = dict(
                    loss=intermediate_losses_t["Total"],
                    reg_loss=intermediate_losses_t["Regression"],
                    cls_loss=intermediate_losses_t["Classification"],
                    cls_binary_loss=intermediate_losses_t["Classification_binary"],
                    val_loss=intermediate_losses_v["Total"],
                    val_reg_loss=intermediate_losses_v["Regression"],
                    val_cls_loss=intermediate_losses_v["Classification"],
                    val_cls_binary_loss=intermediate_losses_v["Classification_binary"],
                    inside_epoch=epoch,
                    step=(epoch - 1) * len(data_loader) + itrain,
                    val_freq_time=val_freq_time.cpu().item(),
                )
                val_freq_log = os.path.join(outdir, "val_freq_log.csv")
                if (rank == 0) or (rank == "cpu"):
                    with open(val_freq_log, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=intermediate_metrics.keys())
                        if os.stat(val_freq_log).st_size == 0:  # only write header if file is empty
                            writer.writeheader()
                        writer.writerow(intermediate_metrics)
                if comet_experiment:
                    comet_experiment.log_metrics(intermediate_losses_v, prefix="valid", step=step)
                val_freq_time_0 = time.time()  # reset intermediate validation spacing timer

    if not is_train and comet_experiment:
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_gen, title="Element to target", row_label="X", column_label="target", epoch=epoch, file_name="cm_X_gen.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_pred, title="Element to pred", row_label="X", column_label="pred", epoch=epoch, file_name="cm_X_pred.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_id, title="Target to pred", row_label="gen", column_label="pred", epoch=epoch, file_name="cm_id.json"
        )

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
    trainable="all",
    dtype=torch.float32,
    start_epoch=1,
    lr_schedule=None,
    use_ray=False,
    checkpoint_freq=None,
    comet_experiment=None,
    comet_step_freq=None,
    val_freq=None,
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
        tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
        tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")
    else:
        tensorboard_writer_train = None
        tensorboard_writer_valid = None

    t0_initial = time.time()

    losses_of_interest = [
        "Total",
        "Classification",
        "Classification_binary",
        "Regression",
        "Bins_energy",
    ]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        # training step, edit here to profile a specific epoch
        if epoch == -1:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                with record_function("model_train"):
                    losses_t = train_and_valid(
                        rank,
                        world_size,
                        outdir,
                        model,
                        optimizer,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        trainable=trainable,
                        is_train=True,
                        lr_schedule=lr_schedule,
                        val_freq=val_freq,
                        dtype=dtype,
                    )
            prof.export_chrome_trace("trace.json")
        else:
            losses_t = train_and_valid(
                rank,
                world_size,
                outdir,
                model,
                optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                trainable=trainable,
                is_train=True,
                lr_schedule=lr_schedule,
                comet_experiment=comet_experiment,
                comet_step_freq=comet_step_freq,
                epoch=epoch,
                val_freq=val_freq,
                dtype=dtype,
                tensorboard_writer=tensorboard_writer_train,
            )
        t_train = time.time()  # epoch time excluding validation

        losses_v = train_and_valid(
            rank,
            world_size,
            outdir,
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            lr_schedule=None,
            comet_experiment=comet_experiment,
            comet_step_freq=comet_step_freq,
            epoch=epoch,
            dtype=dtype,
            tensorboard_writer=tensorboard_writer_valid,
        )
        t_valid = time.time()

        if comet_experiment:
            comet_experiment.log_metrics(losses_t, prefix="epoch_train_loss", epoch=epoch)
            comet_experiment.log_metrics(losses_v, prefix="epoch_valid_loss", epoch=epoch)
            comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), epoch=epoch)
            comet_experiment.log_epoch_end(epoch)

        if (rank == 0) or (rank == "cpu"):
            tensorboard_writer_train.add_scalar("epoch/learning_rate", lr_schedule.get_last_lr()[0], epoch)
            extra_state = {"epoch": epoch, "lr_schedule_state_dict": lr_schedule.state_dict()}
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
            import ray
            from ray.train import Checkpoint

            # save model, optimizer and epoch number for HPO-supported checkpointing
            # Ray automatically syncs the checkpoint to persistent storage
            metrics = dict(
                loss=losses_t["Total"],
                reg_loss=losses_t["Regression"],
                cls_loss=losses_t["Classification"],
                cls_binary_loss=losses_t["Classification_binary"],
                val_loss=losses_v["Total"],
                val_reg_loss=losses_v["Regression"],
                val_cls_loss=losses_v["Classification"],
                val_cls_binary_loss=losses_v["Classification_binary"],
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
                tensorboard_writer_train.add_scalar("epoch/loss_" + k, v, epoch)

            for loss in losses_of_interest:
                losses["train"][loss].append(losses_t[loss])
                losses["valid"][loss].append(losses_v[loss])

            for k, v in losses_v.items():
                tensorboard_writer_valid.add_scalar("epoch/loss_" + k, v, epoch)

            with open(f"{outdir}/mlpf_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

            t1 = time.time()

            epochs_remaining = num_epochs - epoch
            time_per_epoch = (t1 - t0_initial) / epoch
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch} / {num_epochs} "
                + f"train_loss={losses_t['Total']:.4f} "
                + f"valid_loss={losses_v['Total']:.4f} "
                + f"stale={stale_epochs} "
                + f"epoch_train_time={round((t_train-t0)/60, 2)}m "
                + f"epoch_valid_time={round((t_valid-t_train)/60, 2)}m "
                + f"epoch_total_time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m",
                color="bold",
            )

            # save separate json files with stats for each epoch, this is robust to crashed-then-resumed trainings
            history_path = Path(outdir) / "history"
            history_path.mkdir(parents=True, exist_ok=True)
            with open("{}/epoch_{}.json".format(str(history_path), epoch), "w") as fi:
                stats = {"train": losses_t, "valid": losses_v}
                stats["epoch_train_time"] = t_train - t0
                stats["epoch_valid_time"] = t_valid - t_train
                stats["epoch_total_time"] = t1 - t0
                json.dump(stats, fi)

            if tensorboard_writer_train:
                tensorboard_writer_train.flush()
            if tensorboard_writer_valid:
                tensorboard_writer_valid.flush()

    if world_size > 1:
        dist.barrier()

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")


def run(rank, world_size, config, args, outdir, logfile):
    if (rank == 0) or (rank == "cpu"):  # keep writing the logs
        _configLogger("mlpf", filename=logfile)

    use_cuda = rank != "cpu"

    dtype = getattr(torch, config["dtype"])
    _logger.info("configured dtype={} for autocast".format(dtype))

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    start_epoch = 1

    if config["load"]:  # load a pre-trained model
        with open(f"{outdir}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)
        _logger.info("model_kwargs: {}".format(model_kwargs))

        if config["conv_type"] == "attention":
            model_kwargs["attention_type"] = config["model"]["attention"]["attention_type"]

        model = MLPF(**model_kwargs).to(torch.device(rank))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        checkpoint = torch.load(config["load"], map_location=torch.device(rank))
        start_epoch = checkpoint["extra_state"]["epoch"] + 1

        for k in model.state_dict().keys():
            shp0 = model.state_dict()[k].shape
            shp1 = checkpoint["model_state_dict"][k].shape
            if shp0 != shp1:
                raise Exception("shape mismatch in {}, {}!={}".format(k, shp0, shp1))

        if (rank == 0) or (rank == "cpu"):
            _logger.info("Loaded model weights from {}".format(config["load"]), color="bold")

        model, optimizer = load_checkpoint(checkpoint, model, optimizer)
    else:  # instantiate a new model in the outdir created
        model_kwargs = {
            "input_dim": len(X_FEATURES[config["dataset"]]),
            "num_classes": len(CLASS_LABELS[config["dataset"]]),
            "input_encoding": config["model"]["input_encoding"],
            "pt_mode": config["model"]["pt_mode"],
            "eta_mode": config["model"]["eta_mode"],
            "sin_phi_mode": config["model"]["sin_phi_mode"],
            "cos_phi_mode": config["model"]["cos_phi_mode"],
            "energy_mode": config["model"]["energy_mode"],
            "elemtypes_nonzero": ELEM_TYPES_NONZERO[config["dataset"]],
            "learned_representation_mode": config["model"]["learned_representation_mode"],
            **config["model"][config["conv_type"]],
        }
        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    model.to(rank)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    configure_model_trainable(model, config["model"]["trainable"], True)
    trainable_params, nontrainable_params, table = count_parameters(model)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)
        _logger.info(f"Trainable parameters: {trainable_params}")
        _logger.info(f"Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table.to_string(index=False))

    if args.train:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info("Creating experiment dir {}".format(outdir))
            _logger.info(f"Model directory {outdir}", color="bold")

        if args.comet:
            comet_experiment = create_comet_experiment(config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir)
            comet_experiment.set_name(f"rank_{rank}_{Path(outdir).name}")
            comet_experiment.log_parameter("run_id", Path(outdir).name)
            comet_experiment.log_parameter("world_size", world_size)
            comet_experiment.log_parameter("rank", rank)
            comet_experiment.log_parameters(config, prefix="config:")
            comet_experiment.set_model_graph(model)
            comet_experiment.log_parameter(trainable_params, "trainable_params")
            comet_experiment.log_parameter(nontrainable_params, "nontrainable_params")
            comet_experiment.log_parameter(trainable_params + nontrainable_params, "total_trainable_params")
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
            use_ray=False,
        )
        steps_per_epoch = len(loaders["train"])
        last_epoch = -1 if start_epoch == 1 else start_epoch - 1
        lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch)

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
            trainable=config["model"]["trainable"],
            dtype=dtype,
            start_epoch=start_epoch,
            lr_schedule=lr_schedule,
            use_ray=False,
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
            val_freq=config["val_freq"],
        )

        checkpoint = torch.load(f"{outdir}/best_weights.pth", map_location=torch.device(rank))
        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

    if not (config["load"] is None):
        testdir_name = "_" + Path(config["load"]).stem
    else:
        testdir_name = "_best_weights"

    if args.test:
        for sample in args.test_datasets:
            batch_size = config["gpu_batch_multiplier"]
            version = config["test_dataset"][sample]["version"]

            ds = PFDataset(config["data_dir"], f"{sample}:{version}", "test", num_samples=config["ntest"]).ds

            if (rank == 0) or (rank == "cpu"):
                _logger.info(f"test_dataset: {sample}, {len(ds)}", color="blue")

            if world_size > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(ds)
            else:
                sampler = torch.utils.data.RandomSampler(ds)

            test_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                collate_fn=Collater(["X", "ygen", "ycand", "genjets"], ["genmet"]),
                sampler=sampler,
                num_workers=config["num_workers"],
                prefetch_factor=config["prefetch_factor"],
                # pin_memory=use_cuda,
                # pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
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

            device_type = "cuda" if isinstance(rank, int) else "cpu"
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
                run_predictions(
                    world_size,
                    rank,
                    model,
                    test_loader,
                    sample,
                    outdir,
                    jetdef,
                    jet_ptcut=15.0,
                    jet_match_dr=0.1,
                    dir_name=testdir_name,
                )

    if (rank == 0) or (rank == "cpu"):  # make plots only on a single machine
        if args.make_plots:
            for sample in args.test_datasets:
                _logger.info(f"Plotting distributions for {sample}")
                make_plots(outdir, sample, config["dataset"], testdir_name)

    if world_size > 1:
        dist.destroy_process_group()


def override_config(config, args):
    """override config with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            config[arg] = arg_value

    if not (args.attention_type is None):
        config["model"]["attention"]["attention_type"] = args.attention_type

    if not (args.num_convs is None):
        for model in ["gnn_lsh", "attention", "attention", "mamba"]:
            config["model"][model]["num_convs"] = args.num_convs

    if len(args.test_datasets) == 0:
        args.test_datasets = config["test_dataset"]

    return config


def device_agnostic_run(config, args, world_size, outdir):

    if args.train:
        logfile = f"{outdir}/train.log"
    else:
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
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    use_cuda = args.gpus > 0

    rank = ray.train.get_context().get_local_rank() if use_cuda else "cpu"
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    model_kwargs = {
        "input_dim": len(X_FEATURES[config["dataset"]]),
        "num_classes": len(CLASS_LABELS[config["dataset"]]),
        "input_encoding": config["model"]["input_encoding"],
        "pt_mode": config["model"]["pt_mode"],
        "eta_mode": config["model"]["eta_mode"],
        "sin_phi_mode": config["model"]["sin_phi_mode"],
        "cos_phi_mode": config["model"]["cos_phi_mode"],
        "energy_mode": config["model"]["energy_mode"],
        "elemtypes_nonzero": ELEM_TYPES_NONZERO[config["dataset"]],
        "learned_representation_mode": config["model"]["learned_representation_mode"],
        **config["model"][config["conv_type"]],
    }
    model = MLPF(**model_kwargs)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # optimizer should be created after distributing the model to devices with ray.train.torch.prepare_model(model)
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    trainable_params, nontrainable_params, table = count_parameters(model)
    print(table)

    if (rank == 0) or (rank == "cpu"):
        with open(os.path.join(outdir, "num_params.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trainable_params", "nontrainable_params", "total_params"])
            writer.writerow([trainable_params, nontrainable_params, trainable_params + nontrainable_params])
        _logger.info(model)
        _logger.info(f"Trainable parameters: {trainable_params}")
        _logger.info(f"Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table)

    if (rank == 0) or (rank == "cpu"):
        save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

    loaders = get_interleaved_dataloaders(world_size, rank, config, use_cuda, use_ray=True)

    if args.comet:
        comet_experiment = create_comet_experiment(config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir)
        comet_experiment.set_name(f"world_rank_{world_rank}_{Path(outdir).name}")
        comet_experiment.log_parameter("run_id", Path(outdir).name)
        comet_experiment.log_parameter("world_size", world_size)
        comet_experiment.log_parameter("rank", rank)
        comet_experiment.log_parameter("world_rank", world_rank)
        comet_experiment.log_parameters(config, prefix="config:")
        comet_experiment.set_model_graph(model)
        comet_experiment.log_parameter(trainable_params, "trainable_params")
        comet_experiment.log_parameter(nontrainable_params, "nontrainable_params")
        comet_experiment.log_parameter(trainable_params + nontrainable_params, "total_trainable_params")
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

    steps_per_epoch = len(loaders["train"])
    start_epoch = 1
    lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch=-1)

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint = torch.load(Path(checkpoint_dir) / "checkpoint.pth", map_location=torch.device(rank))
                if args.resume_training:
                    model, optimizer = load_checkpoint(checkpoint, model, optimizer)
                    start_epoch = checkpoint["extra_state"]["epoch"] + 1
                    lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch=start_epoch - 1)
                else:  # start a new training with model weights loaded from a pre-trained model
                    model = load_checkpoint(checkpoint, model)

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
        trainable=config["model"]["trainable"],
        start_epoch=start_epoch,
        lr_schedule=lr_schedule,
        use_ray=True,
        checkpoint_freq=config["checkpoint_freq"],
        comet_experiment=comet_experiment,
        comet_step_freq=config["comet_step_freq"],
        dtype=getattr(torch, config["dtype"]),
        val_freq=config["val_freq"],
    )


def run_ray_training(config, args, outdir):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    if not args.local:
        ray.init(address="auto")

    if args.resume_training:
        outdir = args.resume_training  # continue training in the same directory

    _configLogger("mlpf", filename=f"{outdir}/train.log")

    use_gpu = args.gpus > 0
    num_workers = args.gpus if use_gpu else 1
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": max(1, args.ray_cpus // num_workers - 1), "GPU": int(use_gpu)},  # -1 to avoid blocking
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
    # Resume from checkpoint if a checkpoitn is found in outdir
    if TorchTrainer.can_restore(outdir):
        _logger.info(f"Restoring Ray Trainer from {outdir}", color="bold")
        trainer = TorchTrainer.restore(outdir, train_loop_per_worker=trainable, scaling_config=scaling_config)
    else:
        resume_from_checkpoint = ray.train.Checkpoint(config["load"]) if config["load"] else None
        if resume_from_checkpoint:
            _logger.info("Loading checkpoint {}".format(config["load"]), color="bold")
        trainer = TorchTrainer(
            train_loop_per_worker=trainable,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    result = trainer.fit()

    _logger.info("Final loss: {}".format(result.metrics["loss"]), color="bold")
    _logger.info("Final cls_loss: {}".format(result.metrics["cls_loss"]), color="bold")
    _logger.info("Final reg_loss: {}".format(result.metrics["reg_loss"]), color="bold")
    # _logger.info("Final charge_loss: {}".format(result.metrics["charge_loss"]), color="bold")

    _logger.info("Final val_loss: {}".format(result.metrics["val_loss"]), color="bold")
    _logger.info("Final val_cls_loss: {}".format(result.metrics["val_cls_loss"]), color="bold")
    _logger.info("Final val_reg_loss: {}".format(result.metrics["val_reg_loss"]), color="bold")
    # _logger.info("Final val_charge_loss: {}".format(result.metrics["val_charge_loss"]), color="bold")


def set_searchspace_and_run_trial(search_space, config, args):
    import ray
    from raytune.pt_search_space import set_hps_from_search_space

    rank = ray.train.get_context().get_local_rank()

    config = set_hps_from_search_space(search_space, config)
    try:
        # outdir will be taken from the ray.train.context.TrainContext in each trial
        train_ray_trial(config, args, outdir=None)
    except torch.cuda.OutOfMemoryError:
        ray.train.report({"val_loss": np.NAN})
        torch.cuda.empty_cache()  # make sure GPU memory is cleared for next trial
        if rank == 0:
            logging.warning("OOM error encountered, skipping this hyperparameter configuration.")
            skiplog_file_path = Path(config["raytune"]["local_dir"]) / args.hpo / "skipped_configurations.txt"
            lines = ["{}: {}\n".format(item[0], item[1]) for item in search_space.items()]

            with open(skiplog_file_path, "a") as f:
                f.write("#" * 80 + "\n")
                for line in lines:
                    f.write(line)
                    logging.warning(line[:-1])
                f.write("#" * 80 + "\n\n")
            logging.warning("Done writing warnings to log.")


def run_hpo(config, args):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    from raytune.pt_search_space import raytune_num_samples, search_space
    from raytune.utils import get_raytune_schedule, get_raytune_search_alg

    if args.raytune_num_samples:
        raytune_num_samples = args.raytune_num_samples  # noqa: F811

    name = args.hpo  # name of Ray Tune experiment directory

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"  # don't crash if a metric is missing
    if isinstance(config["raytune"]["local_dir"], type(None)):
        raise TypeError("Please specify a local_dir in the raytune section of the config file.")

    expdir = Path(config["raytune"]["local_dir"]) / name
    expdir.mkdir(parents=True, exist_ok=True)
    dirname = Path(config["raytune"]["local_dir"]) / name
    shutil.copy(
        "mlpf/raytune/pt_search_space.py",
        str(dirname / "pt_search_space.py"),
    )  # Copy the search space definition file to the train dir for later reference
    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    with open((dirname / "config.yaml"), "w") as file:
        yaml.dump(config, file)

    if not args.local:
        _logger.info("Inititalizing ray...")
        ray.init(
            address=os.environ["ip_head"],
            _node_ip_address=os.environ["head_node_ip"],
        )
        _logger.info("Done.")

    sched = get_raytune_schedule(config["raytune"])
    search_alg = get_raytune_search_alg(config["raytune"])

    scaling_config = ray.train.ScalingConfig(
        num_workers=args.gpus,
        use_gpu=True,
        resources_per_worker={"CPU": args.ray_cpus // (args.gpus) - 1, "GPU": 1},  # -1 to avoid blocking
    )

    if tune.Tuner.can_restore(str(expdir)):
        args.resume_training = True

    trainable = tune.with_parameters(set_searchspace_and_run_trial, config=config, args=args)
    trainer = TorchTrainer(train_loop_per_worker=trainable, scaling_config=scaling_config)

    if tune.Tuner.can_restore(str(expdir)):
        # resume unfinished HPO run
        tuner = tune.Tuner.restore(str(expdir), trainable=trainer, resume_errored=True, restart_errored=False, resume_unfinished=True)
    else:
        # start new HPO run
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
    _logger.info("Starting tuner.fit()")
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

    logging.info("Total time of Tuner.fit(): {}".format(end - start))
    logging.info("Best hyperparameters found according to {} were: {}".format(config["raytune"]["default_metric"], best_config))
