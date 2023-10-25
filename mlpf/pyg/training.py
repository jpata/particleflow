import pickle as pkl
from tempfile import TemporaryDirectory
import time
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .logger import _logger
from .utils import unpack_predictions, unpack_target

# from torch.profiler import profile, record_function, ProfilerActivity


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
    loss["Classification"] = 100 * loss_obj_id(ypred["cls_id_onehot"], y["cls_id"])

    msk_true_particle = torch.unsqueeze((y["cls_id"] != 0).to(dtype=torch.float32), axis=-1)

    loss["Regression"] = 10 * torch.nn.functional.huber_loss(
        ypred["momentum"] * msk_true_particle, y["momentum"] * msk_true_particle
    )
    loss["Charge"] = torch.nn.functional.cross_entropy(
        ypred["charge"] * msk_true_particle, (y["charge"] * msk_true_particle[:, 0]).to(dtype=torch.int64)
    )

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
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none", ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
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

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

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


def train(
    rank,
    world_size,
    model,
    optimizer,
    train_loader,
    valid_loader,
    best_val_loss,
    stale_epochs,
    patience,
    outdir,
    tensorboard_writer=None,
):
    """
    Performs training over a given epoch. Will run a validation step every N_STEPS and after the last training batch.
    """
    global ISTEP_GLOBAL

    N_STEPS = 1000  # number of steps before running validation

    _logger.info(f"Initiating a training run on device {rank}", color="red")

    # initialize loss counters (note: these will be reset after N_STEPS)
    train_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}
    valid_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}

    istep = 0
    model.train()
    for itrain, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        istep += 1

        ygen = unpack_target(batch.to(rank).ygen)
        ypred = unpack_predictions(model(batch.to(rank)))

        # JP: need to debug this
        # assert np.all(target_charge.unique().cpu().numpy() == [0, 1, 2])
        loss = mlpf_loss(ygen, ypred)

        for param in model.parameters():
            param.grad = None
        loss["Total"].backward()
        optimizer.step()

        for loss_ in train_loss:
            train_loss[loss_] += loss[loss_].detach()
        for loss_ in epoch_loss:
            epoch_loss[loss_] += loss[loss_].detach()

        # run a quick validation run at intervals of N_STEPS or at the last step
        if (((itrain % N_STEPS) == 0) and (itrain != 0)) or (itrain == (len(train_loader) - 1)):
            if itrain == (len(train_loader) - 1):
                nsteps = istep
            else:
                nsteps = N_STEPS
                istep = 0

            if tensorboard_writer:
                for loss_ in train_loss:
                    tensorboard_writer.add_scalar(f"step_train/loss_{loss_}", train_loss[loss_] / nsteps, ISTEP_GLOBAL)
                tensorboard_writer.flush()

            _logger.info(
                f"Rank {rank}: "
                + f"train_loss_tot={train_loss['Total']/nsteps:.2f} "
                + f"train_loss_id={train_loss['Classification']/nsteps:.2f} "
                + f"train_loss_momentum={train_loss['Regression']/nsteps:.2f} "
                + f"train_loss_charge={train_loss['Charge']/nsteps:.2f} "
            )
            train_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}

            if world_size > 1:
                dist.barrier()  # wait until training run is finished on all ranks before running the validation

            if (rank == 0) or (rank == "cpu"):
                _logger.info(f"Initiating a quick validation run on device {rank}", color="red")
                model.eval()

                valid_loss = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}
                with torch.no_grad():
                    for ival, batch in tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader)):
                        ygen = unpack_target(batch.to(rank).ygen)
                        if world_size > 1:  # validation is only run on a single machine
                            ypred = unpack_predictions(model.module(batch.to(rank)))
                        else:
                            ypred = unpack_predictions(model(batch.to(rank)))

                        loss = mlpf_loss(ygen, ypred)

                        for loss_ in valid_loss:
                            valid_loss[loss_] += loss[loss_].detach()

                    for loss_ in valid_loss:
                        valid_loss[loss_] = valid_loss[loss_].cpu().item() / len(valid_loader)

                    if tensorboard_writer:
                        for loss_ in valid_loss:
                            tensorboard_writer.add_scalar(f"step_valid/loss_{loss_}", valid_loss[loss_], ISTEP_GLOBAL)

                    if valid_loss["Total"] < best_val_loss:
                        best_val_loss = valid_loss["Total"]

                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            model_state_dict = model.module.state_dict()
                        else:
                            model_state_dict = model.state_dict()

                        torch.save(
                            {"model_state_dict": model_state_dict, "optimizer_state_dict": optimizer.state_dict()},
                            f"{outdir}/best_weights.pth",
                        )
                        _logger.info(
                            f"finished {itrain+1}/{len(train_loader)} iterations and saved the model at {outdir}/best_weights.pth"  # noqa
                        )
                        stale_epochs = torch.tensor(0, device=rank)
                    else:
                        _logger.info(f"finished {itrain}/{len(train_loader)} iterations")
                        stale_epochs += 1

                    _logger.info(
                        f"Rank {rank}: "
                        + f"val_loss_tot={valid_loss['Total']:.2f} "
                        + f"val_loss_id={valid_loss['Classification']:.2f} "
                        + f"val_loss_momentum={valid_loss['Regression']:.2f} "
                        + f"val_loss_charge={valid_loss['Charge']:.2f} "
                        + f"best_val_loss={best_val_loss:.2f} "
                        + f"stale={stale_epochs} "
                    )
                    ISTEP_GLOBAL += 1

                model.train()  # prepare for next training loop

            if world_size > 1:
                dist.barrier()  # wait until validation run on rank 0 is finished before going to the next epoch
                dist.broadcast(stale_epochs, src=0)  # broadcast stale_epochs to all gpus

            if stale_epochs > patience:
                _logger.info("breaking due to stale epochs")
                return None, None, None, stale_epochs

        if tensorboard_writer:
            tensorboard_writer.flush()

    for loss_ in epoch_loss:
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / len(train_loader)

    return epoch_loss, valid_loss, best_val_loss, stale_epochs


def train_mlpf(rank, world_size, model, optimizer, train_loader, valid_loader, num_epochs, patience, outdir, hpo=False):
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
        tensorboard_writer = SummaryWriter(f"{outpath}/runs/")
    else:
        tensorboard_writer = False

    t0_initial = time.time()

    losses_of_interest = ["Total", "Classification", "Regression"]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), 99999.9
    start_epoch = 0

    if hpo:
        import ray.train as ray_train
        from ray.train import Checkpoint

        checkpoint = ray_train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                with checkpoint.as_directory() as checkpoint_dir:
                    checkpoint_dir = Path(checkpoint_dir)
                    # TODO: EW, check if map_location should be "cpu" below
                    model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))
                    optimizer.load_state_dict(torch.load(checkpoint_dir / "optim.pt"))
                    start_epoch = torch.load(checkpoint_dir / "extra_state.pt")["epoch"] + 1

    for epoch in range(start_epoch, num_epochs):
        _logger.info(f"Initiating epoch # {epoch}", color="bold")
        t0 = time.time()

        # training step
        losses_t, losses_v, best_val_loss, stale_epochs = train(
            rank,
            world_size,
            model,
            optimizer,
            train_loader,
            valid_loader,
            best_val_loss,
            stale_epochs,
            patience,
            outdir,
            tensorboard_writer,
        )

        if hpo:
            # save model, optimizer and epoch number for HPO-supported checkpointing
            if (rank == 0) or (rank == "cpu"):
                # Ray automatically syncs the cehckpoint to persistent storage
                with TemporaryDirectory() as temp_checkpoint_dir:
                    temp_checkpoint_dir = Path(temp_checkpoint_dir)
                    torch.save(model.state_dict(), temp_checkpoint_dir / "model.pt")
                    torch.save(optimizer.state_dict(), temp_checkpoint_dir / "optim.pt")
                    torch.save({"epoch": epoch}, temp_checkpoint_dir / "extra_state.pt")

                    # report metrics and checkpoint to Ray
                    ray_train.report(
                        dict(
                            loss=losses_t["Total"],
                            val_loss=losses_v["Total"],
                            epoch=epoch,
                        ),
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
                    )

        if stale_epochs > patience:
            break

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_train"):
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        for k, v in losses_t.items():
            tensorboard_writer.add_scalar(f"epoch/train_loss_rank_{rank}_" + k, v, epoch)

        if (rank == 0) or (rank == "cpu"):
            for loss in losses_of_interest:
                losses["train"][loss].append(losses_t[loss])
                losses["valid"][loss].append(losses_v[loss])
            for k, v in losses_v.items():
                tensorboard_writer.add_scalar("epoch/valid_loss_" + k, v, epoch)

            tensorboard_writer.flush()

            t1 = time.time()

            epochs_remaining = num_epochs - (epoch + 1)
            time_per_epoch = (t1 - t0_initial) / (epoch + 1)
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch + 1} / {num_epochs} "
                + f"train_loss={losses_t['Total']:.4f} "
                + f"valid_loss={losses_v['Total']:.4f} "
                + f"stale={stale_epochs} "
                + f"time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m"
            )

            for loss in losses_of_interest:
                fig, ax = plt.subplots()

                ax.plot(
                    range(len(losses["train"][loss])),
                    losses["train"][loss],
                    label="training",
                )
                ax.plot(
                    range(len(losses["valid"][loss])),
                    losses["valid"][loss],
                    label=f"validation ({best_val_loss:.3f})",
                )

                ax.set_xlabel("Epochs")
                ax.set_ylabel(f"{loss} Loss")
                ax.set_ylim(0.8 * losses["train"][loss][-1], 1.2 * losses["train"][loss][-1])
                ax.legend(title="MLPF", loc="best", title_fontsize=20, fontsize=15)
                plt.tight_layout()
                plt.savefig(f"{outdir}/mlpf_loss_{loss}.pdf")
                plt.close()

            with open(f"{outdir}/mlpf_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")
