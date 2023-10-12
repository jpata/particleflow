import json
import pickle as pkl
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# from torch.profiler import profile, record_function, ProfilerActivity

from .logger import _logger

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# keep track of step across epochs
ISTEP_GLOBAL_TRAIN = 0
ISTEP_GLOBAL_VALID = 0


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


@torch.no_grad()
def validation_run(rank, model, train_loader, valid_loader, tensorboard_writer=None):
    with torch.no_grad():
        optimizer = None
        ret = train(rank, model, train_loader, valid_loader, optimizer, tensorboard_writer)
    return ret


def train(rank, model, train_loader, valid_loader, optimizer, tensorboard_writer=None):
    """
    A training/validation run over a given epoch that gets called in the train_mlpf() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """
    global ISTEP_GLOBAL_TRAIN, ISTEP_GLOBAL_VALID
    is_train = not (optimizer is None)

    loss_obj_id = FocalLoss(gamma=2.0)

    if is_train:
        _logger.info(f"Initiating a training run on device {rank}", color="red")
        step_type = "train"
        loader = train_loader
        model.train()
    else:
        _logger.info(f"Initiating a validation run on device {rank}", color="red")
        step_type = "valid"
        loader = valid_loader
        model.eval()

    # initialize loss counters
    losses = {"Total": 0.0, "Classification": 0.0, "Regression": 0.0, "Charge": 0.0}

    num_iterations = 0
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
        num_iterations += 1

        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "step_{}/num_elems".format(step_type),
                batch.X.shape[0],
                ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
            )

        event = batch.to(rank)

        # recall target ~ ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"]
        target_ids = event.ygen[:, 0].long()
        target_charge = torch.clamp((event.ygen[:, 1] + 1).to(dtype=torch.float32), 0, 2)  # -1, 0, 1 -> 0, 1, 2
        target_momentum = event.ygen[:, 2:-1].to(dtype=torch.float32)

        # make mlpf forward pass
        # t0 = time.time()
        pred_ids_one_hot, pred_momentum, pred_charge = model(event)
        # print(f"{event}: {(time.time() - t0):.2f}s")

        for icls in range(pred_ids_one_hot.shape[1]):
            if tensorboard_writer:
                tensorboard_writer.add_scalar(
                    "step_{}/num_cls_{}".format(step_type, icls),
                    torch.sum(target_ids == icls),
                    ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
                )

        # JP: need to debug this
        # assert np.all(target_charge.unique().cpu().numpy() == [0, 1, 2])

        loss_ = {}
        # for CLASSIFYING PID
        loss_["Classification"] = 100 * loss_obj_id(pred_ids_one_hot, target_ids)
        # REGRESSING p4: mask the loss in cases there is no true particle
        msk_true_particle = torch.unsqueeze((target_ids != 0).to(dtype=torch.float32), axis=-1)
        loss_["Regression"] = 10 * torch.nn.functional.huber_loss(
            pred_momentum * msk_true_particle, target_momentum * msk_true_particle
        )
        # PREDICTING CHARGE
        loss_["Charge"] = torch.nn.functional.cross_entropy(
            pred_charge * msk_true_particle, (target_charge * msk_true_particle[:, 0]).to(dtype=torch.int64)
        )
        # TOTAL LOSS
        loss_["Total"] = loss_["Classification"] + loss_["Regression"] + loss_["Charge"]

        if tensorboard_writer:
            tensorboard_writer.add_scalar(
                "step_{}/loss".format(step_type),
                loss_["Total"],
                ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
            )
        if is_train:
            for param in model.parameters():
                param.grad = None
            loss_["Total"].backward()
            optimizer.step()

        for loss in losses:
            losses[loss] += loss_[loss].detach().cpu().item()

        if tensorboard_writer:
            tensorboard_writer.flush()

        if is_train:
            ISTEP_GLOBAL_TRAIN += 1
        else:
            ISTEP_GLOBAL_VALID += 1

    for loss in losses:
        losses[loss] = losses[loss] / num_iterations

    _logger.info(
        f"loss_id={losses['Classification']:.4f} loss_momentum={losses['Regression']:.4f} loss_charge={losses['Charge']:.4f}"
    )

    return losses


def train_mlpf(rank, world_size, model, train_loader, valid_loader, n_epochs, patience, lr, outpath):
    """
    Will run a full training by calling train() and validation_run() every epoch.

    Args:
        rank: 'cpu' or int representing the gpu device id
        model: a pytorch model that may be wrapped by DistributedDataParallel
        train_loader: a pytorch Dataloader that loads the training data in the form ~ DataBatch(X, ygen, ycands)
        valid_loader: a pytorch Dataloader that loads the validation data in the form ~ DataBatch(X, ygen, ycands)
        patience: number of stale epochs before stopping the training
        lr: learning rate to use for training
        outpath: path to store the model weights and training plots
    """

    tensorboard_writer = SummaryWriter(outpath)

    t0_initial = time.time()

    losses_of_interest = ["Total", "Classification", "Regression"]

    losses = {}
    losses["train"], losses["valid"], best_val_loss, best_train_loss = {}, {}, {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []
        best_val_loss[loss] = 99999.9

    stale_epochs = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            _logger.info("breaking due to stale epochs")
            break

        # training step
        losses_t = train(rank, model, train_loader, valid_loader, optimizer, tensorboard_writer)
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_train"):
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        for k, v in losses_t.items():
            tensorboard_writer.add_scalar(f"epoch/train_loss_rank_{rank}_" + k, v, epoch)
        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])

        # validation step on a single machine
        if (rank == 0) or (rank == "cpu"):
            if world_size > 1:
                dist.barrier()

            losses_v = validation_run(rank, model, train_loader, valid_loader, tensorboard_writer)
            for loss in losses_of_interest:
                losses["valid"][loss].append(losses_v[loss])
            for k, v in losses_v.items():
                tensorboard_writer.add_scalar("epoch/valid_loss_" + k, v, epoch)

            tensorboard_writer.flush()

            if world_size > 1:
                dist.barrier()

            # save the lowest value of each component of the loss to print it on the legend of the loss plots
            for loss in losses_of_interest:
                if loss == "Total":
                    if losses_v[loss] < best_val_loss[loss]:
                        best_val_loss[loss] = losses_v[loss]
                        best_train_loss[loss] = losses_t[loss]

                        # save the model
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()

                        torch.save(state_dict, f"{outpath}/best_epoch_weights.pth")

                        with open(f"{outpath}/best_epoch.json", "w") as fp:  # dump best epoch
                            json.dump({"best_epoch": epoch}, fp)

                        # for early-stopping purposes
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                else:
                    if losses_v[loss] < best_val_loss[loss]:
                        best_val_loss[loss] = losses_v[loss]
                        best_train_loss[loss] = losses_t[loss]

            t1 = time.time()

            epochs_remaining = n_epochs - (epoch + 1)
            time_per_epoch = (t1 - t0_initial) / (epoch + 1)
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch + 1} / {n_epochs} "
                + f"train_loss={round(losses_t['Total'], 4)} "
                + f"valid_loss={round(losses_v['Total'], 4)} "
                + f"stale={stale_epochs} "
                + f"time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m"
            )

            # make loss plots
            for loss in losses_of_interest:
                fig, ax = plt.subplots()
                ax.plot(
                    range(len(losses["train"][loss])),
                    losses["train"][loss],
                    label="training ({:.3f})".format(best_train_loss[loss]),
                )
                ax.plot(
                    range(len(losses["valid"][loss])),
                    losses["valid"][loss],
                    label="validation ({:.3f})".format(best_val_loss[loss]),
                )
                ax.set_xlabel("Epochs")
                ax.set_ylabel(f"{loss} Loss")
                ax.set_ylim(0.8 * losses["train"][loss][-1], 1.2 * losses["train"][loss][-1])
                ax.legend(title="MLPF", loc="best", title_fontsize=20, fontsize=15)
                plt.tight_layout()
                plt.savefig(f"{outpath}/mlpf_loss_{loss}.pdf")
                plt.close()

            with open(f"{outpath}/mlpf_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")
