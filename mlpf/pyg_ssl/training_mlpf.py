import json
import math
import pickle as pkl
import time
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .utils import combine_PFelements, distinguish_PFelements

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# keep track of the training step across epochs
istep_global = 0


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


def compute_weights(device, target_ids, num_classes):
    """
    computes necessary weights to accomodate class imbalance in the loss function
    """

    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(num_classes).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    # weights[2] = weights[2] * 3  # emphasize nhadrons
    return weights


@torch.no_grad()
def validation_run(device, encoder, mlpf, train_loader, valid_loader, mode, tb):
    with torch.no_grad():
        optimizer = None
        optimizer_VICReg = None
        ret = train(device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode, tb)
    return ret


def train(device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode, tb):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)
    global istep_global

    loss_obj_id = FocalLoss(gamma=2.0)

    if is_train:
        print("---->Initiating a training run")
        mlpf.train()
        loader = train_loader
        if optimizer_VICReg:
            encoder.train()
    else:
        print("---->Initiating a validation run")
        mlpf.eval()
        loader = valid_loader
        if optimizer_VICReg:
            encoder.eval()

    # initialize loss counters
    losses = 0

    epoch_loss_id = 0.0
    epoch_loss_momentum = 0.0
    epoch_loss_charge = 0.0

    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):

        if mode == "ssl":
            # seperate PF-elements
            tracks, clusters = distinguish_PFelements(batch.to(device))

            # ENCODE
            embedding_tracks, embedding_clusters = encoder(tracks, clusters)

            # concat the inputs with embeddings
            tracks.x = torch.cat([batch.x[batch.x[:, 0] == 1], embedding_tracks], axis=1)
            clusters.x = torch.cat([batch.x[batch.x[:, 0] == 2], embedding_clusters], axis=1)

            # combine PF-elements
            event = combine_PFelements(tracks, clusters)

        elif mode == "native":
            event = batch.to(device)

        # make mlpf forward pass
        event_on_device = event.to(device)
        pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event_on_device)
        target_ids = event_on_device.ygen_id

        target_momentum = event_on_device.ygen[:, 1:].to(dtype=torch.float32)
        target_charge = (event_on_device.ygen[:, 0] + 1).to(dtype=torch.float32)  # -1, 0, 1

        loss_id = 100 * loss_obj_id(pred_ids_one_hot, target_ids)
        # loss_id_old = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids)  # for classifying PID

        # for regression, mask the loss in cases there is no true particle
        msk_true_particle = torch.unsqueeze((target_ids != 0).to(dtype=torch.float32), axis=-1)
        loss_momentum = 10 * torch.nn.functional.huber_loss(
            pred_momentum * msk_true_particle, target_momentum * msk_true_particle
        )  # for regressing p4

        loss_charge = torch.nn.functional.cross_entropy(
            pred_charge * msk_true_particle, (target_charge * msk_true_particle[:, 0]).to(dtype=torch.int64)
        )  # for predicting charge
        loss = loss_id + loss_momentum + loss_charge

        if is_train:
            tb.add_scalar("batch/loss_id", loss_id.detach().cpu().item(), istep_global)
            tb.add_scalar("batch/loss_momentum", loss_momentum.detach().cpu().item(), istep_global)
            tb.add_scalar("batch/loss_charge", loss_charge.detach().cpu().item(), istep_global)
            istep_global += 1

        # update parameters
        if is_train:
            for param in mlpf.parameters():
                param.grad = None
            if optimizer_VICReg:
                for param in encoder.parameters():
                    param.grad = None
            loss.backward()
            optimizer.step()

        losses += loss.detach()

        epoch_loss_id += loss_id.detach().cpu().item() / len(loader)
        epoch_loss_momentum += loss_momentum.detach().cpu().item() / len(loader)
        epoch_loss_charge += loss_charge.detach().cpu().item() / len(loader)

    losses = losses.cpu().item() / len(loader)

    print(
        "loss_id={:.2f} loss_momentum={:.2f} loss_charge={:.2f}".format(
            epoch_loss_id, epoch_loss_momentum, epoch_loss_charge
        )
    )
    return losses


def training_loop_mlpf(
    device, encoder, mlpf, train_loader, valid_loader, n_epochs, patience, lr, outpath, mode, FineTune_VICReg
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        encoder: the encoder part of VICReg
        mlpf: the mlpf downstream task
        train_loader: a pytorch Dataloader for training
        valid_loader: a pytorch Dataloader for validation
        patience: number of stale epochs allowed before stopping the training
        lr: lr to use for training
        outpath: path to store the model weights and training plots
        mode: can be either `ssl` or `native`
        FineTune_VICReg: if `True` will finetune VICReg with a lr that is 10% of the MLPF lr
    """

    t0_initial = time.time()

    losses_train, losses_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    tensorboard_writer = SummaryWriter(outpath)

    optimizer = torch.optim.AdamW(mlpf.parameters(), lr=lr)
    if FineTune_VICReg:
        print("Will finetune VICReg during mlpf training")
        optimizer_VICReg = torch.optim.SGD(encoder.parameters(), lr=lr * 0.1)
    else:
        print("Will fix VICReg during mlpf training")
        optimizer_VICReg = None
    
    # set VICReg to evaluation mode
    encoder.eval()
    
    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses_t = train(
            device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode, tensorboard_writer
        )
        tensorboard_writer.add_scalar("epoch/train_loss", losses_t, epoch)
        losses_train.append(losses_t)

        # validation step
        losses_v = validation_run(device, encoder, mlpf, train_loader, valid_loader, mode, tensorboard_writer)
        tensorboard_writer.add_scalar("epoch/val_loss", losses_v, epoch)
        losses_valid.append(losses_v)

        tensorboard_writer.flush()

        # early-stopping
        if losses_v < best_val_loss:
            best_val_loss = losses_v
            best_train_loss = losses_t

            stale_epochs = 0

            try:
                mlpf_state_dict = mlpf.module.state_dict()
            except AttributeError:
                mlpf_state_dict = mlpf.state_dict()

            torch.save(mlpf_state_dict, f"{outpath}/mlpf_{mode}_best_epoch_weights.pth")

            with open(f"{outpath}/mlpf_{mode}_best_epoch.json", "w") as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_train[epoch], 4)} "
            + f"valid_loss={round(losses_valid[epoch], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        fig, ax = plt.subplots()

        ax.plot(range(len(losses_train)), losses_train, label="training ({:.2f})".format(best_train_loss))
        ax.plot(range(len(losses_valid)), losses_valid, label="validation ({:.2f})".format(best_val_loss))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_ylim(0.8 * losses_train[-1], 1.2 * losses_train[-1])
        if mode == "ssl":
            ax.legend(title="SSL-based MLPF", loc="best", title_fontsize=20, fontsize=15)
        else:
            ax.legend(title="Native MLPF", loc="best", title_fontsize=20, fontsize=15)
        plt.savefig(f"{outpath}/mlpf_loss.pdf")

        with open(f"{outpath}/mlpf_{mode}_loss_train.pkl", "wb") as f:
            pkl.dump(losses_train, f)
        with open(f"{outpath}/mlpf_{mode}_loss_valid.pkl", "wb") as f:
            pkl.dump(losses_valid, f)

        print("----------------------------------------------------------")
    print(f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
