import json
import logging
import pickle as pkl
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

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


def train(rank, mlpf, train_loader, valid_loader, optimizer, tensorboard_writer=None):
    """
    A training/validation run over a given epoch that gets called in the train_mlpf() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """
    global ISTEP_GLOBAL_TRAIN, ISTEP_GLOBAL_VALID
    is_train = not (optimizer is None)

    loss_obj_id = FocalLoss(gamma=2.0)

    is_train = not (optimizer is None)
    step_type = "train" if is_train else "valid"

    if is_train:
        logging.info("Initiating a training run on {}".format(rank))
        loader = train_loader
        mlpf.train()
    else:
        logging.info("Initiating a validation run on {}".format(rank))
        loader = valid_loader
        mlpf.eval()

    # initialize loss counters
    losses_of_interest = ["Total", "Classification", "Regression", "Charge"]
    losses = {}
    for loss in losses_of_interest:
        losses[loss] = 0.0

    num_iterations = 0
    for i, batch in tqdm.tqdm(enumerate(loader)):
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
        target_charge = (event.ygen[:, 1] + 1).to(dtype=torch.float32)  # -1, 0, 1 -> 0, 1, 2
        target_momentum = event.ygen[:, 2:-1].to(dtype=torch.float32)

        # make mlpf forward pass
        # c = 0
        for i in range(1000):
            t0 = time.time()
            pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event.X, event.batch)
            print(f"{event}: {(time.time() - t0):.2f}s")
            # c += 1

        for icls in range(pred_ids_one_hot.shape[1]):
            if tensorboard_writer:
                tensorboard_writer.add_scalar(
                    "step_{}/num_cls_{}".format(step_type, icls),
                    torch.sum(target_ids == icls),
                    ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
                )

        assert np.all(target_charge.unique().cpu().numpy() == [0, 1, 2])

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

        # update parameters
        if is_train:
            for param in mlpf.parameters():
                param.grad = None
            loss_["Total"].backward()
            optimizer.step()

        for loss in losses_of_interest:
            losses[loss] += loss_[loss].detach()

        if tensorboard_writer:
            tensorboard_writer.flush()

        if is_train:
            ISTEP_GLOBAL_TRAIN += 1
        else:
            ISTEP_GLOBAL_VALID += 1

        if i == 2:
            break

    for loss in losses_of_interest:
        losses[loss] = losses[loss].cpu().item() / num_iterations

    logging.info(
        "loss_id={:.4f} loss_momentum={:.4f} loss_charge={:.4f}".format(
            losses["Classification"], losses["Regression"], losses["Charge"]
        )
    )

    return losses


def train_mlpf(rank, mlpf, train_loader, valid_loader, n_epochs, patience, lr, outpath):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me).
        mlpf: a pytorch model wrapped by DistributedDataParallel (DDP).
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method.
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method.
        patience: number of stale epochs allowed before stopping the training.
        lr: lr to use for training.
        outpath: path to store the model weights and training plots.
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

    optimizer = torch.optim.AdamW(mlpf.parameters(), lr=lr)

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            logging.info("breaking due to stale epochs")
            break

        # training step
        losses_t = train(rank, mlpf, train_loader, valid_loader, optimizer, tensorboard_writer)
        for k, v in losses_t.items():
            tensorboard_writer.add_scalar("epoch/train_loss_" + k, v, epoch)
        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])

        # validation step
        losses_v = validation_run(rank, mlpf, train_loader, valid_loader, tensorboard_writer)
        for loss in losses_of_interest:
            losses["valid"][loss].append(losses_v[loss])
        for k, v in losses_v.items():
            tensorboard_writer.add_scalar("epoch/valid_loss_" + k, v, epoch)

        tensorboard_writer.flush()

        # save the lowest value of each component of the loss to print it on the legend of the loss plots
        for loss in losses_of_interest:
            if loss == "Total":
                if losses_v[loss] < best_val_loss[loss]:
                    best_val_loss[loss] = losses_v[loss]
                    best_train_loss[loss] = losses_t[loss]

                    # save the model
                    try:
                        state_dict = mlpf.module.state_dict()
                    except AttributeError:
                        state_dict = mlpf.state_dict()
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

        logging.info(
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
                label="training ({:.3f})".format(best_train_loss["Total"]),
            )
            ax.plot(
                range(len(losses["valid"][loss])),
                losses["valid"][loss],
                label="validation ({:.3f})".format(best_val_loss["Total"]),
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

    logging.info(f"Done with training. Total training time on rank {rank} is {round((time.time() - t0_initial)/60,3)}min")


from collections import Counter, defaultdict

from logger import _logger


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)
    return preds


def train_hybrid(
    model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None
):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_reg_i = defaultdict(float)
    num_batches = 0
    total_correct = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            # for classification
            label_cls = y["_label_"].long()
            try:
                label_mask = y["_label_mask"].bool()
            except KeyError:
                label_mask = None
            label_cls = _flatten_label(label_cls, label_mask)
            label_counter.update(label_cls.cpu().numpy())
            label_cls = label_cls.to(dev)

            # for regression
            label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]]
            label_reg = torch.cat(label_reg, dim=1)
            n_reg = label_reg.shape[1]

            num_examples = label_reg.shape[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = _flatten_preds(model_output[:, :-n_reg], label_mask)
                preds_reg = model_output[:, -n_reg:]
                loss, loss_monitor = loss_func(logits, preds_reg, label_cls, label_reg)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, "_update_per_step", False):
                scheduler.step()

            _, preds_cls = logits.max(1)
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds_cls == label_cls).sum().item()

            total_loss += loss
            total_loss_cls += loss_monitor["cls"]
            total_loss_reg += loss_monitor["reg"]
            if n_reg > 1:
                for i in range(n_reg):
                    total_loss_reg_i[i] += loss_monitor[f"reg_{i}"]
            total_correct += correct

            e = preds_reg - label_reg
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix(
                {
                    "lr": "%.2e" % scheduler.get_last_lr()[0] if scheduler else opt.defaults["lr"],
                    "Loss": "%.5f" % loss_monitor["cls"],
                    "LossReg": "%.5f" % loss_monitor["reg"],
                    "LossTot": "%.5f" % loss,
                    # 'AvgLoss': '%.5f' % (total_loss / num_batches),
                    "Acc": "%.5f" % (correct / num_examples),
                    # 'AvgAcc': '%.5f' % (total_correct / count),
                    # 'MSE': '%.5f' % (sqr_err / num_examples),
                    # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    # 'MAE': '%.5f' % (abs_err / num_examples),
                    # 'AvgMAE': '%.5f' % (sum_abs_err / count),
                }
            )

            # stop writing to tensorboard after 500 batches
            if tb_helper and num_batches < 500:
                tb_helper.write_scalars(
                    [
                        (
                            "Loss/train",
                            loss_monitor["cls"],
                            tb_helper.batch_train_count + num_batches,
                        ),  # to compare cls loss to previous loss
                        ("LossReg/train", loss_monitor["reg"], tb_helper.batch_train_count + num_batches),
                        # ("LossTot/train", loss, tb_helper.batch_train_count + num_batches),
                        ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                        ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                        ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                        # ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ]
                )
                if n_reg > 1:
                    for i in range(n_reg):
                        tb_helper.write_scalars(
                            [
                                (f"LossReg{i}/train", loss_monitor[f"reg_{i}"], tb_helper.batch_train_count + num_batches),
                            ]
                        )
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(
                            model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode="train"
                        )

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info("Processed %d entries in total (avg. speed %.1f entries/s)" % (count, count / time_diff))
    _logger.info(
        "Train AvgLoss: %.5f, AvgLossReg: %.5f, AvgLossTot: %.5f, AvgAcc: %.5f, AvgMSE: %.5f, AvgMAE: %.5f"
        % (
            total_loss_cls / num_batches,
            total_loss_reg / num_batches,
            total_loss / num_batches,
            total_correct / count,
            sum_sqr_err / count,
            sum_abs_err / count,
        )
    )
    _logger.info("Train class distribution: \n    %s", str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars(
            [
                ("Loss/train (epoch)", total_loss_cls / num_batches, epoch),  # to compare cls loss to previous loss
                ("LossReg/train (epoch)", total_loss_reg / num_batches, epoch),
                ("LossTot/train (epoch)", total_loss / num_batches, epoch),
                ("Acc/train (epoch)", total_correct / count, epoch),
                ("MSE/train (epoch)", sum_sqr_err / count, epoch),
                ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ]
        )
        if n_reg > 1:
            for i in range(n_reg):
                tb_helper.write_scalars(
                    [
                        (f"LossReg{i}/train (epoch)", total_loss_reg_i[i] / num_batches, epoch),
                    ]
                )
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode="train")
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, "_update_per_step", False):
        scheduler.step()
