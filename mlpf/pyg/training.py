import json
import math
import os
import time

import matplotlib
import numpy as np
import sklearn.metrics
import torch
import torch_geometric
from pyg import make_plot_from_lists
from pyg.cms_utils import CLASS_NAMES_CMS
from pyg.delphes_plots import plot_confusion_matrix

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def compute_weights(rank, target_ids, num_classes):
    """
    computes necessary weights to accomodate class imbalance in the loss function
    """

    vs, cs = torch.unique(target_ids, return_counts=True)
    weights = torch.zeros(num_classes).to(device=rank)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    # weights[2] = weights[2] * 3  # emphasize nhadrons
    return weights


@torch.no_grad()
def validation_run(rank, model, train_loader, valid_loader, batch_size, alpha, target_type, num_classes, outpath):
    with torch.no_grad():
        optimizer = None
        ret = train(rank, model, train_loader, valid_loader, batch_size, optimizer, alpha, target_type, num_classes, outpath)
    return ret


def train(rank, model, train_loader, valid_loader, batch_size, optimizer, alpha, target_type, num_classes, outpath):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print(f"---->Initiating a training run on rank {rank}")
        model.train()
        file_loader = train_loader
    else:
        print(f"---->Initiating a validation run rank {rank}")
        model.eval()
        file_loader = valid_loader

    # initialize loss counters
    losses_clf, losses_reg, losses_tot = 0, 0, 0

    # setup confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes))

    t0, tf = time.time(), 0
    for num, file in enumerate(file_loader):
        print(f"Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - t0, 3)}s")
        tf = tf + (time.time() - t0)

        file = [x for t in file for x in t]  # unpack the list of tuples to a list

        loader = torch_geometric.loader.DataLoader(file, batch_size=batch_size)

        t = 0
        for i, batch in enumerate(loader):

            # run forward pass
            t0 = time.time()
            pred_ids_one_hot, pred_p4 = model(batch.to(rank))
            t1 = time.time()
            # print(
            #     f"batch {i}/{len(loader)}, "
            #     + f"forward pass on rank {rank} = {round(t1 - t0, 3)}s, "
            #     + f"for batch with {batch.num_nodes} nodes"
            # )
            t = t + (t1 - t0)

            # define the target
            if target_type == "gen":
                target_p4 = batch.ygen
                target_ids = batch.ygen_id
            elif target_type == "cand":
                target_p4 = batch.ycand
                target_ids = batch.ycand_id

            # revert one hot encoding for the predictions
            pred_ids = torch.argmax(pred_ids_one_hot, axis=1)

            # define some useful masks
            # msk = (pred_ids != 0) & (target_ids != 0)
            msk2 = (pred_ids != 0) & (pred_ids == target_ids)

            # compute the loss
            weights = compute_weights(rank, target_ids, num_classes)  # to accomodate class imbalance
            loss_clf = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids, weight=weights)  # for classifying PID
            loss_reg = torch.nn.functional.mse_loss(
                pred_p4[msk2], target_p4[msk2]
            )  # for regressing p4 # TODO: add mse weights for scales to match? huber?

            loss_tot = loss_clf + (alpha * loss_reg)

            if is_train:
                for param in model.parameters():
                    # better than calling optimizer.zero_grad()
                    # according to https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
                    param.grad = None
                loss_tot.backward()
                optimizer.step()

            losses_clf = losses_clf + loss_clf.detach()
            losses_reg = losses_reg + loss_reg.detach()
            losses_tot = losses_tot + loss_tot.detach()

            conf_matrix += sklearn.metrics.confusion_matrix(
                target_ids.detach().cpu(), pred_ids.detach().cpu(), labels=range(num_classes)
            )

        #     if i == 2:
        #         break
        # if num == 2:
        #     break

        print(f"Average inference time per batch on rank {rank} is {round((t / len(loader)), 3)}s")

        t0 = time.time()

    print(f"Average time to load a file on rank {rank} is {round((tf / len(file_loader)), 3)}s")

    losses_clf = losses_clf / (len(loader) * len(file_loader))
    losses_reg = losses_reg / (len(loader) * len(file_loader))
    losses_tot = losses_tot / (len(loader) * len(file_loader))

    losses = {
        "losses_clf": losses_clf.cpu().item(),
        "losses_reg": losses_reg.cpu().item(),
        "losses_tot": losses_tot.cpu().item(),
    }

    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    return losses, conf_matrix


def training_loop(
    rank,
    data,
    model,
    train_loader,
    valid_loader,
    batch_size,
    n_epochs,
    patience,
    optimizer,
    alpha,
    target,
    num_classes,
    outpath,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        rank: int representing the gpu device id, or str=='cpu' (both work, trust me)
        data: data sepecification ('cms' or 'delphes')
        model: a pytorch model wrapped by DistributedDataParallel (DDP)
        dataset: a PFGraphDataset object
        train_loader: a pytorch Dataloader that loads .pt files for training when you invoke the get() method
        valid_loader: a pytorch Dataloader that loads .pt files for validation when you invoke the get() method
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        alpha: the hyperparameter controlling the classification vs regression task balance
        target: 'gen' or 'cand' training
        num_classes: number of particle candidate classes to predict (6 for delphes, 9 for cms)
        outpath: path to store the model weights and training plots
    """

    t0_initial = time.time()

    losses_clf_train, losses_reg_train, losses_tot_train = [], [], []
    losses_clf_valid, losses_reg_valid, losses_tot_valid = [], [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        model.train()
        losses, conf_matrix_train = train(
            rank, model, train_loader, valid_loader, batch_size, optimizer, alpha, target, num_classes, outpath
        )

        losses_clf_train.append(losses["losses_clf"])
        losses_reg_train.append(losses["losses_reg"])
        losses_tot_train.append(losses["losses_tot"])

        # validation step
        model.eval()
        losses, conf_matrix_val = validation_run(
            rank, model, train_loader, valid_loader, batch_size, alpha, target, num_classes, outpath
        )

        losses_clf_valid.append(losses["losses_clf"])
        losses_reg_valid.append(losses["losses_reg"])
        losses_tot_valid.append(losses["losses_tot"])

        # early-stopping
        if losses["losses_tot"] < best_val_loss:
            best_val_loss = losses["losses_tot"]
            stale_epochs = 0

            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict, f"{outpath}/best_epoch_weights.pth")

            with open(f"{outpath}/best_epoch.json", "w") as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"Rank {rank}: epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_tot_train[epoch], 4)} "
            + f"valid_loss={round(losses_tot_valid[epoch], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        # save the model's weights
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, f"{outpath}/epoch_{epoch}_weights.pth")

        # create directory to hold training plots
        if not os.path.exists(outpath + "/training_plots/"):
            os.makedirs(outpath + "/training_plots/")

        # make confusion matrix plots
        cm_path = outpath + "/training_plots/confusion_matrix_plots/"
        if not os.path.exists(cm_path):
            os.makedirs(cm_path)

        if data == "delphes":
            target_names = ["none", "ch.had", "n.had", "g", "el", "mu"]
        elif data == "cms":
            target_names = CLASS_NAMES_CMS

        plot_confusion_matrix(conf_matrix_train, target_names, epoch + 1, cm_path, f"epoch_{str(epoch)}_cmTrain")
        plot_confusion_matrix(conf_matrix_val, target_names, epoch + 1, cm_path, f"epoch_{str(epoch)}_cmValid")

        # make loss plots
        make_plot_from_lists(
            "Classification loss",
            "Epochs",
            "Loss",
            "loss_clf",
            [losses_clf_train, losses_clf_valid],
            ["training", "validation"],
            ["clf_losses_train", "clf_losses_valid"],
            outpath + "/training_plots/losses/",
        )
        make_plot_from_lists(
            "Regression loss",
            "Epochs",
            "Loss",
            "loss_reg",
            [losses_reg_train, losses_reg_valid],
            ["training", "validation"],
            ["reg_losses_train", "reg_losses_valid"],
            outpath + "/training_plots/losses/",
        )
        make_plot_from_lists(
            "Total loss",
            "Epochs",
            "Loss",
            "loss_tot",
            [losses_tot_train, losses_tot_valid],
            ["training", "validation"],
            ["tot_losses_train", "tot_losses_valid"],
            outpath + "/training_plots/losses/",
        )

        print("----------------------------------------------------------")
    print(f"Done with training. Total training time on rank {rank} is {round((time.time() - t0_initial)/60,3)}min")
