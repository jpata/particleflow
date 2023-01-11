import json
import math
import os
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch

from .utils import combine_PFelements, distinguish_PFelements

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


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
def validation_run(
    device,
    encoder,
    mlpf,
    train_loader,
    valid_loader,
):
    with torch.no_grad():
        optimizer = None
        ret = train(
            device,
            encoder,
            mlpf,
            train_loader,
            valid_loader,
            optimizer,
        )
    return ret


def train(
    device,
    encoder,
    mlpf,
    train_loader,
    valid_loader,
    optimizer,
):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print(f"---->Initiating a training run")
        mlpf.train()
        loader = train_loader
    else:
        print(f"---->Initiating a validation run")
        mlpf.eval()
        loader = valid_loader

    # initialize loss counters
    losses = 0

    for i, batch in enumerate(loader):

        # make transformation
        tracks, clusters = distinguish_PFelements(batch.to(device))

        ### ENCODE
        embedding_tracks, embedding_clusters = encoder(tracks, clusters)

        tracks.x = embedding_tracks
        clusters.x = embedding_clusters

        event = combine_PFelements(tracks, clusters)

        # make mlpf forward pass
        pred_ids_one_hot = mlpf(event.to(device))
        target_ids = event.to(device).ygen_id

        weights = compute_weights(
            device, target_ids, num_classes=6
        )  # to accomodate class imbalance
        loss = torch.nn.functional.cross_entropy(
            pred_ids_one_hot, target_ids, weight=weights
        )  # for classifying PID

        # update parameters
        if is_train:
            for param in mlpf.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

        losses += loss.detach()

        # if i == 20:
        #     break

    losses = losses.cpu().item() / len(loader)

    return losses


def training_loop_mlpf(
    device,
    encoder,
    mlpf,
    train_loader,
    valid_loader,
    n_epochs,
    patience,
    optimizer,
    outpath,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        encoder: the encoder part of VICReg
        mlpf: the mlpf downstream task
        train_loader: a pytorch Dataloader for training
        valid_loader: a pytorch Dataloader for validation
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        outpath: path to store the model weights and training plots
    """

    t0_initial = time.time()

    losses_train, losses_valid = [], []

    best_val_loss = 99999.9
    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses = train(
            device,
            encoder,
            mlpf,
            train_loader,
            valid_loader,
            optimizer,
        )

        losses_train.append(losses)

        # validation step
        losses = validation_run(
            device,
            encoder,
            mlpf,
            train_loader,
            valid_loader,
        )

        losses_valid.append(losses)

        # early-stopping
        if losses < best_val_loss:
            best_val_loss = losses
            stale_epochs = 0

            try:
                mlpf_state_dict = mlpf.module.state_dict()
            except AttributeError:
                mlpf_state_dict = mlpf.state_dict()

            torch.save(
                mlpf_state_dict, f"{outpath}/mlpf_best_epoch_weights.pth"
            )

            with open(
                f"{outpath}/mlpf_best_epoch.json", "w"
            ) as fp:  # dump best epoch
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
        ax.plot(range(len(losses_train)), losses_train, label="training")
        ax.plot(range(len(losses_valid)), losses_valid, label="validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(
            title="SSL-based MLPF", loc="best", title_fontsize=20, fontsize=15
        )
        plt.savefig(f"{outpath}/mlpf_loss.pdf")

        with open(f"{outpath}/mlpf_loss_train.pkl", "wb") as f:
            pkl.dump(losses_train, f)
        with open(f"{outpath}/mlpf_loss_valid.pkl", "wb") as f:
            pkl.dump(losses_valid, f)

        print("----------------------------------------------------------")
    print(
        f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min"
    )
