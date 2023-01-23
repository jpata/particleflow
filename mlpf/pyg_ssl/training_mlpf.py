import json
import math
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
def validation_run(device, encoder, mlpf, train_loader, valid_loader, mode):
    with torch.no_grad():
        optimizer = None
        optimizer_VICReg = None
        ret = train(device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode)
    return ret


def train(device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

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
        encoder.eval()

    # initialize loss counters
    losses = 0

    for i, batch in enumerate(loader):

        if mode == "ssl":
            # make transformation
            tracks, clusters = distinguish_PFelements(batch.to(device))

            # ENCODE
            embedding_tracks, embedding_clusters = encoder(tracks, clusters)

            tracks.x = embedding_tracks
            clusters.x = embedding_clusters

            event = combine_PFelements(tracks, clusters)

        elif mode == "native":
            event = batch.to(device)

        # make mlpf forward pass
        pred_ids_one_hot = mlpf(event.to(device))
        target_ids = event.to(device).ygen_id

        weights = compute_weights(device, target_ids, num_classes=6)  # to accomodate class imbalance
        loss = torch.nn.functional.cross_entropy(pred_ids_one_hot, target_ids, weight=weights)  # for classifying PID

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

    losses = losses.cpu().item() / len(loader)

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

    optimizer = torch.optim.SGD(mlpf.parameters(), lr=lr)
    if FineTune_VICReg:
        print("Will finetune VICReg during mlpf training")
        optimizer_VICReg = torch.optim.SGD(mlpf.parameters(), lr=lr * 0.1)
    else:
        print("Will fix VICReg during mlpf training")
        optimizer_VICReg = None

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses = train(device, encoder, mlpf, train_loader, valid_loader, optimizer, optimizer_VICReg, mode)
        losses_train.append(losses)

        # validation step
        losses = validation_run(device, encoder, mlpf, train_loader, valid_loader, mode)
        losses_valid.append(losses)

        # early-stopping
        if losses < best_val_loss:
            best_val_loss = losses
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
        ax.plot(range(len(losses_train)), losses_train, label="training")
        ax.plot(range(len(losses_valid)), losses_valid, label="validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
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
