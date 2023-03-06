import json
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def off_diagonal(x):
    """Copied from VICReg paper github https://github.com/facebookresearch/vicreg/"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# VICReg loss function
def criterion(tracks, clusters, loss_hparams):
    """Based on the pytorch pseudocode presented at the paper in Appendix A."""
    loss_ = {}

    N = tracks.size(0)  # batch size
    D = tracks.size(1)  # dim of representations

    # invariance loss
    loss_["Invariance"] = F.mse_loss(tracks, clusters)
    loss_["Invariance"] *= loss_hparams["lmbd"]

    # variance loss
    std_tracks = torch.sqrt(tracks.var(dim=0) + 1e-04)
    std_clusters = torch.sqrt(clusters.var(dim=0) + 1e-04)
    loss_["Variance"] = torch.mean(F.relu(1 - std_tracks)) + torch.mean(F.relu(1 - std_clusters))
    loss_["Variance"] *= loss_hparams["mu"]

    # covariance loss
    tracks = tracks - tracks.mean(dim=0)
    clusters = clusters - clusters.mean(dim=0)
    cov_tracks = (tracks.T @ tracks) / (N - 1)
    cov_clusters = (clusters.T @ clusters) / (N - 1)

    # loss_["Covariance"] = ( sum_off_diagonal(cov_tracks.pow_(2)) + sum_off_diagonal(cov_clusters.pow_(2)) ) / D
    loss_["Covariance"] = off_diagonal(cov_tracks).pow_(2).sum().div(D) + off_diagonal(cov_clusters).pow_(2).sum().div(D)
    loss_["Covariance"] *= loss_hparams["nu"]

    return loss_


@torch.no_grad()
def validation_run(multi_gpu, device, vicreg, loaders, loss_hparams):
    with torch.no_grad():
        optimizer = None
        ret = train(multi_gpu, device, vicreg, loaders, optimizer, loss_hparams)
    return ret


def train(multi_gpu, device, vicreg, loaders, optimizer, loss_hparams):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print("---->Initiating a training run")
        vicreg.train()
        loader = loaders["train"]
    else:
        print("---->Initiating a validation run")
        vicreg.eval()
        loader = loaders["valid"]

    # initialize loss counters
    losses_of_interest = ["Total", "Invariance", "Variance", "Covariance"]
    losses = {}
    for loss in losses_of_interest:
        losses[loss] = 0.0

    for i, batch in enumerate(loader):

        if multi_gpu:
            X = batch
        else:
            X = batch.to(device)

        # run VICReg forward pass to get the embeddings
        embedding_tracks, embedding_clusters = vicreg(X)

        # compute loss
        loss_ = criterion(embedding_tracks, embedding_clusters, loss_hparams)
        loss_["Total"] = loss_["Invariance"] + loss_["Variance"] + loss_["Covariance"]

        # update parameters
        if is_train:
            for param in vicreg.parameters():
                param.grad = None
            loss_["Total"].backward()
            optimizer.step()

        # accumulate the loss to make plots
        for loss in losses_of_interest:
            losses[loss] += loss_[loss].detach()

    for loss in losses_of_interest:
        losses[loss] = losses[loss].cpu().item() / (len(loader))

    return losses


def training_loop_VICReg(multi_gpu, device, vicreg, loaders, n_epochs, patience, optimizer, loss_hparams, outpath):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        multi_gpu: flag to indicate if there's more than 1 gpu available to use.
        device: "cpu" or "cuda".
        vicreg: the VICReg model composed of an Encoder/Decoder.
        loaders: a dict() object with keys "train" and "valid", each refering to a pytorch Dataloader.
        patience: number of stale epochs allowed before stopping the training.
        optimizer: optimizer to use for training (by default SGD which proved more stable).
        loss_hparams: a dict() object with keys "lmbd", "u", "v" containing loss hyperparameters.
        outpath: path to store the model weights and training plots
    """

    t0_initial = time.time()

    losses_of_interest = ["Total", "Invariance", "Variance", "Covariance"]

    best_val_loss, best_train_loss = {}, {}
    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        best_val_loss[loss] = 9999999.9
        losses["train"][loss] = []
        losses["valid"][loss] = []

    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses_t = train(multi_gpu, device, vicreg, loaders, optimizer, loss_hparams)

        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])

        # validation step
        losses_v = validation_run(multi_gpu, device, vicreg, loaders, loss_hparams)

        for loss in losses_of_interest:
            losses["valid"][loss].append(losses_v[loss])

        # save the lowest value of each component of the loss to print it on the legend of the loss plots
        for loss in losses_of_interest:
            if loss == "Total":
                if losses_v[loss] < best_val_loss[loss]:
                    best_val_loss[loss] = losses_v[loss]
                    best_train_loss[loss] = losses_t[loss]

                    # save the model differently if the model was wrapped with DataParallel
                    if multi_gpu:
                        state_dict = vicreg.module.state_dict()
                    else:
                        state_dict = vicreg.state_dict()

                    torch.save(state_dict, f"{outpath}/VICReg_best_epoch_weights.pth")

                    # dump best epoch
                    with open(f"{outpath}/VICReg_best_epoch.json", "w") as fp:
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

        print(
            f"epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_t['Total'], 4)} "
            + f"valid_loss={round(losses_v['Total'], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        for loss in losses_of_interest:

            # make total loss plot
            fig, ax = plt.subplots()
            ax.plot(
                range(len(losses["train"][loss])), losses["train"][loss], label=f"training ({best_train_loss[loss]:.4f})"
            )
            ax.plot(
                range(len(losses["valid"][loss])), losses["valid"][loss], label=f"validation ({best_val_loss[loss]:.4f})"
            )
            ax.set_xlabel("Epochs")
            ax.set_ylabel(f"{loss} Loss")
            ax.legend(
                title=rf"VICReg - ($\lambda={loss_hparams['lmbd']} - \mu={loss_hparams['mu']} - \nu={loss_hparams['nu']}$)",
                loc="best",
                title_fontsize=20,
                fontsize=15,
            )
            plt.tight_layout()
            plt.savefig(f"{outpath}/VICReg_loss_{loss}.pdf")
            plt.close()

        with open(f"{outpath}/VICReg_losses.pkl", "wb") as f:
            pkl.dump(losses, f)

        plt.tight_layout()
        plt.close()

    print("----------------------------------------------------------")
    print(f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min")
