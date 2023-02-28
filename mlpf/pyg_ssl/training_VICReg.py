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

# # VICReg loss function
# def criterion(x, y, device="cuda", lmbd=25, epsilon=1e-3):
#     bs = x.size(0)
#     emb = x.size(1)

#     std_x = torch.sqrt(x.var(dim=0) + epsilon)
#     std_y = torch.sqrt(y.var(dim=0) + epsilon)
#     var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

#     invar_loss = F.mse_loss(x, y)

#     xNorm = (x - x.mean(0)) / x.std(0)
#     yNorm = (y - y.mean(0)) / y.std(0)
#     crossCorMat = (xNorm.T @ yNorm) / bs
#     cross_loss = (crossCorMat * lmbd - torch.eye(emb, device=torch.device(device)) * lmbd).pow(2).sum()

#     return var_loss, invar_loss, cross_loss


def sum_off_diagonal(M):
    """Sums the off-diagonal elements of a square matrix M."""
    return M.sum() - torch.diagonal(M).sum()


def off_diagonal(x):
    """Copied from VICReg paper github https://github.com/facebookresearch/vicreg/"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# VICReg loss function
def criterion(tracks, clusters):
    """Based on the pytorch pseudocode presented at the paper in Appendix A."""
    loss_ = {}

    N = tracks.size(0)  # batch size
    D = tracks.size(1)  # dim of representations

    # invariance loss
    loss_["Invariance"] = F.mse_loss(tracks, clusters)

    # variance loss
    std_tracks = torch.sqrt(tracks.var(dim=0) + 1e-04)
    std_clusters = torch.sqrt(clusters.var(dim=0) + 1e-04)
    loss_["Variance"] = torch.mean(F.relu(1 - std_tracks)) + torch.mean(F.relu(1 - std_clusters))

    # covariance loss
    tracks = tracks - tracks.mean(dim=0)
    clusters = clusters - clusters.mean(dim=0)
    cov_tracks = (tracks.T @ tracks) / (N - 1)
    cov_clusters = (clusters.T @ clusters) / (N - 1)

    # loss_["Covariance"] = ( sum_off_diagonal(cov_tracks.pow_(2)) + sum_off_diagonal(cov_clusters.pow_(2)) ) / D
    loss_["Covariance"] = off_diagonal(cov_tracks).pow_(2).sum().div(D) + off_diagonal(cov_clusters).pow_(2).sum().div(D)

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
        loss_ = criterion(embedding_tracks, embedding_clusters)
        loss_["Total"] = (
            (loss_hparams["lmbd"] * loss_["Invariance"])
            + (loss_hparams["mu"] * loss_["Variance"])
            + (loss_hparams["nu"] * loss_["Covariance"])
        )

        # update parameters
        if is_train:
            for param in vicreg.parameters():
                param.grad = None
            loss_["Total"].backward()
            optimizer.step()

        # accumulate the loss to make plots
        for loss in losses_of_interest:
            losses[loss] += loss_[loss].detach().cpu().item() / (len(loader))

        print(f'debug: tot={losses["Total"]} - {losses["Invariance"]} - {losses["Variance"]} - {losses["Covariance"]}')

        if i == 10:
            break

    return losses


def training_loop_VICReg(multi_gpu, device, vicreg, loaders, n_epochs, patience, optimizer, loss_hparams, outpath):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        multi_gpu: flag to indicate if there's more than 1 gpu available to use.
        encoder: the VICReg model composed of an Ecnoder/Decoder.
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
            if losses_v[loss] < best_val_loss[loss]:
                best_val_loss[loss] = losses_v[loss]
                best_train_loss[loss] = losses_t[loss]

            if loss == "Total":  # for early-stopping purposes
                stale_epochs, best_epoch = 0, epoch

                # save the model
                torch.save(vicreg.state_dict(), f"{outpath}/VICReg_best_epoch_weights.pth")
            else:
                stale_epochs += 1

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
            ax.legend(title=f"VICReg - best epoch is {best_epoch}", loc="best", title_fontsize=20, fontsize=15)
            plt.savefig(f"{outpath}/VICReg_loss_{loss}.pdf")

        with open(f"{outpath}/VICReg_losses_train.pkl", "wb") as f:
            pkl.dump(losses_t, f)
        with open(f"{outpath}/VICReg_losses_valid.pkl", "wb") as f:
            pkl.dump(losses_v, f)

        plt.tight_layout()

    print("----------------------------------------------------------")
    print(f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min")
