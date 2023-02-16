import json
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .utils import distinguish_PFelements

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


# VICReg loss function
def criterion(x, y, device="cuda", lmbd=25, epsilon=1e-3):
    bs = x.size(0)
    emb = x.size(1)

    std_x = torch.sqrt(x.var(dim=0) + epsilon)
    std_y = torch.sqrt(y.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

    invar_loss = F.mse_loss(x, y)

    xNorm = (x - x.mean(0)) / x.std(0)
    yNorm = (y - y.mean(0)) / y.std(0)
    crossCorMat = (xNorm.T @ yNorm) / bs
    cross_loss = (crossCorMat * lmbd - torch.eye(emb, device=torch.device(device)) * lmbd).pow(2).sum()

    return var_loss, invar_loss, cross_loss


@torch.no_grad()
def validation_run(
    device,
    encoder,
    decoder,
    train_loader,
    valid_loader,
    lmbd,
    u,
    v,
):
    with torch.no_grad():
        optimizer = None
        ret = train(
            device,
            encoder,
            decoder,
            train_loader,
            valid_loader,
            optimizer,
            lmbd,
            u,
            v,
        )
    return ret


def train(
    device,
    encoder,
    decoder,
    train_loader,
    valid_loader,
    optimizer,
    lmbd,
    u,
    v,
):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """

    is_train = not (optimizer is None)

    if is_train:
        print("---->Initiating a training run")
        encoder.train()
        decoder.train()
        loader = train_loader
    else:
        print("---->Initiating a validation run")
        encoder.eval()
        decoder.eval()
        loader = valid_loader

    # initialize loss counters
    losses = 0.0
    var_losses, invar_losses, cross_losses = 0.0, 0.0, 0.0

    for i, batch in enumerate(loader):

        # make transformation
        tracks, clusters = distinguish_PFelements(batch.to(device))

        # ENCODE
        embedding_tracks, embedding_clusters = encoder(tracks, clusters)
        # POOLING
        pooled_tracks = global_mean_pool(embedding_tracks, tracks.batch)
        pooled_clusters = global_mean_pool(embedding_clusters, clusters.batch)
        # DECODE
        out_tracks, out_clusters = decoder(pooled_tracks, pooled_clusters)

        # compute loss
        var_loss, invar_loss, cross_loss = criterion(out_tracks, out_clusters, device, lmbd)
        loss = u * var_loss + v * invar_loss + cross_loss

        # update parameters
        if is_train:
            for param in encoder.parameters():
                param.grad = None
            for param in decoder.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()

        losses += loss.detach()
        var_losses += var_loss.detach()
        invar_losses += invar_loss.detach()
        cross_losses += cross_loss.detach()

    losses = losses.cpu().item() / (len(loader))
    var_losses = var_losses.cpu().item() / (len(loader))
    invar_losses = invar_losses.cpu().item() / (len(loader))
    cross_losses = cross_losses.cpu().item() / (len(loader))

    return losses, var_losses, invar_losses, cross_losses


def training_loop_VICReg(
    device,
    encoder,
    decoder,
    train_loader,
    valid_loader,
    n_epochs,
    patience,
    optimizer,
    outpath,
    lmbd,
    u,
    v,
):
    """
    Main function to perform training. Will call the train() and validation_run() functions every epoch.

    Args:
        encoder: the encoder part of VICReg
        decoder: the decoder part of VICReg
        train_loader: a pytorch Dataloader for training
        valid_loader: a pytorch Dataloader for validation
        patience: number of stale epochs allowed before stopping the training
        optimizer: optimizer to use for training (by default: Adam)
        outpath: path to store the model weights and training plots
    """

    t0_initial = time.time()

    losses_train_tot, losses_train_var, losses_train_invar, losses_train_cross = [], [], [], []
    losses_valid_tot, losses_valid_var, losses_valid_invar, losses_valid_cross = [], [], [], []

    best_val_loss_tot, best_val_loss_var, best_val_loss_invar, best_val_cross = 99999.9, 99999.9, 99999.9, 99999.9
    stale_epochs = 0

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses_t_tot, losses_t_var, losses_t_invar, losses_t_cross = train(
            device,
            encoder,
            decoder,
            train_loader,
            valid_loader,
            optimizer,
            lmbd,
            u,
            v,
        )

        losses_train_tot.append(losses_t_tot)
        losses_train_var.append(losses_t_var)
        losses_train_invar.append(losses_t_invar)
        losses_train_cross.append(losses_t_cross)

        # validation step
        losses_v_tot, losses_v_var, losses_v_invar, losses_v_cross = validation_run(
            device,
            encoder,
            decoder,
            train_loader,
            valid_loader,
            lmbd,
            u,
            v,
        )

        losses_valid_tot.append(losses_v_tot)
        losses_valid_var.append(losses_v_var)
        losses_valid_invar.append(losses_v_invar)
        losses_valid_cross.append(losses_v_cross)

        # save the lowest value of each component of the loss to print it on the legend of the loss plots
        if losses_v_var < best_val_loss_var:
            best_val_loss_var = losses_v_var
            best_train_loss_var = losses_t_var

        if losses_v_invar < best_val_loss_invar:
            best_val_loss_invar = losses_v_invar
            best_train_loss_invar = losses_t_invar

        if losses_v_cross < best_val_cross:
            best_val_loss_cross = losses_v_cross
            best_train_loss_cross = losses_t_cross

        # early-stopping
        if losses_v_tot < best_val_loss_tot:
            best_val_loss_tot = losses_v_tot
            best_train_loss_tot = losses_t_tot

            stale_epochs = 0

            try:
                encoder_state_dict = encoder.module.state_dict()
            except AttributeError:
                encoder_state_dict = encoder.state_dict()
            try:
                decoder_state_dict = decoder.module.state_dict()
            except AttributeError:
                decoder_state_dict = decoder.state_dict()

            torch.save(encoder_state_dict, f"{outpath}/encoder_best_epoch_weights.pth")
            torch.save(decoder_state_dict, f"{outpath}/decoder_best_epoch_weights.pth")

            with open(f"{outpath}/VICReg_best_epoch.json", "w") as fp:  # dump best epoch
                json.dump({"best_epoch": epoch}, fp)
        else:
            stale_epochs += 1

        t1 = time.time()

        epochs_remaining = n_epochs - (epoch + 1)
        time_per_epoch = (t1 - t0_initial) / (epoch + 1)
        eta = epochs_remaining * time_per_epoch / 60

        print(
            f"epoch={epoch + 1} / {n_epochs} "
            + f"train_loss={round(losses_train_tot[epoch], 4)} "
            + f"valid_loss={round(losses_valid_tot[epoch], 4)} "
            + f"stale={stale_epochs} "
            + f"time={round((t1-t0)/60, 2)}m "
            + f"eta={round(eta, 1)}m"
        )

        # make total loss plot
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train_tot)), losses_train_tot, label="training ({:.2f})".format(best_train_loss_tot))
        ax.plot(range(len(losses_valid_tot)), losses_valid_tot, label="validation ({:.2f})".format(best_val_loss_tot))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Total Loss")
        ax.legend(title="VICReg", loc="best", title_fontsize=20, fontsize=15)
        plt.savefig(f"{outpath}/VICReg_loss_tot.pdf")

        with open(f"{outpath}/VICReg_loss_train_tot.pkl", "wb") as f:
            pkl.dump(losses_train_tot, f)
        with open(f"{outpath}/VICReg_loss_valid_tot.pkl", "wb") as f:
            pkl.dump(losses_valid_tot, f)

        # make variance loss plot
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train_var)), losses_train_var, label="training ({:.2f})".format(best_train_loss_var))
        ax.plot(range(len(losses_valid_var)), losses_valid_var, label="validation ({:.2f})".format(best_val_loss_var))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Variance Loss")
        ax.legend(title="VICReg", loc="best", title_fontsize=20, fontsize=15)
        plt.savefig(f"{outpath}/VICReg_loss_var.pdf")

        with open(f"{outpath}/VICReg_loss_train_var.pkl", "wb") as f:
            pkl.dump(losses_train_var, f)
        with open(f"{outpath}/VICReg_loss_valid_var.pkl", "wb") as f:
            pkl.dump(losses_valid_var, f)

        # make invariance loss plot
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train_invar)), losses_train_invar, label="training ({:.2f})".format(best_train_loss_invar))
        ax.plot(range(len(losses_valid_invar)), losses_valid_invar, label="validation ({:.2f})".format(best_val_loss_invar))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Invariance Loss")
        ax.legend(title="VICReg", loc="best", title_fontsize=20, fontsize=15)
        plt.savefig(f"{outpath}/VICReg_loss_invar.pdf")

        with open(f"{outpath}/VICReg_loss_train_invar.pkl", "wb") as f:
            pkl.dump(losses_train_invar, f)
        with open(f"{outpath}/VICReg_loss_valid_invar.pkl", "wb") as f:
            pkl.dump(losses_valid_invar, f)

        # make covariance loss plot
        fig, ax = plt.subplots()
        ax.plot(range(len(losses_train_cross)), losses_train_cross, label="training ({:.2f})".format(best_train_loss_cross))
        ax.plot(range(len(losses_valid_cross)), losses_valid_cross, label="validation ({:.2f})".format(best_val_loss_cross))
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Covariance Loss")
        ax.legend(title="VICReg", loc="best", title_fontsize=20, fontsize=15)
        plt.savefig(f"{outpath}/VICReg_loss_cov.pdf")

        with open(f"{outpath}/VICReg_loss_train_cov.pkl", "wb") as f:
            pkl.dump(losses_train_cross, f)
        with open(f"{outpath}/VICReg_loss_valid_cov.pkl", "wb") as f:
            pkl.dump(losses_valid_cross, f)

    print("----------------------------------------------------------")
    print(f"Done with training. Total training time is {round((time.time() - t0_initial)/60,3)}min")
