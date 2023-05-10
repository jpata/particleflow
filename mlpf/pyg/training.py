import json
import math
import pickle as pkl
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import tqdm

# https://github.com/mathiaszinnen/focal_loss_torch
from focal_loss.focal_loss import FocalLoss
from pyg.ssl.utils import combine_PFelements, distinguish_PFelements
from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

# keep track of step across epochs
ISTEP_GLOBAL_TRAIN = 0
ISTEP_GLOBAL_VALID = 0


def compute_weights(gen_ids_one_hot, device):
    output_dim_id = len(torch.unique(gen_ids_one_hot))
    vs, cs = torch.unique(gen_ids_one_hot, return_counts=True)
    weights = torch.zeros(output_dim_id).to(device=device)
    for k, v in zip(vs, cs):
        weights[k] = 1.0 / math.sqrt(float(v))
    return weights


@torch.no_grad()
def validation_run(
    rank,
    model,
    train_loader,
    valid_loader,
    batch_size,
    ssl_encoder=None,
    tensorboard_writer=None,
    alpha=-1,
    penalize_NCH=False,
):
    with torch.no_grad():
        optimizer = None
        ret = train(
            rank,
            model,
            train_loader,
            valid_loader,
            batch_size,
            optimizer,
            ssl_encoder,
            tensorboard_writer,
            alpha,
            penalize_NCH,
        )
    return ret


def train(
    rank,
    mlpf,
    train_loader,
    valid_loader,
    batch_size,
    optimizer,
    ssl_encoder=None,
    tensorboard_writer=None,
    alpha=-1,
    penalize_NCH=False,
):
    """
    A training/validation run over a given epoch that gets called in the training_loop() function.
    When optimizer is set to None, it freezes the model for a validation_run.
    """
    softmax = torch.nn.Softmax(dim=-1)

    global ISTEP_GLOBAL_TRAIN, ISTEP_GLOBAL_VALID
    is_train = not (optimizer is None)

    step_type = "train" if is_train else "valid"

    if is_train:
        print(f"---->Initiating a training run on rank {rank}")
        mlpf.train()
        file_loader = train_loader
    else:
        print(f"---->Initiating a validation run rank {rank}")
        mlpf.eval()
        file_loader = valid_loader

    # initialize loss counters
    losses_of_interest = ["Total", "Classification", "Regression", "Charge"]
    losses = {}
    for loss in losses_of_interest:
        losses[loss] = 0.0

    tf_0, tf_f = time.time(), 0
    for num, file in enumerate(file_loader):
        if "utils" in str(type(file_loader)):  # it must be converted to a pyg DataLoader if it's not (only needed for CMS)
            print(f"Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - tf_0, 3)}s")
            tf_f = tf_f + (time.time() - tf_0)
            file = torch_geometric.loader.DataLoader([x for t in file for x in t], batch_size=batch_size)

        tf = 0
        for i, batch in tqdm.tqdm(enumerate(file), total=len(file)):
            if tensorboard_writer:
                tensorboard_writer.add_scalar(
                    "step_{}/num_elems".format(step_type),
                    batch.x.shape[0],
                    ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
                )

            if ssl_encoder is not None:
                # separate PF-elements
                tracks, clusters = distinguish_PFelements(batch.to(rank))
                # ENCODE
                embedding_tracks, embedding_clusters = ssl_encoder(tracks, clusters)
                # concat the inputs with embeddings
                tracks.x = torch.cat([batch.x[batch.x[:, 0] == 1], embedding_tracks], axis=1)
                clusters.x = torch.cat([batch.x[batch.x[:, 0] == 2], embedding_clusters], axis=1)
                # combine PF-elements
                event = combine_PFelements(tracks, clusters).to(rank)

            else:
                event = batch.to(rank)

            # make mlpf forward pass
            t0 = time.time()
            pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event)
            tf = tf + (time.time() - t0)

            target_ids = event.ygen_id
            for icls in range(pred_ids_one_hot.shape[1]):
                if tensorboard_writer:
                    tensorboard_writer.add_scalar(
                        "step_{}/num_cls_{}".format(step_type, icls),
                        torch.sum(target_ids == icls),
                        ISTEP_GLOBAL_TRAIN if is_train else ISTEP_GLOBAL_VALID,
                    )

            target_momentum = event.ygen[:, 1:].to(dtype=torch.float32)
            target_charge = (event.ygen[:, 0] + 1).to(dtype=torch.float32)  # -1, 0, 1 -> 0, 1, 2
            assert np.all(target_charge.unique().cpu().numpy() == [0, 1, 2])

            loss_ = {}
            # for CLASSIFYING PID
            weights = compute_weights(target_ids, rank)
            if penalize_NCH:
                weights[5] = 0  # penalize the charged hadron predictions?
            loss_obj_id = 100 * FocalLoss(gamma=2.0, weights=weights)

            loss_["Classification"] = 100 * loss_obj_id(softmax(pred_ids_one_hot), target_ids)

            # REGRESSING p4: mask the loss in cases there is no true particle (when target_ids>4)
            if alpha == -1:  # old code
                msk_true_particle = torch.unsqueeze((target_ids != 0).to(dtype=torch.float32), axis=-1)
                loss_["Regression"] = 10 * torch.nn.functional.huber_loss(
                    pred_momentum * msk_true_particle, target_momentum * msk_true_particle
                )
                loss_["Charge"] = torch.nn.functional.cross_entropy(
                    pred_charge * msk_true_particle, (target_charge * msk_true_particle[:, 0]).to(dtype=torch.int64)
                )
            else:
                msk_true_particle = torch.unsqueeze((target_ids <= 4).to(dtype=torch.float32), axis=-1)
                msk_null_particle = torch.unsqueeze((target_ids > 4).to(dtype=torch.float32), axis=-1)
                loss_["Regression"] = 10 * torch.nn.functional.huber_loss(
                    pred_momentum * msk_true_particle, target_momentum * msk_true_particle
                )
                loss_["Regression"] += (
                    alpha
                    * 10
                    * torch.nn.functional.huber_loss(pred_momentum * msk_null_particle, target_momentum * msk_null_particle)
                )
                loss_["Charge"] = torch.nn.functional.cross_entropy(
                    pred_charge * msk_true_particle, (target_charge * msk_true_particle[:, 0]).to(dtype=torch.int64)
                )
                loss_["Charge"] += alpha * torch.nn.functional.cross_entropy(
                    pred_charge * msk_null_particle, (target_charge * msk_null_particle[:, 0]).to(dtype=torch.int64)
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

        print(f"Average inference time per batch on rank {rank} is {(tf / len(file)):.3f}s")

    for loss in losses_of_interest:
        losses[loss] = losses[loss].cpu().item() / (len(file) * (len(file_loader)))

    print(
        "loss_id={:.4f} loss_momentum={:.4f} loss_charge={:.4f}".format(
            losses["Classification"], losses["Regression"], losses["Charge"]
        )
    )

    return losses


def training_loop(
    rank,
    mlpf,
    train_loader,
    valid_loader,
    batch_size,
    n_epochs,
    patience,
    lr,
    alpha=-1,
    outpath="",
    ssl_encoder=None,
    penalize_NCH=False,
):
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
        ssl_encoder: the encoder part of VICReg. If None is provided then the function will run a supervised training.
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

    if ssl_encoder is not None:
        mode = "ssl"
        ssl_encoder.eval()
    else:
        mode = "native"
    print(f"Will launch a {mode} training of MLPF.")

    for epoch in range(n_epochs):
        t0 = time.time()

        if stale_epochs > patience:
            print("breaking due to stale epochs")
            break

        # training step
        losses_t = train(
            rank,
            mlpf,
            train_loader,
            valid_loader,
            batch_size,
            optimizer,
            ssl_encoder,
            tensorboard_writer,
            alpha,
            penalize_NCH,
        )
        for k, v in losses_t.items():
            tensorboard_writer.add_scalar("epoch/train_loss_" + k, v, epoch)
        for loss in losses_of_interest:
            losses["train"][loss].append(losses_t[loss])

        # validation step
        losses_v = validation_run(
            rank, mlpf, train_loader, valid_loader, batch_size, ssl_encoder, tensorboard_writer, alpha, penalize_NCH
        )
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

        print(
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
            if mode == "ssl":
                ax.legend(title="SSL-based MLPF", loc="best", title_fontsize=20, fontsize=15)
            else:
                ax.legend(title="Native MLPF", loc="best", title_fontsize=20, fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{outpath}/mlpf_{mode}_loss_{loss}.pdf")
            plt.close()
        with open(f"{outpath}/mlpf_{mode}_losses.pkl", "wb") as f:
            pkl.dump(losses, f)

        print("----------------------------------------------------------")
    print(f"Done with training. Total training time on rank {rank} is {round((time.time() - t0_initial)/60,3)}min")
