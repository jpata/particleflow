import pickle as pkl
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from pyg.logger import _logger
from pyg.utils import (
    get_model_state_dict,
    save_checkpoint,
    unpack_predictions,
    unpack_target,
)
from torch.utils.tensorboard import SummaryWriter

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")


def configure_model_trainable(model, trainable, is_training):
    if is_training:
        model.train()
        if trainable != "all":
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            for layer in trainable:
                layer = getattr(model, layer)
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        model.eval()


def train_and_valid(
    rank,
    world_size,
    deepmet,
    mlpf,
    backbone_mode,
    downstream_input,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    epoch=None,
    dtype=torch.float32,
):
    """
    Performs training over a given epoch.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    configure_model_trainable(mlpf, trainable, False if backbone_mode == "freeze" else is_train)
    configure_model_trainable(deepmet, trainable, is_train)

    epoch_loss = {}  # this one will keep accumulating `train_loss` and then return the average

    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(data_loader)
    else:
        iterator = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}"
        )

    device_type = "cuda" if isinstance(rank, int) else "cpu"

    loss = {}  # this one is redefined every iteration

    if downstream_input == "latents":  # must set forward hooks to retrieve the intermediate latent representations
        latent_reps = {}

        def get_latent_reps(name):
            def hook(mlpf, input, output):
                latent_reps[name] = output  # note: with gradients set to True unless --freeze-backbone is True

            return hook

        if isinstance(mlpf, torch.nn.parallel.distributed.DistributedDataParallel):
            mlpf.module.conv_reg[2].dropout.register_forward_hook(get_latent_reps("conv_reg2"))
            mlpf.module.nn_id.register_forward_hook(get_latent_reps("nn_id"))
        else:
            mlpf.conv_reg[2].dropout.register_forward_hook(get_latent_reps("conv_reg2"))
            mlpf.nn_id.register_forward_hook(get_latent_reps("nn_id"))

    for itrain, batch in iterator:

        batch = batch.to(rank, non_blocking=True)

        ygen = unpack_target(batch.ygen)
        ycand = unpack_target(batch.ycand)

        # ----------------------- Run backbone inference -----------------------

        if downstream_input == "pfcands":  # no need to use the backbone

            msk_ycand = ycand["cls_id"] != 0
            reco_px = (ycand["pt"] * ycand["cos_phi"]) * msk_ycand
            reco_py = (ycand["pt"] * ycand["sin_phi"]) * msk_ycand

            X = torch.cat([ycand["momentum"], ycand["cls_id"].unsqueeze(-1)], axis=-1)
            X = X * msk_ycand.unsqueeze(-1)  # run downstream on actual particles (i.e. ignore the Nulls)

        else:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device_type == "cuda"):
                if is_train and not backbone_mode == "freeze":
                    ymlpf = mlpf(batch.X, batch.mask)
                else:
                    with torch.no_grad():
                        ymlpf = mlpf(batch.X, batch.mask)

            ymlpf = unpack_predictions(ymlpf)

            msk_ymlpf = ymlpf["cls_id"] != 0
            reco_px = (ymlpf["pt"] * ymlpf["cos_phi"]) * msk_ymlpf
            reco_py = (ymlpf["pt"] * ymlpf["sin_phi"]) * msk_ymlpf

            if downstream_input == "latents":  # use the latent representations
                for layer in latent_reps:
                    if backbone_mode == "freeze":
                        latent_reps[layer] = latent_reps[layer].detach()

                    if "conv" in layer:
                        latent_reps[layer] *= batch.mask.unsqueeze(-1)

                X = torch.cat(
                    [
                        batch.X,  # 17
                        latent_reps["conv_reg2"],  # 256
                        latent_reps["nn_id"],  # 6
                    ],
                    axis=-1,
                )

            elif downstream_input == "mlpfcands":  # use the MLPF cands
                X = torch.cat([ymlpf["momentum"], ymlpf["cls_id_onehot"]], axis=-1)

            X = X * msk_ymlpf.unsqueeze(-1)  # run downstream on actual particles (i.e. ignore the Nulls)

        # ----------------------- Run finetuning -----------------------

        if is_train:
            w = deepmet(X)

        else:
            with torch.no_grad():
                w = deepmet(X)

        pred_met_x = torch.sum(w * reco_px, axis=1)
        pred_met_y = torch.sum(w * reco_py, axis=1)

        # get the gen MET to compute the loss
        msk_gen = ygen["cls_id"] != 0
        gen_px = (ygen["pt"] * ygen["cos_phi"]) * msk_gen
        gen_py = (ygen["pt"] * ygen["sin_phi"]) * msk_gen

        true_met_x = torch.sum(gen_px, axis=1)
        true_met_y = torch.sum(gen_py, axis=1)

        if is_train:
            loss["MET"] = torch.nn.MSELoss(true_met_x, pred_met_x) + torch.nn.MSELoss(true_met_y, pred_met_y)

            for param in deepmet.parameters():
                param.grad = None

            for param in mlpf.parameters():
                param.grad = None

            loss["MET"].backward()
            optimizer.step()

        else:
            with torch.no_grad():
                loss["MET"] = torch.nn.MSELoss(true_met_x, pred_met_x) + torch.nn.MSELoss(true_met_y, pred_met_y)

        # monitor the MLPF and PF MET loss
        with torch.no_grad():
            if downstream_input != "pfcands":  # monitor MLPF loss only if the backbone inference was on
                loss["MET_MLPF"] = torch.nn.MSELoss(true_met_x, torch.sum(reco_px, axis=1)) + torch.nn.MSELoss(
                    true_met_y, torch.sum(reco_py, axis=1)
                )

            msk_ycand = ycand["cls_id"] != 0
            cand_px = (ycand["pt"] * ycand["cos_phi"]) * msk_ycand
            cand_py = (ycand["pt"] * ycand["sin_phi"]) * msk_ycand

            loss["MET_PF"] = torch.nn.MSELoss(true_met_x, torch.sum(cand_px, axis=1)) + torch.nn.MSELoss(
                true_met_y, torch.sum(cand_py, axis=1)
            )

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

    num_data = torch.tensor(len(data_loader), device=rank)

    # sum up the number of steps from all workers
    if world_size > 1:
        torch.distributed.all_reduce(num_data)

    for loss_ in epoch_loss:
        # sum up the losses from all workers
        if world_size > 1:
            torch.distributed.all_reduce(epoch_loss[loss_])
        epoch_loss[loss_] = epoch_loss[loss_].cpu().item() / num_data.cpu().item()

    if world_size > 1:
        dist.barrier()

    return epoch_loss


def finetune_mlpf(
    rank,
    world_size,
    deepmet,
    mlpf,
    backbone_mode,
    downstream_input,
    optimizer,
    train_loader,
    valid_loader,
    num_epochs,
    patience,
    outdir,
    trainable="all",
    dtype=torch.float32,
    checkpoint_freq=None,
):
    """
    Will run a full training by calling train().

    Args:
        rank: 'cpu' or int representing the gpu device id
        model: a pytorch model (may be wrapped by DistributedDataParallel)
        train_loader: a pytorch geometric Dataloader that loads the training data in the form ~ DataBatch(X, ygen, ycands)
        valid_loader: a pytorch geometric Dataloader that loads the validation data in the form ~ DataBatch(X, ygen, ycands)
        patience: number of stale epochs before stopping the training
        outdir: path to store the model weights and training plots
    """
    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
        tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")
    else:
        tensorboard_writer_train = None
        tensorboard_writer_valid = None

    t0_initial = time.time()

    losses_of_interest = ["MET", "MET_PF"]
    if downstream_input != "pfcands":  # monitor MLPF loss only if the backbone inference was run
        losses_of_interest += ["MET_MLPF"]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), float("inf")

    start_epoch = 1
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        losses_t = train_and_valid(
            rank,
            world_size,
            deepmet,
            mlpf,
            backbone_mode,
            downstream_input,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=True,
            epoch=epoch,
            dtype=dtype,
        )

        losses_v = train_and_valid(
            rank,
            world_size,
            deepmet,
            mlpf,
            backbone_mode,
            downstream_input,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            epoch=epoch,
            dtype=dtype,
        )

        if (rank == 0) or (rank == "cpu"):
            extra_state = {"epoch": epoch}
            if losses_v["MET"] < best_val_loss:
                best_val_loss = losses_v["MET"]
                stale_epochs *= 0

                torch.save(
                    {
                        "mlpf_state_dict": get_model_state_dict(mlpf),
                        "deepmet_state_dict": get_model_state_dict(deepmet),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"{outdir}/best_weights.pth",
                )
                save_checkpoint(f"{outdir}/best_weights_mlpf.pth", mlpf, optimizer, extra_state)
                save_checkpoint(f"{outdir}/best_weights_deepmet.pth", deepmet, optimizer, extra_state)
            else:
                stale_epochs += 1

        if world_size > 1:
            dist.barrier()
            torch.distributed.broadcast(stale_epochs, src=0)

        if stale_epochs > patience:
            if (rank == 0) or (rank == "cpu"):
                _logger.info(f"stale_epochs = {patience}, will stop the training.")
            break

        if (rank == 0) or (rank == "cpu"):
            if checkpoint_freq and (epoch != 0) and (epoch % checkpoint_freq == 0):
                checkpoint_dir = Path(outdir) / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}_mlpf.pth".format(checkpoint_dir, epoch, losses_v["MET"])
                save_checkpoint(checkpoint_path, mlpf, optimizer, extra_state)
                checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}_deepmet.pth".format(checkpoint_dir, epoch, losses_v["MET"])
                save_checkpoint(checkpoint_path, deepmet, optimizer, extra_state)

            for k, v in losses_t.items():
                tensorboard_writer_train.add_scalar("epoch/loss_" + k, v, epoch)

            for loss in losses_of_interest:
                losses["train"][loss].append(losses_t[loss])
                losses["valid"][loss].append(losses_v[loss])

            for k, v in losses_v.items():
                tensorboard_writer_valid.add_scalar("epoch/loss_" + k, v, epoch)

            t1 = time.time()

            epochs_remaining = num_epochs - epoch
            time_per_epoch = (t1 - t0_initial) / epoch
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch} / {num_epochs} "
                + f"train_loss={losses_t['MET']:.4f} "
                + f"valid_loss={losses_v['MET']:.4f} "
                + f"stale={stale_epochs} "
                + f"time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m",
                color="bold",
            )

            with open(f"{outdir}/met_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

            if tensorboard_writer_train:
                tensorboard_writer_train.flush()
            if tensorboard_writer_valid:
                tensorboard_writer_valid.flush()

    if world_size > 1:
        dist.barrier()

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")


def override_config(config, args):
    """override config with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            config[arg] = arg_value

    if not (args.attention_type is None):
        config["model"]["attention"]["attention_type"] = args.attention_type

    if not (args.num_convs is None):
        for model in ["gnn_lsh", "gravnet", "attention", "attention", "mamba"]:
            config["model"][model]["num_convs"] = args.num_convs

    args.test_datasets = config["test_dataset"]

    return config
