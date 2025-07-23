import os
import os.path as osp
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import tqdm
import yaml
import json
import sklearn
import sklearn.metrics
import numpy as np
from typing import Union, List

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from mlpf.model.logger import _logger, _configLogger
from mlpf.model.utils import (
    unpack_predictions,
    unpack_target,
    load_checkpoint,
    save_checkpoint,
    CLASS_LABELS,
    X_FEATURES,
    ELEM_TYPES_NONZERO,
    save_HPs,
    get_lr_schedule,
    count_parameters,
)
from mlpf.model.monitoring import log_open_files_to_tensorboard, log_step_to_tensorboard
from mlpf.model.inference import make_plots, run_predictions
from mlpf.model.mlpf import set_save_attention
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import Collater, PFDataset, get_interleaved_dataloaders
from mlpf.model.losses import mlpf_loss
from mlpf.utils import create_comet_experiment
from mlpf.model.plots import validation_plots
from mlpf.optimizers.lamb import Lamb


def configure_model_trainable(model: MLPF, trainable: Union[str, List[str]], is_training: bool):
    """Set only the given layers as trainable in the model"""

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise Exception("configure trainability before distributing the model")
    if is_training:
        model.train()
        if trainable != "all":
            model.eval()

            # first set all parameters as non-trainable
            for param in model.parameters():
                param.requires_grad = False

            # now explicitly enable specific layers
            for layer in trainable:
                layer = getattr(model, layer)
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        model.eval()


def model_step(batch, model, loss_fn):
    ypred_raw = model(batch.X, batch.mask)
    ypred = unpack_predictions(ypred_raw)
    ytarget = unpack_target(batch.ytarget, model)
    loss_opt, losses_detached = loss_fn(ytarget, ypred, batch)
    return loss_opt, losses_detached, ypred_raw, ypred, ytarget


def optimizer_step(model, loss_opt, optimizer, lr_schedule, scaler):
    # Clear gradients
    for param in model.parameters():
        param.grad = None

    # Backward pass and optimization
    scaler.scale(loss_opt).backward()
    scaler.step(optimizer)
    scaler.update()
    if lr_schedule:
        # ReduceLROnPlateau scheduler should only be updated after each full epoch
        if not isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_schedule.step()


def get_optimizer(model, config):
    """
    Returns the optimizer for the given model based on the configuration provided.
    Parameters:
    model (torch.nn.Module): The model for which the optimizer is to be created.
    config (dict): Configuration dictionary containing optimizer settings.
                   Must include the key "lr" for learning rate.
                   Optionally includes the key "optimizer" to specify the type of optimizer.
                   Supported values for "optimizer" are "adamw", "lamb", and "sgd".
                   If "optimizer" is not provided, "adamw" is used by default.
    Returns:
    torch.optim.Optimizer: The optimizer specified in the configuration.
    Raises:
    ValueError: If the specified optimizer type is not supported.
    """

    wd = config["weight_decay"] if "weight_decay" in config else 0.01
    if "optimizer" not in config:
        return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=wd)
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=wd)
    elif config["optimizer"] == "lamb":
        return Lamb(model.parameters(), lr=config["lr"], weight_decay=wd)
    elif config["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["lr"], weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")


def train_epoch(
    rank: Union[int, str],
    world_size: int,
    model: MLPF,
    optimizer,
    train_loader,
    lr_schedule,
    epoch: int,
    tensorboard_writer=None,
    comet_experiment=None,
    comet_step_freq=None,
    checkpoint_dir="",
    device_type="cuda",
    dtype=torch.float32,
    scaler=None,
):
    """Run one training epoch

    Args:
        rank: Device rank (GPU id or 'cpu')
        world_size: Number of devices being used
        model: The neural network model
        optimizer: The optimizer
        train_loader: Training data loader
        lr_schedule: Learning rate scheduler
        epoch: Current epoch number
        tensorboard_writer: TensorBoard writer object
        comet_experiment: Comet.ml experiment object
        comet_step_freq: How often to log to comet
        checkpoint_dir: Directory to save checkpoints
        device_type: 'cuda' or 'cpu'
        dtype: Torch dtype for computations

    Returns:
        dict: Dictionary of epoch losses
    """
    model.train()
    epoch_loss = {}

    # Only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(train_loader)
    else:
        iterator = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} train loop on rank={rank}")

    for itrain, batch in iterator:
        batch = batch.to(rank, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            loss_opt, loss, _, _, _ = model_step(batch, model, mlpf_loss)

        optimizer_step(model, loss_opt, optimizer, lr_schedule, scaler)

        # Accumulate losses
        for loss_name in loss:
            if loss_name not in epoch_loss:
                epoch_loss[loss_name] = 0.0
            epoch_loss[loss_name] += loss[loss_name]

        # Log step metrics
        step = (epoch - 1) * len(train_loader) + itrain
        if tensorboard_writer is not None and step % 100 == 0:
            log_open_files_to_tensorboard(tensorboard_writer, step)
            log_step_to_tensorboard(batch, loss["Total"], lr_schedule, tensorboard_writer, step)
            tensorboard_writer.flush()

            # Save step checkpoint
            extra_state = {"step": step, "lr_schedule_state_dict": lr_schedule.state_dict()}
            save_checkpoint(f"{checkpoint_dir}/step_weights.pth", model, optimizer, extra_state)

        if comet_experiment is not None and (itrain % comet_step_freq == 0):
            comet_experiment.log_metrics(loss, prefix="train", step=step)
            comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), step=step)

    # Average losses across steps
    num_steps = torch.tensor(float(len(train_loader)), device=rank, dtype=torch.float32)
    if world_size > 1:
        torch.distributed.all_reduce(num_steps)

    for loss_name in epoch_loss:
        if world_size > 1:
            torch.distributed.all_reduce(epoch_loss[loss_name])
        epoch_loss[loss_name] = epoch_loss[loss_name].cpu().item() / num_steps.cpu().item()

    if world_size > 1:
        dist.barrier()

    return epoch_loss


def eval_epoch(
    rank: Union[int, str],
    world_size: int,
    model: MLPF,
    valid_loader,
    epoch: int,
    tensorboard_writer=None,
    comet_experiment=None,
    save_attention=False,
    outdir=None,
    device_type="cuda",
    dtype=torch.float32,
):
    """Run one evaluation epoch

    Args:
        rank: Device rank (GPU id or 'cpu')
        world_size: Number of devices being used
        model: The neural network model
        valid_loader: Validation data loader
        epoch: Current epoch number
        tensorboard_writer: TensorBoard writer object
        comet_experiment: Comet.ml experiment object
        save_attention: Whether to save attention weights
        outdir: Output directory path
        device_type: 'cuda' or 'cpu'
        dtype: Torch dtype for computations

    Returns:
        dict: Dictionary of epoch losses
    """
    model.eval()
    epoch_loss = {}

    # Confusion matrix tracking
    cm_X_target = np.zeros((13, 13))
    cm_X_pred = np.zeros((13, 13))
    cm_id = np.zeros((13, 13))

    # Only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(valid_loader)
    else:
        iterator = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch} eval loop on rank={rank}")

    for ival, batch in iterator:
        batch = batch.to(rank, non_blocking=True)

        # Save attention on first batch if requested
        if save_attention and (rank == 0 or rank == "cpu") and ival == 0:
            set_save_attention(model, outdir, True)
        else:
            set_save_attention(model, outdir, False)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            with torch.no_grad():
                loss_opt, loss, ypred_raw, ypred, ytarget = model_step(batch, model, mlpf_loss)

        # Update confusion matrices
        cm_X_target += sklearn.metrics.confusion_matrix(
            batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ytarget["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
        )
        cm_X_pred += sklearn.metrics.confusion_matrix(
            batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
        )
        cm_id += sklearn.metrics.confusion_matrix(
            ytarget["cls_id"][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
        )

        # Save validation plots for first batch
        if (rank == 0 or rank == "cpu") and ival == 0:
            validation_plots(batch, ypred_raw, ytarget, ypred, tensorboard_writer, epoch, outdir)

        # Accumulate losses
        for loss_name in loss:
            if loss_name not in epoch_loss:
                epoch_loss[loss_name] = 0.0
            epoch_loss[loss_name] += loss[loss_name]

    # Log confusion matrices
    if comet_experiment:
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_target, title="Element to target", row_label="X", column_label="target", epoch=epoch, file_name="cm_X_target.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_pred, title="Element to pred", row_label="X", column_label="pred", epoch=epoch, file_name="cm_X_pred.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_id, title="Target to pred", row_label="target", column_label="pred", epoch=epoch, file_name="cm_id.json"
        )

    # Average losses across steps
    num_steps = torch.tensor(float(len(valid_loader)), device=rank, dtype=torch.float32)
    if world_size > 1:
        torch.distributed.all_reduce(num_steps)

    for loss_name in epoch_loss:
        if world_size > 1:
            torch.distributed.all_reduce(epoch_loss[loss_name])
        epoch_loss[loss_name] = epoch_loss[loss_name].cpu().item() / num_steps.cpu().item()

    if world_size > 1:
        dist.barrier()

    return epoch_loss


def train_all_epochs(
    rank,
    world_size,
    model,
    optimizer,
    train_loader,
    valid_loader,
    num_epochs,
    patience,
    outdir,
    config,
    trainable="all",
    dtype=torch.float32,
    start_epoch=1,
    lr_schedule=None,
    use_ray=False,
    checkpoint_freq=None,
    comet_experiment=None,
    comet_step_freq=None,
    val_freq=None,
    save_attention=False,
    checkpoint_dir="",
):
    """Main training loop that handles all epochs and validation

    Args:
        rank: Device rank (GPU id or 'cpu')
        world_size: Number of devices being used
        model: The neural network model
        optimizer: The optimizer
        train_loader: Training data loader
        valid_loader: Validation data loader
        num_epochs: Total number of epochs to train
        patience: Early stopping patience
        outdir: Output directory for logs and checkpoints
        trainable: Which model parts to train ("all" or list of layer names)
        dtype: Torch dtype for computations
        start_epoch: Epoch to start/resume from
        lr_schedule: Learning rate scheduler
        use_ray: Whether using Ray for distributed training
        checkpoint_freq: How often to save checkpoints
        comet_experiment: Comet.ml experiment object
        comet_step_freq: How often to log to comet
        val_freq: How often to run validation
        save_attention: Whether to save attention weights
        checkpoint_dir: Directory to save checkpoints
    """

    # run per-worker setup here so all processes / threads get configured.
    # Ignore divide by 0 errors
    np.seterr(divide="ignore", invalid="ignore")
    # disable GUI
    import matplotlib

    matplotlib.use("agg")

    # Setup tensorboard writers
    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
        tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")
    else:
        tensorboard_writer_train = None
        tensorboard_writer_valid = None

    device_type = "cuda" if isinstance(rank, int) else "cpu"
    t0_initial = time.time()

    # Early stopping setup
    stale_epochs = torch.tensor(0, device=rank)
    best_val_loss = float("inf")

    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        # Training epoch
        losses_train = train_epoch(
            rank=rank,
            world_size=world_size,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            lr_schedule=lr_schedule,
            epoch=epoch,
            tensorboard_writer=tensorboard_writer_train,
            comet_experiment=comet_experiment,
            comet_step_freq=comet_step_freq,
            checkpoint_dir=checkpoint_dir,
            device_type=device_type,
            dtype=dtype,
            scaler=scaler,
        )
        train_time = time.time() - epoch_start_time

        # Validation epoch
        losses_valid = eval_epoch(
            rank=rank,
            world_size=world_size,
            model=model,
            valid_loader=valid_loader,
            epoch=epoch,
            tensorboard_writer=tensorboard_writer_valid,
            comet_experiment=comet_experiment,
            save_attention=save_attention,
            outdir=outdir,
            device_type=device_type,
            dtype=dtype,
        )
        valid_time = time.time() - train_time - epoch_start_time
        total_time = time.time() - epoch_start_time

        if lr_schedule:
            # ReduceLROnPlateau scheduler should only be updated after each full epoch
            # Other schedulers are updated after each step inside the optimizer_step() function
            if isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_schedule.step(losses_valid["Total"])

        # Log metrics
        if comet_experiment:
            comet_experiment.log_metrics(losses_train, prefix="epoch_train_loss", epoch=epoch)
            comet_experiment.log_metrics(losses_valid, prefix="epoch_valid_loss", epoch=epoch)
            comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), epoch=epoch)
            comet_experiment.log_epoch_end(epoch)

        # Handle checkpointing and logging on rank 0
        if (rank == 0) or (rank == "cpu"):
            # Log learning rate
            tensorboard_writer_train.add_scalar("epoch/learning_rate", lr_schedule.get_last_lr()[0], epoch)

            # Prepare checkpoint state
            extra_state = {"epoch": epoch, "lr_schedule_state_dict": lr_schedule.state_dict()}

            # Save best model if validation loss improved
            if losses_valid["Total"] < best_val_loss:
                best_val_loss = losses_valid["Total"]
                stale_epochs = 0
                save_checkpoint(f"{checkpoint_dir}/best_weights.pth", model, optimizer, extra_state)
            else:
                stale_epochs += 1

            # Periodic epoch checkpointing
            if checkpoint_freq and (epoch % checkpoint_freq == 0):
                checkpoint_path = f"{checkpoint_dir}/checkpoint-{epoch:02d}-{losses_valid['Total']:.6f}.pth"
                save_checkpoint(checkpoint_path, model, optimizer, extra_state)

            # Update loss history
            for loss in losses_train:
                tensorboard_writer_train.add_scalar(f"epoch/loss_{loss}", losses_train[loss], epoch)
                tensorboard_writer_valid.add_scalar(f"epoch/loss_{loss}", losses_valid[loss], epoch)

            # Save epoch stats to JSON
            history_path = Path(outdir) / "history"
            history_path.mkdir(parents=True, exist_ok=True)
            stats = {
                "train": losses_train,
                "valid": losses_valid,
                "epoch_train_time": train_time,
                "epoch_valid_time": valid_time,
                "epoch_total_time": total_time,
            }
            with open(f"{history_path}/epoch_{epoch}.json", "w") as f:
                json.dump(stats, f)

            # Calculate and log ETA
            epochs_remaining = num_epochs - epoch
            time_per_epoch = (time.time() - t0_initial) / epoch
            eta = epochs_remaining * time_per_epoch / 60

            # Log epoch summary
            _logger.info(
                f"Rank {rank}: epoch={epoch}/{num_epochs} "
                f"train_loss={losses_train['Total']:.4f} "
                f"valid_loss={losses_valid['Total']:.4f} "
                f"stale={stale_epochs} "
                f"epoch_train_time={train_time/60:.2f}m "
                f"epoch_valid_time={valid_time/60:.2f}m "
                f"epoch_total_time={total_time/60:.2f}m "
                f"eta={eta:.1f}m",
                color="bold",
            )

            # Flush tensorboard
            tensorboard_writer_train.flush()
            tensorboard_writer_valid.flush()

        # evaluate the model at this epoch on test datasets, make plots, track metrics
        testdir_name = f"_epoch_{epoch}"
        for sample in config["enabled_test_datasets"]:
            run_test(rank, world_size, config, outdir, model, sample, testdir_name, dtype)
        if (rank == 0) or (rank == "cpu"):  # plot only on rank 0
            for sample in config["enabled_test_datasets"]:
                plot_metrics = make_plots(outdir, sample, config["dataset"], testdir_name, config["ntest"])

                # track the following jet metrics in tensorboard
                for k in ["med", "iqr", "match_frac"]:
                    tensorboard_writer_valid.add_scalar(
                        "epoch/{}/jet_ratio/jet_ratio_target_to_pred_pt/{}".format(sample, k),
                        plot_metrics["jet_ratio"]["jet_ratio_target_to_pred_pt"][k],
                        epoch,
                    )
                    if comet_experiment:
                        comet_experiment.log_metric(
                            "epoch/{}/jet_ratio/jet_ratio_target_to_pred_pt/{}".format(sample, k),
                            plot_metrics["jet_ratio"]["jet_ratio_target_to_pred_pt"][k],
                            epoch=epoch,
                        )
                    # Add jet metrics entry to the JSON logging file
                    additional_stats = {
                        "epoch/{}/jet_ratio/jet_ratio_target_to_pred_pt/{}".format(sample, k): plot_metrics["jet_ratio"][
                            "jet_ratio_target_to_pred_pt"
                        ][k]
                    }
                    with open(f"{history_path}/epoch_{epoch}.json", "r+") as f:
                        data = json.load(f)
                        data.update(additional_stats)
                        f.seek(0)
                        json.dump(data, f)
                        f.truncate()

        # Ray training specific logging
        if use_ray:
            import ray

            metrics = {
                "loss": losses_train["Total"],
                "val_loss": losses_valid["Total"],
                "epoch": epoch,
                **{f"train_{k}": v for k, v in losses_train.items()},
                **{f"valid_{k}": v for k, v in losses_valid.items()},
            }

            if (rank == 0) or (rank == "cpu"):
                with TemporaryDirectory() as temp_checkpoint_dir:
                    temp_checkpoint_dir = Path(temp_checkpoint_dir)
                    save_checkpoint(temp_checkpoint_dir / "checkpoint.pth", model, optimizer, extra_state)
                    ray.train.report(
                        metrics,
                        checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir) if rank == 0 else None,
                    )
            else:
                ray.train.report(metrics)

        # Check early stopping
        if stale_epochs > patience:
            _logger.info(f"Breaking due to stale epochs: {stale_epochs}")
            break

        # Synchronize processes
        if world_size > 1:
            dist.barrier()
    # End loop over epochs, training completed
    _logger.info(f"Training completed. Total time on device {rank}: {(time.time() - t0_initial)/60:.3f}min")

    # Clean up
    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer_train.close()
        tensorboard_writer_valid.close()


def run_test(rank, world_size, config, outdir, model, sample, testdir_name, dtype):
    batch_size = config["gpu_batch_multiplier"]
    version = config["test_dataset"][sample]["version"]

    split_configs = config["test_dataset"][sample]["splits"]
    _logger.info("split_configs={}".format(split_configs))

    dataset = []

    ntest = None
    if not (config["ntest"] is None):
        ntest = config["ntest"] // len(split_configs)

    for split_config in split_configs:
        ds = PFDataset(config["data_dir"], f"{sample}/{split_config}:{version}", "test", num_samples=ntest).ds
        dataset.append(ds)
    ds = torch.utils.data.ConcatDataset(dataset)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(f"test_dataset: {sample}, {len(ds)}", color="blue")

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds)
    else:
        sampler = torch.utils.data.RandomSampler(ds)

    test_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=Collater(["X", "ytarget", "ytarget_pt_orig", "ytarget_e_orig", "ycand", "genjets", "targetjets"], ["genmet"]),
        sampler=sampler,
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        # pin_memory=use_cuda,
        # pin_memory_device="cuda:{}".format(rank) if use_cuda else "",
    )

    if not osp.isdir(f"{outdir}/preds{testdir_name}/{sample}"):
        if (rank == 0) or (rank == "cpu"):
            os.system(f"mkdir -p {outdir}/preds{testdir_name}/{sample}")

    _logger.info(f"Running predictions on {sample}")
    torch.cuda.empty_cache()

    # FIXME: import this from a central place
    if config["dataset"] == "clic":
        import fastjet

        jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.4, -1.0)
        jet_ptcut = 5
    elif config["dataset"] == "cms":
        import fastjet

        jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
        jet_ptcut = 3
    else:
        raise Exception("not implemented")

    device_type = "cuda" if isinstance(rank, int) else "cpu"
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
        run_predictions(
            world_size,
            rank,
            model,
            test_loader,
            sample,
            outdir,
            jetdef,
            jet_ptcut=jet_ptcut,
            jet_match_dr=0.1,
            dir_name=testdir_name,
        )
    if world_size > 1:
        dist.barrier()  # block until all workers finished executing run_predictions()


def run(rank, world_size, config, outdir, logfile):
    if (rank == 0) or (rank == "cpu"):  # keep writing the logs
        _configLogger("mlpf", filename=logfile)

    use_cuda = rank != "cpu"

    dtype = getattr(torch, config["dtype"])
    _logger.info("configured dtype={} for autocast".format(dtype))

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # (nccl should be faster than gloo)

    start_epoch = 1
    checkpoint_dir = Path(outdir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "input_dim": len(X_FEATURES[config["dataset"]]),
        "num_classes": len(CLASS_LABELS[config["dataset"]]),
        "input_encoding": config["model"]["input_encoding"],
        "pt_mode": config["model"]["pt_mode"],
        "eta_mode": config["model"]["eta_mode"],
        "sin_phi_mode": config["model"]["sin_phi_mode"],
        "cos_phi_mode": config["model"]["cos_phi_mode"],
        "energy_mode": config["model"]["energy_mode"],
        "elemtypes_nonzero": ELEM_TYPES_NONZERO[config["dataset"]],
        "learned_representation_mode": config["model"]["learned_representation_mode"],
        **config["model"][config["conv_type"]],
    }

    # load a pre-trained checkpoint (continue an aborted training or fine-tune)
    if config["load"]:
        model = MLPF(**model_kwargs).to(torch.device(rank))
        optimizer = get_optimizer(model, config)

        checkpoint = torch.load(config["load"], map_location=torch.device(rank))

        if config["start_epoch"] is None:
            start_epoch = int(os.path.basename(config["load"]).split("-")[1]) + 1
        else:
            start_epoch = config["start_epoch"]

        missing_keys, strict = [], True
        for k in model.state_dict().keys():
            shp0 = model.state_dict()[k].shape
            try:
                shp1 = checkpoint["model_state_dict"][k].shape
            except KeyError:
                missing_keys.append(k)
                continue
            if shp0 != shp1:
                raise Exception("shape mismatch in {}, {}!={}".format(k, shp0, shp1))

        if len(missing_keys) > 0:
            _logger.warning(f"The following parameters are missing in the checkpoint file {missing_keys}", color="red")
            if config.get("relaxed_load", True):
                _logger.warning("Optimizer checkpoint will not be loaded", color="bold")
                strict = False
            else:
                _logger.warning("Use option --relaxed-load if you insist to ignore the missing parameters")
                raise KeyError

        if (rank == 0) or (rank == "cpu"):
            _logger.info("Loaded model weights from {}".format(config["load"]), color="bold")

        model, optimizer = load_checkpoint(checkpoint, model, optimizer, strict)

        # if we are rewinding the epoch counter to 1, then we also want to set the learning rate to the initial value
        if start_epoch == 1:
            for g in optimizer.param_groups:
                if g["lr"] != config["lr"]:
                    _logger.info("optimizer param group lr changed {} -> {}".format(g["lr"], config["lr"]))
                    g["lr"] = config["lr"]

    else:  # instantiate a new model in the outdir created
        model = MLPF(**model_kwargs)
        optimizer = get_optimizer(model, config)

    model.to(rank)
    model.compile()
    configure_model_trainable(model, config["model"]["trainable"], True)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    trainable_params, nontrainable_params, table = count_parameters(model)
    print(table)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)
        _logger.info(f"Trainable parameters: {trainable_params}")
        _logger.info(f"Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table.to_string(index=False))

    if config["train"]:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(config, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info("Creating experiment dir {}".format(outdir))
            _logger.info(f"Model directory {outdir}", color="bold")

        if config["comet"]:
            comet_experiment = create_comet_experiment(config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir)
            comet_experiment.set_name(f"rank_{rank}_{Path(outdir).name}")
            comet_experiment.log_parameter("run_id", Path(outdir).name)
            comet_experiment.log_parameter("world_size", world_size)
            comet_experiment.log_parameter("rank", rank)
            comet_experiment.log_parameters(config, prefix="config:")
            comet_experiment.set_model_graph(model)
            comet_experiment.log_parameter(trainable_params, "trainable_params")
            comet_experiment.log_parameter(nontrainable_params, "nontrainable_params")
            comet_experiment.log_parameter(trainable_params + nontrainable_params, "total_trainable_params")
            comet_experiment.log_code("mlpf/model/training.py")
            comet_experiment.log_code("mlpf/model/mlpf.py")
            comet_experiment.log_code("mlpf/model/utils.py")
            comet_experiment.log_code("mlpf/pipeline.py")
            # save overridden config then log to comet
            config_filename = "overridden_config.yaml"
            with open((Path(outdir) / config_filename), "w") as file:
                yaml.dump(config, file)
            comet_experiment.log_code(str(Path(outdir) / config_filename))
        else:
            comet_experiment = None

        loaders = get_interleaved_dataloaders(
            world_size,
            rank,
            config,
            use_cuda,
            use_ray=False,
        )
        steps_per_epoch = len(loaders["train"])
        last_epoch = -1 if start_epoch == 1 else start_epoch - 1
        lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch)

        train_all_epochs(
            rank,
            world_size,
            model,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_epochs"],
            config["patience"],
            outdir,
            config,
            trainable=config["model"]["trainable"],
            dtype=dtype,
            start_epoch=start_epoch,
            lr_schedule=lr_schedule,
            use_ray=False,
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
            val_freq=config["val_freq"],
            save_attention=config["save_attention"],
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint = torch.load(f"{checkpoint_dir}/best_weights.pth", map_location=torch.device(rank))
        model, optimizer = load_checkpoint(checkpoint, model, optimizer)

    if not (config["load"] is None):
        testdir_name = "_" + Path(config["load"]).stem
    else:
        testdir_name = "_best_weights"

    if config["test"]:
        for sample in config["enabled_test_datasets"]:
            run_test(rank, world_size, config, outdir, model, sample, testdir_name, dtype)

    # make plots only on a single machine
    if (rank == 0) or (rank == "cpu"):
        if config["make_plots"]:
            ntest_files = -1
            for sample in config["enabled_test_datasets"]:
                _logger.info(f"Plotting distributions for {sample}")
                make_plots(outdir, sample, config["dataset"], testdir_name, ntest_files)

    if world_size > 1:
        dist.destroy_process_group()


def override_config(config: dict, args):
    """override config dictionary with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if (arg_value is not None) and (arg in config):
            _logger.info("overriding config item {}={} with {} from cmdline".format(arg, config[arg], arg_value))
            config[arg] = arg_value

    if not (args.attention_type is None):
        config["model"]["attention"]["attention_type"] = args.attention_type

    if not (args.num_convs is None):
        for model in ["gnn_lsh", "attention", "attention", "mamba"]:
            config["model"][model]["num_convs"] = args.num_convs

    config["enabled_test_datasets"] = list(config["test_dataset"].keys())
    if len(args.test_datasets) != 0:
        config["enabled_test_datasets"] = args.test_datasets

    config["train"] = args.train
    config["test"] = args.test
    config["make_plots"] = args.make_plots

    if args.start_epoch is not None:
        args.start_epoch = int(args.start_epoch)
    config["start_epoch"] = args.start_epoch

    if config["load"] is None:
        if config["start_epoch"] is None:
            config["start_epoch"] = 1

    return config


# Run either on CPU, single GPU or multi-GPU using pytorch
def device_agnostic_run(config, world_size, outdir):
    if config["train"]:
        logfile = f"{outdir}/train.log"
    else:
        logfile = f"{outdir}/test.log"
    _configLogger("mlpf", filename=logfile)

    if config["gpus"]:
        assert (
            world_size <= torch.cuda.device_count()
        ), f"--gpus is too high (specified {world_size} gpus but only {torch.cuda.device_count()} gpus are available)"

        torch.cuda.empty_cache()
        if world_size > 1:
            _logger.info(f"Will use torch.nn.parallel.DistributedDataParallel() and {world_size} gpus", color="purple")
            for rank in range(world_size):
                _logger.info(torch.cuda.get_device_name(rank), color="purple")

            mp.spawn(
                run,
                args=(world_size, config, outdir, logfile),
                nprocs=world_size,
                join=True,
            )
        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
            run(rank, world_size, config, outdir, logfile)

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")
        run(rank, world_size, config, outdir, logfile)
