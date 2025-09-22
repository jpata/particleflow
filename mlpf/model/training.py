"""
This script defines the training, validation, and testing workflow for the MLPF model.

The main entry point is `device_agnostic_run`, which handles CPU, single-GPU, and multi-GPU (DDP) training configurations.

The overall training flow is as follows:
1. `device_agnostic_run`: Sets up the environment and launches the `run` function for each process/GPU.
2. `run`: Initializes the process group, model, optimizer, data loaders, and learning rate scheduler. It then calls `train_all_steps` to begin the main training loop.
3. `train_all_steps`: The core training loop that iterates over training steps.
    - For each step, it calls `train_step` to perform a forward and backward pass.
    - Manages checkpointing, learning rate scheduling, and early stopping.
    - Periodically, it calls `_run_validation_cycle` to evaluate the model.
4. `_run_validation_cycle`:
    - Calls `evaluate` to compute validation loss.
    - Calls `run_test` to generate predictions on the test datasets.
    - Calls `make_plots` (from inference.py) to create performance plots.

Key Functions:
- `device_agnostic_run`: Entry point that manages device configuration (CPU/GPU/multi-GPU).
- `run`: Main function for a single process, responsible for setup and starting the training loop.
- `train_all_steps`: Orchestrates the entire training process over all steps.
- `train_step`: Executes a single training step (forward pass, loss calculation, backward pass, optimizer step).
- `evaluate`: Computes model performance and loss on the validation dataset.
- `model_step`: Performs a single forward pass through the model and computes the loss.
- `optimizer_step`: Executes the backward pass and updates model weights.
- `_log_and_checkpoint_step`: Handles logging and periodic checkpoint saving.
- `_run_validation_cycle`: Coordinates the validation, testing, and plotting process at regular intervals.
- `run_test`: Runs inference on a specified test dataset.
- `get_optimizer`: Utility to create an optimizer based on the configuration.
- `configure_model_trainable`: Utility to set specific model layers as trainable.
- `override_config`: Merges command-line arguments into the configuration dictionary.
"""

import os
import os.path as osp
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import tqdm
import yaml
import json
import numpy as np
from typing import Union
import sys
from packaging.version import Version

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from mlpf.optimizers import get_optimizer
from mlpf.logger import _logger, _configLogger, log_smi, log_memory
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
    load_lr_schedule,
)
from mlpf.model.monitoring import log_step_to_tensorboard, log_dataloader_to_tensorboard
from mlpf.model.inference import make_plots, run_predictions
from mlpf.model.mlpf import MLPF, configure_model_trainable
from mlpf.model.PFDataset import Collater, PFDataset, get_interleaved_dataloaders
from mlpf.model.losses import mlpf_loss
from mlpf.utils import create_comet_experiment


def model_step(batch, model, loss_fn):
    _logger.debug(f"model_step X={batch.X.shape}")
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
    _logger.debug(f"optimizer_step scale={scaler.get_scale():.2E}")
    scaler.scale(loss_opt).backward()
    scaler.step(optimizer)
    scaler.update()
    if lr_schedule:
        # ReduceLROnPlateau scheduler should only be updated after each validation step
        if not isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_schedule.step()


def train_step(
    rank: Union[int, str],
    world_size: int,
    model: MLPF,
    optimizer,
    batch,
    lr_schedule,
    step: int,
    tensorboard_writer=None,
    comet_experiment=None,
    comet_step_freq=None,
    checkpoint_dir="",
    device_type="cuda",
    dtype=torch.float32,
    scaler=None,
    loader_state_dict={},
):
    """Run one training step

    Args:
        rank: Device rank (GPU id or 'cpu')
        world_size: Number of devices being used
        model: The neural network model
        optimizer: The optimizer
        batch: Training batch
        lr_schedule: Learning rate scheduler
        step: Current step number
        tensorboard_writer: TensorBoard writer object
        comet_experiment: Comet.ml experiment object
        comet_step_freq: How often to log to comet
        checkpoint_dir: Directory to save checkpoints
        device_type: 'cuda' or 'cpu'
        dtype: Torch dtype for computations

    Returns:
        dict: Dictionary of step losses
    """
    if world_size > 1:
        dist.barrier()

    model.train()
    step_loss = {}

    batch = batch.to(rank, non_blocking=True)

    with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
        loss_opt, loss, _, _, _ = model_step(batch, model, mlpf_loss)

    optimizer_step(model, loss_opt, optimizer, lr_schedule, scaler)

    # Accumulate losses
    for loss_name in loss:
        if loss_name not in step_loss:
            step_loss[loss_name] = 0.0
        step_loss[loss_name] += loss[loss_name]

    # Log step metrics
    if tensorboard_writer is not None:
        # log_open_files_to_tensorboard(tensorboard_writer, step)
        log_step_to_tensorboard(batch, loss["Total"], lr_schedule, tensorboard_writer, step)
        log_dataloader_to_tensorboard(loader_state_dict, tensorboard_writer, step)
        tensorboard_writer.flush()

    if comet_experiment is not None and (step % comet_step_freq == 0):
        comet_experiment.log_metrics(loss, prefix="train", step=step)
        comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), step=step)

    # Average losses across steps
    num_steps = torch.tensor(1.0, device=rank, dtype=torch.float32)
    if world_size > 1:
        torch.distributed.all_reduce(num_steps)

    for loss_name in step_loss:
        _logger.debug(f"train_step {loss_name}={step_loss[loss_name]}")
        if world_size > 1:
            torch.distributed.all_reduce(step_loss[loss_name])
        step_loss[loss_name] = step_loss[loss_name].cpu().item() / num_steps.cpu().item()

    if world_size > 1:
        dist.barrier()
    _logger.debug(f"train_step reduced {step_loss}")

    return step_loss


def evaluate(
    rank: Union[int, str],
    world_size: int,
    model: MLPF,
    valid_loader,
    step: int,
    tensorboard_writer=None,
    comet_experiment=None,
    outdir=None,
    device_type="cuda",
    dtype=torch.float32,
    make_plots=False,
):
    """Run one evaluation step

    Args:
        rank: Device rank (GPU id or 'cpu')
        world_size: Number of devices being used
        model: The neural network model
        valid_loader: Validation data loader
        step: Current step number
        tensorboard_writer: TensorBoard writer object
        comet_experiment: Comet.ml experiment object
        outdir: Output directory path
        device_type: 'cuda' or 'cpu'
        dtype: Torch dtype for computations

    Returns:
        dict: Dictionary of evaluation losses
    """

    if world_size > 1:
        dist.barrier()

    model.eval()
    eval_loss = {}

    # Only show progress bar on rank 0
    is_interactive = ((world_size <= 1) or (rank == 0)) and sys.stdout.isatty()
    assert len(valid_loader) > 0
    iterator = enumerate(valid_loader)
    if is_interactive:
        iterator = tqdm.tqdm(iterator, total=len(valid_loader), desc=f"Step {step} eval loop on rank={rank}")

    for ival, batch in iterator:
        batch = batch.to(rank, non_blocking=True)

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            with torch.no_grad():
                _, loss, ypred_raw, ypred, ytarget = model_step(batch, model, mlpf_loss)

        # Save validation plots for first batch
        # if (rank == 0 or rank == "cpu") and ival == 0 and make_plots:
        #     validation_plots(batch, ypred_raw, ytarget, ypred, tensorboard_writer, step, outdir)

        # Accumulate losses
        for loss_name in loss:
            if loss_name not in eval_loss:
                eval_loss[loss_name] = 0.0
            eval_loss[loss_name] += loss[loss_name]

    # Average losses across steps
    num_steps = torch.tensor(float(len(valid_loader)), device=rank, dtype=torch.float32)
    if world_size > 1:
        torch.distributed.all_reduce(num_steps)

    for loss_name in eval_loss:
        if world_size > 1:
            torch.distributed.all_reduce(eval_loss[loss_name])
        eval_loss[loss_name] = eval_loss[loss_name].cpu().item() / num_steps.cpu().item()

    if world_size > 1:
        dist.barrier()

    return eval_loss


def _log_and_checkpoint_step(
    rank,
    world_size,
    step,
    model,
    optimizer,
    lr_schedule,
    losses_train,
    tensorboard_writer_train,
    comet_experiment,
    checkpoint_freq,
    checkpoint_dir,
    train_loader,
    valid_loader,
    train_sampler,
    valid_sampler,
    num_patience,
):
    """Helper function to log training information and save periodic checkpoints."""

    # Log training losses to TensorBoard and CometML on the main process
    if (rank == 0) or (rank == "cpu"):
        # Log training losses
        for loss, value in losses_train.items():
            tensorboard_writer_train.add_scalar(f"step/loss_{loss}", value, step)

        tensorboard_writer_train.flush()

    if comet_experiment:
        comet_experiment.log_metrics(losses_train, prefix="step_train_loss", step=step)

    # Save a periodic checkpoint
    if checkpoint_freq and (step % checkpoint_freq == 0):
        if (rank == 0) or (rank == "cpu"):
            extra_state = {
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_schedule_state_dict": lr_schedule.state_dict(),
                "train_loader_state_dict": train_loader.state_dict(),
                "valid_loader_state_dict": valid_loader.state_dict(),
            }

            checkpoint_path = f"{checkpoint_dir}/checkpoint-{step:02d}.pth"
            save_checkpoint(checkpoint_path, model, optimizer, extra_state)

            # Clean up old checkpoints, keeping the last num_patience
            checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint-*.pth"), key=os.path.getmtime)
            for i in range(len(checkpoints) - num_patience):
                _logger.info("removing old checkpoint {}".format(checkpoints[i]))
                os.remove(checkpoints[i])


def _run_validation_cycle(
    rank,
    world_size,
    model,
    optimizer,
    lr_schedule,
    valid_loader,
    step,
    num_steps,
    losses_train,
    best_val_loss,
    stale_steps,
    outdir,
    config,
    device_type,
    dtype,
    tensorboard_writer_valid,
    comet_experiment,
    checkpoint_dir,
    train_time,
    t0_initial,
    use_ray,
    train_loader,
    valid_sampler,
    train_sampler,
):
    """Run the validation, testing, and plotting cycle."""

    _logger.info(f"Running validation on rank{rank}")
    valid_loader.reset()

    # Run validation
    log_memory("evaluate_start", rank, tensorboard_writer_valid, step)
    losses_valid = evaluate(
        rank=rank,
        world_size=world_size,
        model=model,
        valid_loader=valid_loader,
        step=step,
        tensorboard_writer=tensorboard_writer_valid,
        comet_experiment=comet_experiment,
        outdir=outdir,
        device_type=device_type,
        dtype=dtype,
        make_plots=config["make_plots"],
    )
    log_memory("evaluate_end", rank, tensorboard_writer_valid, step)
    valid_time = time.time() - train_time - t0_initial
    total_time = time.time() - t0_initial

    # Update learning rate scheduler that depends on validation loss
    if lr_schedule and isinstance(lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_schedule.step(losses_valid["Total"])

    # Log validation metrics to CometML
    if comet_experiment:
        comet_experiment.log_metrics(losses_valid, prefix="step_valid_loss", step=step)
        comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), step=step)

    # All subsequent actions are only done on the main process
    if (rank == 0) or (rank == "cpu"):
        # Save the best model checkpoint if validation loss improves
        if losses_valid["Total"] < best_val_loss:
            best_val_loss = losses_valid["Total"]
            stale_steps = 0
        else:
            stale_steps += 1

        # Log validation losses to TensorBoard
        for loss, value in losses_valid.items():
            tensorboard_writer_valid.add_scalar(f"step/loss_{loss}", value, step)

        # Save step statistics to a JSON file
        history_path = Path(outdir) / "history"
        history_path.mkdir(parents=True, exist_ok=True)
        stats = {
            "train": losses_train,
            "valid": losses_valid,
            "step_train_time": train_time,
            "step_valid_time": valid_time,
            "step_total_time": total_time,
        }
        with open(f"{history_path}/step_{step}.json", "w") as f:
            json.dump(stats, f)

        # Calculate and log ETA
        steps_remaining = num_steps - step
        time_per_step = (time.time() - t0_initial) / step
        eta = steps_remaining * time_per_step / 60

        # Log a summary of the validation step
        _logger.info(
            f"VALIDATION | Step={step}/{num_steps} | "
            f"Train Loss={losses_train['Total']:.4f} | "
            f"Valid Loss={losses_valid['Total']:.4f} | "
            f"Stale={stale_steps} | "
            f"ETA={eta:.1f}m"
        )

        tensorboard_writer_valid.flush()

    # Run inference and plotting on test datasets for this step
    testdir_name = f"_step_{step}"
    log_memory("run_test_start", rank, tensorboard_writer_valid, step)
    for sample in config["enabled_test_datasets"]:
        run_test(rank, world_size, config, outdir, model, sample, testdir_name, dtype)
    log_memory("run_test_end", rank, tensorboard_writer_valid, step)

    plot_metrics_sample = {}
    if (rank == 0) or (rank == "cpu"):
        log_memory("make_plots_start", rank, tensorboard_writer_valid, step)
        for sample in config["enabled_test_datasets"]:
            plot_metrics = make_plots(outdir, sample, config["dataset"], testdir_name, config["ntest"])
            plot_metrics_sample[sample] = plot_metrics
            # Log key jet metrics to TensorBoard and CometML
            for k in ["med", "iqr", "match_frac"]:
                metric_name = f"step/{sample}/jet_ratio/jet_ratio_target_to_pred_pt/{k}"
                metric_value = plot_metrics["jet_ratio"]["jet_ratio_target_to_pred_pt"][k]
                tensorboard_writer_valid.add_scalar(metric_name, metric_value, step)
                if comet_experiment:
                    comet_experiment.log_metric(metric_name, metric_value, step=step)
                # Add jet metrics to the JSON log file
                with open(f"{history_path}/step_{step}.json", "r+") as f:
                    data = json.load(f)
                    data.update({metric_name: metric_value})
                    f.seek(0)
                    json.dump(data, f)
                    f.truncate()
        log_memory("make_plots_end", rank, tensorboard_writer_valid, step)

    # Ray-specific reporting and checkpointing
    if use_ray:
        import ray

        metrics = {
            "loss": losses_train["Total"],
            "val_loss": losses_valid["Total"],
            "step": step,
            **{f"train_{k}": v for k, v in losses_train.items()},
            **{f"valid_{k}": v for k, v in losses_valid.items()},
        }
        if (rank == 0) or (rank == "cpu"):
            with TemporaryDirectory() as temp_checkpoint_dir:
                extra_state = {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_schedule_state_dict": lr_schedule.state_dict(),
                    "train_loader_state_dict": train_loader.state_dict(),
                    "valid_loader_state_dict": valid_loader.state_dict(),
                }
                for sample in plot_metrics_sample.keys():
                    for metric in ["iqr", "match_frac"]:
                        metric_name = f"step/{sample}/jet_ratio/jet_ratio_target_to_pred_pt/{metric}"
                        metrics[metric_name] = plot_metrics_sample[sample]["jet_ratio"]["jet_ratio_target_to_pred_pt"][metric]
                    metrics[f"step/{sample}/jet_ratio/jet_ratio_target_to_pred_pt/combined"] = (
                        metrics[f"step/{sample}/jet_ratio/jet_ratio_target_to_pred_pt/iqr"]
                        - metrics[f"step/{sample}/jet_ratio/jet_ratio_target_to_pred_pt/match_frac"]
                    )
                save_checkpoint(Path(temp_checkpoint_dir) / "checkpoint.pth", model, optimizer, extra_state)
                ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir))
        else:
            ray.train.report(metrics)

    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()

    return best_val_loss, stale_steps


def train_all_steps(
    rank,
    world_size,
    model,
    optimizer,
    train_loader,
    valid_loader,
    num_steps,
    patience,
    outdir,
    config,
    trainable="all",
    dtype=torch.float32,
    start_step=1,
    lr_schedule=None,
    use_ray=False,
    checkpoint_freq=None,
    comet_experiment=None,
    comet_step_freq=None,
    val_freq=None,
    checkpoint_dir: str = "",
    train_sampler=None,
    valid_sampler=None,
):
    """Main training loop that handles all steps and validation"""
    _logger.info(f"Starting training from step {start_step} to {num_steps} on rank {rank} of {world_size}")
    assert len(train_loader) > 0

    # Per-worker setup
    np.seterr(divide="ignore", invalid="ignore")

    # Setup TensorBoard writers on the main process
    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer_train = SummaryWriter(f"{outdir}/runs/train")
        tensorboard_writer_valid = SummaryWriter(f"{outdir}/runs/valid")
    else:
        tensorboard_writer_train = None
        tensorboard_writer_valid = None

    device_type = "cuda" if isinstance(rank, int) else "cpu"
    t0_initial = time.time()

    # Early stopping setup
    stale_steps = 0
    best_val_loss = float("inf")

    scaler = torch.amp.GradScaler()
    train_iterator = iter(train_loader)

    # Use tqdm for progress bar only on the main process in an interactive session
    is_interactive = ((world_size <= 1) or (rank == 0)) and sys.stdout.isatty()
    iterator = range(start_step, num_steps + 1)
    if is_interactive:
        iterator = tqdm.tqdm(iterator, initial=start_step, total=num_steps, desc=f"Training on rank={rank}")

    # loop over the dataset
    for step in iterator:
        step_start_time = time.time()

        # Get next training batch
        batch = next(train_iterator)
        _logger.debug(f"rank={rank} batch={batch.X.shape}")

        # Run a single training step
        log_memory("train_step_start", rank, tensorboard_writer_train, step)
        losses_train = train_step(
            rank=rank,
            world_size=world_size,
            model=model,
            optimizer=optimizer,
            batch=batch,
            lr_schedule=lr_schedule,
            step=step,
            tensorboard_writer=tensorboard_writer_train,
            comet_experiment=comet_experiment,
            comet_step_freq=comet_step_freq,
            checkpoint_dir=checkpoint_dir,
            device_type=device_type,
            dtype=dtype,
            scaler=scaler,
            loader_state_dict=train_loader.state_dict()["loader_state_dict"],
        )
        log_memory("train_step_end", rank, tensorboard_writer_train, step)
        train_time = time.time() - step_start_time

        # Log a brief training status every 100 steps on the main process
        if (step % 100 == 0):
            # Get the current learning rate, handling the case of multiple parameter groups
            current_lr = lr_schedule.get_last_lr()[0]
            _logger.info(f"Step {step}/{num_steps} rank{rank} | " f"Train Loss: {losses_train['Total']:.4f} | " f"LR: {current_lr:.2e}")

            # check smi status
            log_smi(rank)

        # Synchronize all processes at the end of the step
        if world_size > 1:
            dist.barrier()

        # Log training info and save periodic checkpoint immediately after training
        _log_and_checkpoint_step(
            rank,
            world_size,
            step,
            model,
            optimizer,
            lr_schedule,
            losses_train,
            tensorboard_writer_train,
            comet_experiment,
            checkpoint_freq,
            checkpoint_dir,
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
            config["patience"],
        )

        # Run validation, testing, and plotting cycle at specified frequency, or at the last step
        if (step % val_freq == 0) or (step == num_steps):
            best_val_loss, stale_steps = _run_validation_cycle(
                rank,
                world_size,
                model,
                optimizer,
                lr_schedule,
                valid_loader,
                step,
                num_steps,
                losses_train,
                best_val_loss,
                stale_steps,
                outdir,
                config,
                device_type,
                dtype,
                tensorboard_writer_valid,
                comet_experiment,
                checkpoint_dir,
                train_time,
                t0_initial,
                use_ray,
                train_loader,
                valid_sampler,
                train_sampler,
            )

        # Check for early stopping
        if stale_steps > patience:
            _logger.info(f"Stopping early due to stale steps: {stale_steps} > {patience}")
            break

    # End of training loop
    _logger.info(f"Training completed. Total time on device {rank}: {(time.time() - t0_initial)/60:.3f}min")

    # Clean up TensorBoard writers
    if (rank == 0) or (rank == "cpu"):
        tensorboard_writer_train.close()
        tensorboard_writer_valid.close()


def run_test(rank, world_size, config, outdir, model, sample, testdir_name, dtype):
    batch_size = config["test_dataset"][sample]["batch_size"] * config["gpu_batch_multiplier"]
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

    vals_for_test = ["X", "ytarget", "ytarget_pt_orig", "ytarget_e_orig", "ycand", "genjets", "targetjets"]

    # pythia branch was introduced for cms in version 2.8.0
    if sample.startswith("cms_") and Version(version) >= Version("2.8.0"):
        vals_for_test += ["pythia"]

    test_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=Collater(vals_for_test, ["genmet"]),
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


def run(rank: int | str, world_size: int, config: dict, outdir: str, logfile: str):
    # per-rank log
    _configLogger("mlpf", rank, filename=f"{logfile}.{rank}")

    use_cuda = rank != "cpu"

    dtype = getattr(torch, config["dtype"])
    _logger.info("configured dtype={} for autocast".format(dtype))

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=int(rank), world_size=world_size)  # (nccl should be faster than gloo)

    checkpoint_dir = Path(outdir) / "checkpoints"
    if (rank == 0) | (rank == "cpu"):
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

    start_step = 1
    lr_schedule = None
    checkpoint = None

    # load a pre-trained checkpoint (continue an aborted training or fine-tune)
    if config["load"]:
        model = MLPF(**model_kwargs).to(torch.device(rank))
        optimizer = get_optimizer(model, config)
        lr_schedule = get_lr_schedule(config, optimizer, config["num_steps"])

        checkpoint = torch.load(config["load"], map_location=torch.device(rank))
        start_step = checkpoint["extra_state"]["step"] + 1

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

        _logger.info("Loaded model weights from {}".format(config["load"]), color="bold")
        _logger.info(f"Restoring training from step {start_step}")

        load_lr_schedule(lr_schedule, checkpoint, start_step=start_step)
        model, optimizer = load_checkpoint(checkpoint, model, optimizer, strict)

    else:  # instantiate a new model in the outdir created
        model = MLPF(**model_kwargs)
        optimizer = get_optimizer(model, config)
        lr_schedule = get_lr_schedule(config, optimizer, config["num_steps"])

    model.to(rank)
    # CPU: the compilation does not work with bs>1
    # Nvidia: compilation should generally be used, but can be disabled
    # ROCM: compilation seems to be needed for ROCm to work properly
    if rank != "cpu":
        model.compile()
    configure_model_trainable(model, config["model"]["trainable"], True)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    trainable_params, nontrainable_params, table = count_parameters(model)
    _logger.info(str(table))
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
            if comet_experiment is not None:
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
                # save overridden config, then log to comet
                config_filename = "overridden_config.yaml"
                with open((Path(outdir) / config_filename), "w") as file:
                    yaml.dump(config, file)
                comet_experiment.log_code(str(Path(outdir) / config_filename))
        else:
            comet_experiment = None

        loaders, samplers = get_interleaved_dataloaders(
            world_size,
            rank,
            config,
            use_cuda,
            use_ray=False,
        )

        if config["load"] and checkpoint:
            train_loader = loaders["train"]
            valid_loader = loaders["valid"]
            if "train_loader_state_dict" in checkpoint["extra_state"]:
                train_loader.load_state_dict(checkpoint["extra_state"]["train_loader_state_dict"])
            if "valid_loader_state_dict" in checkpoint["extra_state"]:
                valid_loader.load_state_dict(checkpoint["extra_state"]["valid_loader_state_dict"])

        for split in loaders.keys():
            _logger.info("loader {} rank={} len={}".format(split, rank, len(loaders[split])))

        train_all_steps(
            rank,
            world_size,
            model,
            optimizer,
            loaders["train"],
            loaders["valid"],
            config["num_steps"],
            config["patience"],
            outdir,
            config,
            trainable=config["model"]["trainable"],
            dtype=dtype,
            start_step=start_step,
            lr_schedule=lr_schedule,
            use_ray=False,
            checkpoint_freq=config["checkpoint_freq"],
            comet_experiment=comet_experiment,
            comet_step_freq=config["comet_step_freq"],
            val_freq=config["val_freq"],
            checkpoint_dir=str(checkpoint_dir),
            train_sampler=samplers["train"],
            valid_sampler=samplers["valid"],
        )

    if world_size > 1:
        dist.destroy_process_group()


def override_config(config: dict, args):
    """override config dictionary with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            if arg in config:
                _logger.info("overriding config item {}={} with {} from cmdline".format(arg, config[arg], arg_value))
                config[arg] = arg_value
            else:
                _logger.info("skipping {}".format(arg))

    if "attention_type" in args and args.attention_type is not None:
        config["model"]["attention"]["attention_type"] = args.attention_type

    if "num_convs" in args and args.num_convs is not None:
        for model in ["gnn_lsh", "attention"]:
            config["model"][model]["num_convs"] = args.num_convs

    config["enabled_test_datasets"] = list(config["test_dataset"].keys())
    if "test_datasets" in args:
        if len(args.test_datasets) != 0:
            config["enabled_test_datasets"] = args.test_datasets

    config["train"] = args.train
    config["test"] = args.test
    if "make_plots" in args:
        config["make_plots"] = args.make_plots

    return config


# Run either single GPU or single-node multi-GPU using pytorch DDP
def device_agnostic_run(config, world_size, outdir):
    if config["train"]:
        logfile = f"{outdir}/train.log"
    else:
        logfile = f"{outdir}/test.log"
    _configLogger("mlpf", 0, filename=logfile)

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
