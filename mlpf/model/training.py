import os
import os.path as osp
import pickle as pkl
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import tqdm
import yaml
import json
import sklearn
import sklearn.metrics
import numpy as np
import csv

# comet needs to be imported before torch
from comet_ml import OfflineExperiment, Experiment  # noqa: F401, isort:skip

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile, record_function
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


from mlpf.model.inference import make_plots, run_predictions
from mlpf.model.mlpf import set_save_attention
from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import Collater, PFDataset, get_interleaved_dataloaders
from mlpf.model.losses import mlpf_loss
from mlpf.utils import create_comet_experiment
from mlpf.model.plots import validation_plots


def configure_model_trainable(model, trainable, is_training):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise Exception("configure trainability before distributing the model")
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


def one_epoch(
    rank,
    world_size,
    outdir,
    model,
    optimizer,
    train_loader,
    valid_loader,
    trainable,
    is_train=True,
    lr_schedule=None,
    comet_experiment=None,
    comet_step_freq=None,
    epoch=None,
    val_freq=None,
    dtype=torch.float32,
    tensorboard_writer=None,
    save_attention=False,
    checkpoint_dir="",
):
    """
    Performs training over a given epoch. Will run a validation step every N_STEPS and after the last training batch.
    """

    train_or_valid = "train" if is_train else "valid"
    _logger.info(f"Initiating epoch #{epoch} {train_or_valid} run on device rank={rank}", color="red")

    # this one will keep accumulating `train_loss` and then return the average
    epoch_loss = {}

    if is_train:
        data_loader = train_loader
    else:
        data_loader = valid_loader

    # only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(data_loader)
    else:
        iterator = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch} {train_or_valid} loop on rank={rank}")

    device_type = "cuda" if isinstance(rank, int) else "cpu"

    loss_accum = 0.0
    val_freq_time_0 = time.time()

    if not is_train:
        cm_X_target = np.zeros((13, 13))
        cm_X_pred = np.zeros((13, 13))
        cm_id = np.zeros((13, 13))

    for itrain, batch in iterator:
        set_save_attention(model, outdir, False)
        batch = batch.to(rank, non_blocking=True)

        ytarget = unpack_target(batch.ytarget, model)

        num_elems = batch.X[batch.mask].shape[0]
        num_batch = batch.X.shape[0]

        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                ypred_raw = model(batch.X, batch.mask)
            else:
                with torch.no_grad():
                    # save some attention matrices
                    if save_attention and (rank == 0 or rank == "cpu") and itrain == 0:
                        set_save_attention(model, outdir, True)
                    ypred_raw = model(batch.X, batch.mask)

        ypred = unpack_predictions(ypred_raw)

        if not is_train:
            cm_X_target += sklearn.metrics.confusion_matrix(
                batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ytarget["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            cm_X_pred += sklearn.metrics.confusion_matrix(
                batch.X[:, :, 0][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            cm_id += sklearn.metrics.confusion_matrix(
                ytarget["cls_id"][batch.mask].detach().cpu().numpy(), ypred["cls_id"][batch.mask].detach().cpu().numpy(), labels=range(13)
            )
            # save the events of the first validation batch for quick checks
            if (rank == 0 or rank == "cpu") and itrain == 0:
                validation_plots(batch, ypred_raw, ytarget, ypred, tensorboard_writer, epoch, outdir)
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            if is_train:
                loss = mlpf_loss(ytarget, ypred, batch)
                for param in model.parameters():
                    param.grad = None
            else:
                with torch.no_grad():
                    loss = mlpf_loss(ytarget, ypred, batch)

        if is_train:
            loss["Total"].backward()
            loss_accum += loss["Total"].detach().cpu().item()
            optimizer.step()
            if lr_schedule:
                lr_schedule.step()

        for loss_ in loss.keys():
            if loss_ not in epoch_loss:
                epoch_loss[loss_] = 0.0
            epoch_loss[loss_] += loss[loss_].detach()

        if is_train:
            step = (epoch - 1) * len(data_loader) + itrain
            if not (tensorboard_writer is None):
                if step % 100 == 0:
                    tensorboard_writer.add_scalar("step/loss", loss_accum / num_elems, step)
                    tensorboard_writer.add_scalar("step/num_elems", num_elems, step)
                    tensorboard_writer.add_scalar("step/num_batch", num_batch, step)
                    tensorboard_writer.add_scalar("step/learning_rate", lr_schedule.get_last_lr()[0], step)
                    tensorboard_writer.flush()
                    loss_accum = 0.0

                    extra_state = {"step": step, "lr_schedule_state_dict": lr_schedule.state_dict()}
                    save_checkpoint(f"{checkpoint_dir}/step_weights.pth", model, optimizer, extra_state)

            if not (comet_experiment is None) and (itrain % comet_step_freq == 0):
                # this loss is not normalized to batch size
                comet_experiment.log_metrics(loss, prefix=f"{train_or_valid}", step=step)
                comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), step=step)

        if val_freq is not None and is_train:
            if itrain != 0 and itrain % val_freq == 0:
                # time since last intermediate validation run
                val_freq_time = torch.tensor(time.time() - val_freq_time_0, device=rank)
                if world_size > 1:
                    torch.distributed.all_reduce(val_freq_time)
                # compute intermediate training loss
                intermediate_losses_t = {key: epoch_loss[key] for key in epoch_loss}
                for loss_ in epoch_loss:
                    # sum up the losses from all workers and dicide by
                    if world_size > 1:
                        torch.distributed.all_reduce(intermediate_losses_t[loss_])
                    intermediate_losses_t[loss_] = intermediate_losses_t[loss_].cpu().item() / itrain

                # compute intermediate validation loss
                intermediate_losses_v = one_epoch(
                    rank,
                    world_size,
                    outdir,
                    model,
                    optimizer,
                    train_loader,
                    valid_loader,
                    is_train=False,
                    epoch=epoch,
                    dtype=dtype,
                    checkpoint_dir=checkpoint_dir,
                )
                intermediate_metrics = dict(
                    loss=intermediate_losses_t["Total"],
                    reg_pt_loss=intermediate_losses_t["Regression_pt"],
                    reg_eta_loss=intermediate_losses_t["Regression_eta"],
                    reg_sin_phi_loss=intermediate_losses_t["Regression_sin_phi"],
                    reg_cos_phi_loss=intermediate_losses_t["Regression_cos_phi"],
                    reg_energy_loss=intermediate_losses_t["Regression_energy"],
                    cls_loss=intermediate_losses_t["Classification"],
                    cls_binary_loss=intermediate_losses_t["Classification_binary"],
                    val_loss=intermediate_losses_v["Total"],
                    val_reg_pt_loss=intermediate_losses_v["Regression_pt"],
                    val_reg_eta_loss=intermediate_losses_v["Regression_eta"],
                    val_reg_sin_phi_loss=intermediate_losses_v["Regression_sin_phi"],
                    val_reg_cos_phi_loss=intermediate_losses_v["Regression_cos_phi"],
                    val_reg_energy_loss=intermediate_losses_v["Regression_energy"],
                    val_cls_loss=intermediate_losses_v["Classification"],
                    val_cls_binary_loss=intermediate_losses_v["Classification_binary"],
                    inside_epoch=epoch,
                    step=(epoch - 1) * len(data_loader) + itrain,
                    val_freq_time=val_freq_time.cpu().item(),
                )
                val_freq_log = os.path.join(outdir, "val_freq_log.csv")
                if (rank == 0) or (rank == "cpu"):
                    with open(val_freq_log, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=intermediate_metrics.keys())
                        if os.stat(val_freq_log).st_size == 0:  # only write header if file is empty
                            writer.writeheader()
                        writer.writerow(intermediate_metrics)
                if comet_experiment:
                    comet_experiment.log_metrics(intermediate_losses_v, prefix="valid", step=step)
                val_freq_time_0 = time.time()  # reset intermediate validation spacing timer

    if not is_train and comet_experiment:
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_target, title="Element to target", row_label="X", column_label="target", epoch=epoch, file_name="cm_X_target.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_X_pred, title="Element to pred", row_label="X", column_label="pred", epoch=epoch, file_name="cm_X_pred.json"
        )
        comet_experiment.log_confusion_matrix(
            matrix=cm_id, title="Target to pred", row_label="target", column_label="pred", epoch=epoch, file_name="cm_id.json"
        )

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
    if device_type == "cuda":
        torch.cuda.empty_cache()

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
    """
    Will run the training pipeline

    Args:
        rank: 'cpu' or int representing the gpu device id
        model: a pytorch model (may be wrapped by DistributedDataParallel)
        train_loader: a pytorch geometric Dataloader that loads the training data in the form ~ DataBatch(X, ytarget, ycands)
        valid_loader: a pytorch geometric Dataloader that loads the validation data in the form ~ DataBatch(X, ytarget, ycands)
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

    losses_of_interest = [
        "Total",
        "Classification",
        "Classification_binary",
        "Regression_pt",
        "Regression_eta",
        "Regression_sin_phi",
        "Regression_cos_phi",
        "Regression_energy",
    ]

    losses = {}
    losses["train"], losses["valid"] = {}, {}
    for loss in losses_of_interest:
        losses["train"][loss], losses["valid"][loss] = [], []

    stale_epochs, best_val_loss = torch.tensor(0, device=rank), float("inf")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        # training step, edit here to profile a specific epoch
        if epoch == -1:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
                with record_function("model_train"):
                    losses_t = one_epoch(
                        rank,
                        world_size,
                        outdir,
                        model,
                        optimizer,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        trainable=trainable,
                        is_train=True,
                        lr_schedule=lr_schedule,
                        val_freq=val_freq,
                        dtype=dtype,
                        checkpoint_dir=checkpoint_dir,
                    )
            prof.export_chrome_trace("trace.json")
        else:
            losses_t = one_epoch(
                rank,
                world_size,
                outdir,
                model,
                optimizer,
                train_loader=train_loader,
                valid_loader=valid_loader,
                trainable=trainable,
                is_train=True,
                lr_schedule=lr_schedule,
                comet_experiment=comet_experiment,
                comet_step_freq=comet_step_freq,
                epoch=epoch,
                val_freq=val_freq,
                dtype=dtype,
                tensorboard_writer=tensorboard_writer_train,
                checkpoint_dir=checkpoint_dir,
            )
        t_train = time.time()  # epoch time excluding validation

        losses_v = one_epoch(
            rank,
            world_size,
            outdir,
            model,
            optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            trainable=trainable,
            is_train=False,
            lr_schedule=None,
            comet_experiment=comet_experiment,
            comet_step_freq=comet_step_freq,
            epoch=epoch,
            dtype=dtype,
            tensorboard_writer=tensorboard_writer_valid,
            save_attention=save_attention,
            checkpoint_dir=checkpoint_dir,
        )
        t_valid = time.time()

        if comet_experiment:
            comet_experiment.log_metrics(losses_t, prefix="epoch_train_loss", epoch=epoch)
            comet_experiment.log_metrics(losses_v, prefix="epoch_valid_loss", epoch=epoch)
            comet_experiment.log_metric("learning_rate", lr_schedule.get_last_lr(), epoch=epoch)
            comet_experiment.log_epoch_end(epoch)

        if (rank == 0) or (rank == "cpu"):
            tensorboard_writer_train.add_scalar("epoch/learning_rate", lr_schedule.get_last_lr()[0], epoch)
            extra_state = {"epoch": epoch, "lr_schedule_state_dict": lr_schedule.state_dict()}
            if losses_v["Total"] < best_val_loss:
                best_val_loss = losses_v["Total"]
                stale_epochs = 0
                save_checkpoint(f"{checkpoint_dir}/best_weights.pth", model, optimizer, extra_state)
            else:
                stale_epochs += 1

            if checkpoint_freq and (epoch != 0) and (epoch % checkpoint_freq == 0):
                checkpoint_path = "{}/checkpoint-{:02d}-{:.6f}.pth".format(checkpoint_dir, epoch, losses_v["Total"])
                save_checkpoint(checkpoint_path, model, optimizer, extra_state)

        if use_ray:
            import ray
            from ray.train import Checkpoint

            # save model, optimizer and epoch number for HPO-supported checkpointing
            # Ray automatically syncs the checkpoint to persistent storage
            metrics = dict(
                loss=losses_t["Total"],
                reg_pt_loss=losses_t["Regression_pt"],
                reg_eta_loss=losses_t["Regression_eta"],
                reg_sin_phi_loss=losses_t["Regression_sin_phi"],
                reg_cos_phi_loss=losses_t["Regression_cos_phi"],
                reg_energy_loss=losses_t["Regression_energy"],
                cls_loss=losses_t["Classification"],
                cls_binary_loss=losses_t["Classification_binary"],
                val_loss=losses_v["Total"],
                val_reg_pt_loss=losses_v["Regression_pt"],
                val_reg_eta_loss=losses_v["Regression_eta"],
                val_reg_sin_phi_loss=losses_v["Regression_sin_phi"],
                val_reg_cos_phi_loss=losses_v["Regression_cos_phi"],
                val_reg_energy_loss=losses_v["Regression_energy"],
                val_cls_loss=losses_v["Classification"],
                val_cls_binary_loss=losses_v["Classification_binary"],
                epoch=epoch,
            )
            if (rank == 0) or (rank == "cpu"):
                # only save checkpoint on first worker
                with TemporaryDirectory() as temp_checkpoint_dir:
                    temp_checkpoint_dir = Path(temp_checkpoint_dir)
                    save_checkpoint(temp_checkpoint_dir / "checkpoint.pth", model, optimizer, extra_state)

                    # report metrics and checkpoint to Ray
                    ray.train.report(
                        metrics,
                        checkpoint=Checkpoint.from_directory(temp_checkpoint_dir) if rank == 0 else None,
                    )
            else:
                # ray requires all workers to report metrics
                ray.train.report(metrics)

        if stale_epochs > patience:
            logging.info(f"breaking due to stale epochs: {stale_epochs}")
            break

        if (rank == 0) or (rank == "cpu"):
            for k, v in losses_t.items():
                tensorboard_writer_train.add_scalar("epoch/loss_" + k, v, epoch)

            for loss in losses_of_interest:
                losses["train"][loss].append(losses_t[loss])
                losses["valid"][loss].append(losses_v[loss])

            for k, v in losses_v.items():
                tensorboard_writer_valid.add_scalar("epoch/loss_" + k, v, epoch)

            with open(f"{outdir}/mlpf_losses.pkl", "wb") as f:
                pkl.dump(losses, f)

            t1 = time.time()

            epochs_remaining = num_epochs - epoch
            time_per_epoch = (t1 - t0_initial) / epoch
            eta = epochs_remaining * time_per_epoch / 60

            _logger.info(
                f"Rank {rank}: epoch={epoch} / {num_epochs} "
                + f"train_loss={losses_t['Total']:.4f} "
                + f"valid_loss={losses_v['Total']:.4f} "
                + f"stale={stale_epochs} "
                + f"epoch_train_time={round((t_train-t0)/60, 2)}m "
                + f"epoch_valid_time={round((t_valid-t_train)/60, 2)}m "
                + f"epoch_total_time={round((t1-t0)/60, 2)}m "
                + f"eta={round(eta, 1)}m",
                color="bold",
            )

            # save separate json files with stats for each epoch, this is robust to crashed-then-resumed trainings
            history_path = Path(outdir) / "history"
            history_path.mkdir(parents=True, exist_ok=True)
            with open("{}/epoch_{}.json".format(str(history_path), epoch), "w") as fi:
                stats = {"train": losses_t, "valid": losses_v}
                stats["epoch_train_time"] = t_train - t0
                stats["epoch_valid_time"] = t_valid - t_train
                stats["epoch_total_time"] = t1 - t0
                json.dump(stats, fi)

            if tensorboard_writer_train:
                tensorboard_writer_train.flush()
            if tensorboard_writer_valid:
                tensorboard_writer_valid.flush()

    if world_size > 1:
        dist.barrier()

    _logger.info(f"Done with training. Total training time on device {rank} is {round((time.time() - t0_initial)/60,3)}min")


def run(rank, world_size, config, args, outdir, logfile):
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

    if config["load"]:  # load a pre-trained model
        with open(f"{outdir}/model_kwargs.pkl", "rb") as f:
            model_kwargs = pkl.load(f)
        _logger.info("model_kwargs: {}".format(model_kwargs))

        if config["conv_type"] == "attention":
            model_kwargs["attention_type"] = config["model"]["attention"]["attention_type"]

        model = MLPF(**model_kwargs).to(torch.device(rank))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

        checkpoint = torch.load(config["load"], map_location=torch.device(rank))

        # check if we reached the first epoch in the checkpoint
        if "epoch" in checkpoint["extra_state"]:
            start_epoch = checkpoint["extra_state"]["epoch"] + 1

        for k in model.state_dict().keys():
            shp0 = model.state_dict()[k].shape
            shp1 = checkpoint["model_state_dict"][k].shape
            if shp0 != shp1:
                raise Exception("shape mismatch in {}, {}!={}".format(k, shp0, shp1))

        if (rank == 0) or (rank == "cpu"):
            _logger.info("Loaded model weights from {}".format(config["load"]), color="bold")

        model, optimizer = load_checkpoint(checkpoint, model, optimizer)
    else:  # instantiate a new model in the outdir created
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
        model = MLPF(**model_kwargs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    model.to(rank)
    configure_model_trainable(model, config["model"]["trainable"], True)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    trainable_params, nontrainable_params, table = count_parameters(model)

    if (rank == 0) or (rank == "cpu"):
        _logger.info(model)
        _logger.info(f"Trainable parameters: {trainable_params}")
        _logger.info(f"Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table.to_string(index=False))

    if args.train:
        if (rank == 0) or (rank == "cpu"):
            save_HPs(args, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
            _logger.info("Creating experiment dir {}".format(outdir))
            _logger.info(f"Model directory {outdir}", color="bold")

        if args.comet:
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

    if args.test:
        for sample in args.test_datasets:
            batch_size = config["gpu_batch_multiplier"]
            version = config["test_dataset"][sample]["version"]

            split_configs = config["test_dataset"][sample]["splits"]
            print("split_configs", split_configs)

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

    if (rank == 0) or (rank == "cpu"):  # make plots only on a single machine
        if args.make_plots:

            ntest_files = -1
            # ntest_files = 10000
            for sample in args.test_datasets:
                _logger.info(f"Plotting distributions for {sample}")
                make_plots(outdir, sample, config["dataset"], testdir_name, ntest_files)

    if world_size > 1:
        dist.destroy_process_group()


def override_config(config, args):
    """override config with values from argparse Namespace"""
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            config[arg] = arg_value

    if not (args.attention_type is None):
        config["model"]["attention"]["attention_type"] = args.attention_type

    if not (args.num_convs is None):
        for model in ["gnn_lsh", "attention", "attention", "mamba"]:
            config["model"][model]["num_convs"] = args.num_convs

    if len(args.test_datasets) == 0:
        args.test_datasets = config["test_dataset"]

    return config


def device_agnostic_run(config, args, world_size, outdir):
    if args.train:
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
                args=(world_size, config, args, outdir, logfile),
                nprocs=world_size,
                join=True,
            )
        elif world_size == 1:
            rank = 0
            _logger.info(f"Will use single-gpu: {torch.cuda.get_device_name(rank)}", color="purple")
            run(rank, world_size, config, args, outdir, logfile)

    else:
        rank = "cpu"
        _logger.info("Will use cpu", color="purple")
        run(rank, world_size, config, args, outdir, logfile)
