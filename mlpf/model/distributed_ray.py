import torch
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
import os
import yaml
import csv

from mlpf.model.mlpf import MLPF
from mlpf.model.logger import _logger, _configLogger
from mlpf.model.PFDataset import get_interleaved_dataloaders
from mlpf.utils import create_comet_experiment
from mlpf.model.training import train_all_epochs

from mlpf.model.utils import (
    load_checkpoint,
    CLASS_LABELS,
    X_FEATURES,
    ELEM_TYPES_NONZERO,
    save_HPs,
    get_lr_schedule,
    count_parameters,
)


def run_ray_training(config, args, outdir):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    if not args.ray_local:
        ray.init(address="auto")

    _configLogger("mlpf", filename=f"{outdir}/train.log")

    use_gpu = args.gpus > 0
    num_workers = args.gpus if use_gpu else 1
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": max(1, args.ray_cpus // num_workers - 1), "GPU": int(use_gpu)},  # -1 to avoid blocking
    )
    storage_path = Path(args.experiments_dir if args.experiments_dir else "experiments").resolve()
    run_config = ray.train.RunConfig(
        name=Path(outdir).name,
        storage_path=storage_path,
        log_to_file=False,
        failure_config=ray.train.FailureConfig(max_failures=2),
        checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),  # keep only latest checkpoint
        sync_config=ray.train.SyncConfig(sync_artifacts=True),
    )
    trainable = tune.with_parameters(train_ray_trial, args=args, outdir=outdir)
    # Resume from checkpoint if a checkpoint is found in outdir
    if TorchTrainer.can_restore(outdir):
        _logger.info(f"Restoring Ray Trainer from {outdir}", color="bold")
        trainer = TorchTrainer.restore(outdir, train_loop_per_worker=trainable, scaling_config=scaling_config)
    else:
        resume_from_checkpoint = ray.train.Checkpoint(config["load"]) if config["load"] else None
        if resume_from_checkpoint:
            _logger.info("Loading checkpoint {}".format(config["load"]), color="bold")
        trainer = TorchTrainer(
            train_loop_per_worker=trainable,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    result = trainer.fit()

    for loss_key in sorted([k for k in result.metrics.keys() if k.startswith("train_")]):
        _logger.info("Final {}: {}".format(loss_key, result.metrics[loss_key]), color="bold")
        loss_key = loss_key.replace("train_", "valid_")
        _logger.info("Final {}: {}".format(loss_key, result.metrics[loss_key]), color="bold")


def set_searchspace_and_run_trial(search_space, config, args):
    import ray
    from raytune.pt_search_space import set_hps_from_search_space

    rank = ray.train.get_context().get_local_rank()

    config = set_hps_from_search_space(search_space, config)
    try:
        # outdir will be taken from the ray.train.context.TrainContext in each trial
        train_ray_trial(config, args, outdir=None)
    except torch.cuda.OutOfMemoryError:
        ray.train.report({"val_loss": np.NAN})
        torch.cuda.empty_cache()  # make sure GPU memory is cleared for next trial
        if rank == 0:
            logging.warning("OOM error encountered, skipping this hyperparameter configuration.")
            skiplog_file_path = Path(config["raytune"]["local_dir"]) / args.hpo / "skipped_configurations.txt"
            lines = ["{}: {}\n".format(item[0], item[1]) for item in search_space.items()]

            with open(skiplog_file_path, "a") as f:
                f.write("#" * 80 + "\n")
                for line in lines:
                    f.write(line)
                    logging.warning(line[:-1])
                f.write("#" * 80 + "\n\n")
            logging.warning("Done writing warnings to log.")


def run_hpo(config, args):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer

    from raytune.pt_search_space import raytune_num_samples, search_space
    from raytune.utils import get_raytune_schedule, get_raytune_search_alg

    if args.raytune_num_samples:
        raytune_num_samples = args.raytune_num_samples  # noqa: F811

    name = args.hpo  # name of Ray Tune experiment directory

    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"  # don't crash if a metric is missing
    if isinstance(config["raytune"]["local_dir"], type(None)):
        raise TypeError("Please specify a local_dir in the raytune section of the config file.")

    expdir = Path(config["raytune"]["local_dir"]) / name
    expdir.mkdir(parents=True, exist_ok=True)
    dirname = Path(config["raytune"]["local_dir"]) / name
    shutil.copy(
        "mlpf/raytune/pt_search_space.py",
        str(dirname / "pt_search_space.py"),
    )  # Copy the search space definition file to the train dir for later reference
    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    with open((dirname / "config.yaml"), "w") as file:
        yaml.dump(config, file)

    if not args.ray_local:
        _logger.info("Inititalizing ray...")
        ray.init(
            address=os.environ["ip_head"],
            _node_ip_address=os.environ["head_node_ip"],
        )
        _logger.info("Done.")

    sched = get_raytune_schedule(config["raytune"])
    search_alg = get_raytune_search_alg(config["raytune"])

    scaling_config = ray.train.ScalingConfig(
        num_workers=args.gpus,
        use_gpu=True,
        resources_per_worker={"CPU": args.ray_cpus // (args.gpus) - 1, "GPU": 1},  # -1 to avoid blocking
    )

    trainable = tune.with_parameters(set_searchspace_and_run_trial, config=config, args=args)
    trainer = TorchTrainer(train_loop_per_worker=trainable, scaling_config=scaling_config)

    if tune.Tuner.can_restore(str(expdir)):
        # resume unfinished HPO run
        tuner = tune.Tuner.restore(str(expdir), trainable=trainer, resume_errored=True, restart_errored=False, resume_unfinished=True)
    else:
        # start new HPO run
        search_space = {"train_loop_config": search_space}  # the ray TorchTrainer only takes a single arg: train_loop_config
        tuner = tune.Tuner(
            trainer,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=raytune_num_samples,
                metric=config["raytune"]["default_metric"] if (search_alg is None and sched is None) else None,
                mode=config["raytune"]["default_mode"] if (search_alg is None and sched is None) else None,
                search_alg=search_alg,
                scheduler=sched,
            ),
            run_config=ray.train.RunConfig(
                name=name,
                storage_path=config["raytune"]["local_dir"],
                log_to_file=False,
                failure_config=ray.train.FailureConfig(max_failures=2),
                checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),  # keep only latest checkpoint
                sync_config=ray.train.SyncConfig(sync_artifacts=True),
            ),
        )
    start = datetime.now()
    _logger.info("Starting tuner.fit()")
    result_grid = tuner.fit()
    end = datetime.now()

    print("Number of errored trials: {}".format(result_grid.num_errors))
    print("Number of terminated (not errored) trials: {}".format(result_grid.num_terminated))
    print("Ray Tune experiment path: {}".format(result_grid.experiment_path))

    best_result = result_grid.get_best_result(scope="last-10-avg", metric="val_loss", mode="min")
    best_config = best_result.config
    print("Best trial path: {}".format(best_result.path))

    result_df = result_grid.get_dataframe()
    print(result_df)
    print(result_df.columns)

    logging.info("Total time of Tuner.fit(): {}".format(end - start))
    logging.info("Best hyperparameters found according to {} were: {}".format(config["raytune"]["default_metric"], best_config))


def train_ray_trial(config, args, outdir=None):
    import ray

    if outdir is None:
        outdir = ray.train.get_context().get_trial_dir()
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    use_cuda = args.gpus > 0

    rank = ray.train.get_context().get_local_rank() if use_cuda else "cpu"
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

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

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # optimizer should be created after distributing the model to devices with ray.train.torch.prepare_model(model)
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    trainable_params, nontrainable_params, table = count_parameters(model)
    print(table)

    if (rank == 0) or (rank == "cpu"):
        with open(os.path.join(outdir, "num_params.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trainable_params", "nontrainable_params", "total_params"])
            writer.writerow([trainable_params, nontrainable_params, trainable_params + nontrainable_params])
        _logger.info(model)
        _logger.info(f"Trainable parameters: {trainable_params}")
        _logger.info(f"Non-trainable parameters: {nontrainable_params}")
        _logger.info(f"Total parameters: {trainable_params + nontrainable_params}")
        _logger.info(table)

    if (rank == 0) or (rank == "cpu"):
        save_HPs(config, model, model_kwargs, outdir)  # save model_kwargs and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

    loaders = get_interleaved_dataloaders(world_size, rank, config, use_cuda, use_ray=True)

    if args.comet:
        comet_experiment = create_comet_experiment(config["comet_name"], comet_offline=config["comet_offline"], outdir=outdir)
        comet_experiment.set_name(f"world_rank_{world_rank}_{Path(outdir).name}")
        comet_experiment.log_parameter("run_id", Path(outdir).name)
        comet_experiment.log_parameter("world_size", world_size)
        comet_experiment.log_parameter("rank", rank)
        comet_experiment.log_parameter("world_rank", world_rank)
        comet_experiment.log_parameters(config, prefix="config:")
        comet_experiment.set_model_graph(model)
        comet_experiment.log_parameter(trainable_params, "trainable_params")
        comet_experiment.log_parameter(nontrainable_params, "nontrainable_params")
        comet_experiment.log_parameter(trainable_params + nontrainable_params, "total_trainable_params")
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/model/training.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/model/mlpf.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/model/utils.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/pipeline.py"))
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/raytune/pt_search_space.py"))
        # save overridden config then log to comet
        config_filename = "overridden_config.yaml"
        with open((Path(outdir) / config_filename), "w") as file:
            yaml.dump(config, file)
        comet_experiment.log_code(str(Path(outdir) / config_filename))
    else:
        comet_experiment = None

    steps_per_epoch = len(loaders["train"])
    start_epoch = 1
    lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch=-1)

    checkpoint_dir = os.path.join(outdir, "checkpoints")
    checkpoint_dir = Path(outdir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as _checkpoint_dir:
            checkpoint = torch.load(Path(_checkpoint_dir) / "checkpoint.pth", map_location=torch.device(rank))
            model, optimizer = load_checkpoint(checkpoint, model, optimizer)
            start_epoch = checkpoint["extra_state"]["epoch"] + 1
            lr_schedule = get_lr_schedule(config, optimizer, config["num_epochs"], steps_per_epoch, last_epoch=start_epoch - 1)

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
        start_epoch=start_epoch,
        lr_schedule=lr_schedule,
        use_ray=True,
        checkpoint_freq=config["checkpoint_freq"],
        comet_experiment=comet_experiment,
        comet_step_freq=config["comet_step_freq"],
        dtype=getattr(torch, config["dtype"]),
        val_freq=config["val_freq"],
        checkpoint_dir=checkpoint_dir,
    )
