import torch
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import logging
import os
import yaml
import csv

from mlpf.model.mlpf import MLPF
from mlpf.logger import _logger, _configLogger
from mlpf.model.PFDataset import get_interleaved_dataloaders
from mlpf.utils import create_comet_experiment
from mlpf.model.training import train_all_steps, get_optimizer
from mlpf.conf import MLPFConfig

from mlpf.model.utils import (
    load_checkpoint,
    save_HPs,
    get_lr_schedule,
    count_parameters,
    load_lr_schedule,
)


def _init_ray(args):
    import ray

    if ray.is_initialized() or args.ray_local:
        return

    _logger.info("Inititalizing ray...")
    ip_head = os.environ.get("ip_head")
    head_node_ip = os.environ.get("head_node_ip")

    if ip_head and head_node_ip:
        _logger.info("IP: " + head_node_ip)
        ray.init(
            address=ip_head,
            _node_ip_address=head_node_ip,
        )
    else:
        _logger.info("Ray cluster env vars not set, using auto address discovery.")
        ray.init(address="auto")
    _logger.info("Ray initialized.")


def _get_scaling_config(args):
    from ray import train

    use_gpu = args.gpus > 0
    num_workers = args.gpus if use_gpu else 1
    # Reserve 1 CPU per worker for the TrainController actor to avoid resource deadlock
    cpus_per_worker = max(1, args.ray_cpus // num_workers - 1) if args.ray_cpus else 1

    return train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker, "GPU": int(use_gpu)},
    )


def _apply_search_space_overrides(search_overrides, base_config):
    from mlpf.raytune.search_space import set_hps_from_search_space

    return set_hps_from_search_space(search_overrides, deepcopy(base_config))


def _log_skipped_hpo_configuration(expdir, sampled_config):
    skiplog_file_path = Path(expdir) / "skipped_configurations.txt"
    lines = ["{}: {}\n".format(item[0], item[1]) for item in sampled_config.items()]

    with open(skiplog_file_path, "a") as f:
        f.write("#" * 80 + "\n")
        for line in lines:
            f.write(line)
            logging.warning(line.strip())
        f.write("#" * 80 + "\n\n")

    logging.warning("Done writing warnings to log.")


def run_ray_training(config, args, outdir, loglevel=logging.INFO):
    import ray
    from ray import tune
    from ray.train.torch import TorchTrainer  # , TorchConfig

    _init_ray(args)

    _configLogger("mlpf", filename=f"{outdir}/train.log", loglevel=loglevel)

    scaling_config = _get_scaling_config(args)
    storage_path = Path(args.experiments_dir if args.experiments_dir else "experiments").resolve()
    run_config = ray.train.RunConfig(
        name=Path(outdir).name,
        storage_path=storage_path,
        failure_config=ray.train.FailureConfig(max_failures=2),
        checkpoint_config=ray.train.CheckpointConfig(num_to_keep=1),  # keep only latest checkpoint
    )
    trainable = tune.with_parameters(train_ray_trial, args=args, outdir=outdir)

    trainer = TorchTrainer(
        train_loop_per_worker=trainable,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()

    for loss_key in sorted([k for k in result.metrics.keys() if k.startswith("train_")]):
        _logger.info("Final {}: {}".format(loss_key, result.metrics[loss_key]), color="bold")
        loss_key = loss_key.replace("train_", "valid_")
        _logger.info("Final {}: {}".format(loss_key, result.metrics[loss_key]), color="bold")


def run_hpo_train_loop(config, args, outdir, sampled_config, expdir):
    from ray import train

    rank = train.get_context().get_local_rank() if args.gpus > 0 else "cpu"
    try:
        train_ray_trial(config, args, outdir=outdir)
    except torch.cuda.OutOfMemoryError:
        train.report({"loss": np.nan, "val_loss": np.nan, "step": 0, "training_iteration": 0})
        torch.cuda.empty_cache()  # make sure GPU memory is cleared for next trial
        if rank == 0:
            logging.warning("OOM error encountered, skipping this hyperparameter configuration.")
            _log_skipped_hpo_configuration(expdir, sampled_config)


def run_hpo_trial(sampled_config, base_config, args):
    from ray import train, tune
    from ray.train.torch import TorchTrainer
    from ray.tune.integration.ray_train import TuneReportCallback

    trial_context = tune.get_context()
    trial_dir = Path(trial_context.get_trial_dir())
    trial_config = _apply_search_space_overrides(sampled_config, base_config)
    expdir = Path(base_config["raytune"]["local_dir"]) / args.hpo

    trainer = TorchTrainer(
        train_loop_per_worker=tune.with_parameters(
            run_hpo_train_loop,
            args=args,
            outdir=str(trial_dir),
            sampled_config=sampled_config,
            expdir=str(expdir),
        ),
        train_loop_config=trial_config,
        scaling_config=_get_scaling_config(args),
        run_config=train.RunConfig(
            name="train",
            storage_path=str(trial_dir),
            callbacks=[TuneReportCallback()],
            failure_config=train.FailureConfig(max_failures=2),
            checkpoint_config=train.CheckpointConfig(num_to_keep=1),
        ),
    )
    trainer.fit()


def run_hpo(config, args, loglevel=logging.INFO):
    import ray
    from ray import tune

    from mlpf.raytune.search_space import raytune_num_samples, search_space
    from mlpf.raytune.utils import get_raytune_schedule, get_raytune_search_alg

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
        "mlpf/raytune/search_space.py",
        str(dirname / "search_space.py"),
    )  # Copy the search space definition file to the train dir for later reference
    # Save config for later reference. Note that saving happens after parameters are overwritten by cmd line args.
    with open((dirname / "config.yaml"), "w") as file:
        yaml.dump(MLPFConfig.model_validate(config).model_dump(mode="json"), file)

    _init_ray(args)

    sched = get_raytune_schedule(config["raytune"])
    search_alg = get_raytune_search_alg(config["raytune"])

    # With function-based trainables that create their own TorchTrainer inside,
    # we cannot declare GPU resources at the Tune level — Tune would hold them
    # on the trial actor, preventing the inner TorchTrainer workers from
    # acquiring GPUs (deadlock). Instead, declare only 1 CPU for the lightweight
    # trial function and use max_concurrent_trials to prevent over-subscription.
    trainable = tune.with_parameters(run_hpo_trial, base_config=config, args=args)
    if ray.is_initialized():
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
    else:
        total_gpus = torch.cuda.device_count()
    gpus_per_trial = args.gpus if args.gpus > 0 else 1
    max_concurrent_trials = max(1, total_gpus // gpus_per_trial)
    _logger.info(f"HPO: {total_gpus} total GPUs, {gpus_per_trial} per trial, max {max_concurrent_trials} concurrent trials")

    if tune.Tuner.can_restore(str(expdir)):
        # resume unfinished HPO run
        tuner = tune.Tuner.restore(
            str(expdir),
            trainable=trainable,
            param_space=search_space,
            resume_errored=True,
            restart_errored=False,
            resume_unfinished=True,
        )
    else:
        # start new HPO run
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=raytune_num_samples,
                max_concurrent_trials=max_concurrent_trials,
                metric=config["raytune"]["default_metric"] if (search_alg is None and sched is None) else None,
                mode=config["raytune"]["default_mode"] if (search_alg is None and sched is None) else None,
                search_alg=search_alg,
                scheduler=sched,
            ),
            run_config=ray.tune.RunConfig(
                name=name,
                storage_path=config["raytune"]["local_dir"],
                failure_config=ray.tune.FailureConfig(max_failures=2),
            ),
        )
    start = datetime.now()
    _logger.info("Starting tuner.fit()")
    result_grid = tuner.fit()
    end = datetime.now()

    print("Number of errored trials: {}".format(result_grid.num_errors))
    print("Number of terminated (not errored) trials: {}".format(result_grid.num_terminated))
    print("Ray Tune experiment path: {}".format(result_grid.experiment_path))

    best_result = result_grid.get_best_result(
        scope="last-10-avg",
        metric=config["raytune"]["default_metric"],
        mode=config["raytune"]["default_mode"],
    )
    best_config = _apply_search_space_overrides(best_result.config, config)
    print("Best trial path: {}".format(best_result.path))

    result_df = result_grid.get_dataframe()
    print(result_df)
    print(result_df.columns)

    logging.info("Total time of Tuner.fit(): {}".format(end - start))
    logging.info("Best hyperparameters found according to {} were: {}".format(config["raytune"]["default_metric"], best_config))


def train_ray_trial(config, args, outdir=None):
    import ray

    if outdir is None:
        outdir = os.getcwd()

    if outdir is not None:
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

    use_cuda = args.gpus > 0

    rank = ray.train.get_context().get_local_rank() if use_cuda else "cpu"
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    mlpf_config = MLPFConfig.model_validate(config)
    model = MLPF(mlpf_config)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # optimizer should be created after distributing the model to devices with ray.train.torch.prepare_model(model)
    model = ray.train.torch.prepare_model(model)
    optimizer = get_optimizer(model, mlpf_config)

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
        save_HPs(mlpf_config, model, outdir)  # save config and hyperparameters
        _logger.info("Creating experiment dir {}".format(outdir))
        _logger.info(f"Model directory {outdir}", color="bold")

    loaders, samplers = get_interleaved_dataloaders(world_size, rank, mlpf_config, use_cuda, use_ray=True)

    if mlpf_config.comet:
        comet_experiment = create_comet_experiment(mlpf_config.comet_name, comet_offline=mlpf_config.comet_offline, outdir=outdir)
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
        comet_experiment.log_code(str(Path(outdir).parent.parent / "mlpf/raytune/search_space.py"))
        # save overridden config then log to comet
        config_filename = "overridden_config.yaml"
        with open((Path(outdir) / config_filename), "w") as file:
            yaml.dump(mlpf_config.model_dump(mode="json"), file)
        comet_experiment.log_code(str(Path(outdir) / config_filename))
    else:
        comet_experiment = None

    lr_schedule = get_lr_schedule(mlpf_config, optimizer, mlpf_config.num_steps)

    checkpoint_dir = Path(outdir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 1
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as _checkpoint_dir:
            checkpoint_path = Path(_checkpoint_dir) / "checkpoint.pth"
            _logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(rank))
            load_lr_schedule(lr_schedule, checkpoint, start_step=start_step)
            model, optimizer = load_checkpoint(checkpoint, model, optimizer, strict=True)
            start_step = checkpoint["extra_state"]["step"] + 1
            if "train_loader_state_dict" in checkpoint["extra_state"]:
                loaders["train"].load_state_dict(checkpoint["extra_state"]["train_loader_state_dict"])
            if "valid_loader_state_dict" in checkpoint["extra_state"]:
                loaders["valid"].load_state_dict(checkpoint["extra_state"]["valid_loader_state_dict"])

    train_all_steps(
        rank,
        world_size,
        model,
        optimizer,
        loaders["train"],
        loaders["valid"],
        mlpf_config.num_steps,
        mlpf_config.patience,
        outdir,
        mlpf_config,
        trainable=mlpf_config.model.trainable,
        start_step=start_step,
        lr_schedule=lr_schedule,
        use_ray=True,
        checkpoint_freq=mlpf_config.checkpoint_freq,
        comet_experiment=comet_experiment,
        comet_step_freq=mlpf_config.comet_step_freq,
        dtype=getattr(torch, mlpf_config.dtype),
        val_freq=mlpf_config.val_freq,
        checkpoint_dir=checkpoint_dir,
        train_sampler=samplers["train"],
        valid_sampler=samplers["valid"],
    )
