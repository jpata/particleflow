import logging

logging.basicConfig(level=logging.INFO)

from comet_ml import OfflineExperiment, Experiment  # isort:skip

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    logging.warning("horovod not found, ignoring")

import os
import platform
import random
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from customizations import customization_functions
from tfmodel import hypertuning
from tfmodel.datasets.BaseDatasetFactory import mlpf_dataset_from_config
from tfmodel.lr_finder import LRFinder
from tfmodel.model_setup import eval_model, freeze_model, prepare_callbacks
from tfmodel.utils import (
    create_experiment_dir,
    delete_all_but_best_checkpoint,
    get_best_checkpoint,
    get_datasets,
    get_latest_checkpoint,
    get_strategy,
    get_train_test_val_datasets,
    get_tuner,
    initialize_horovod,
    load_config,
    model_scope,
    parse_config,
)
from tfmodel.utils_analysis import (
    analyze_ray_experiment,
    count_skipped_configurations,
    plot_ray_analysis,
    summarize_top_k,
    topk_summary_plot_v2,
)


@click.group()
@click.help_option("-h", "--help")
def main():
    pass


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training steps", type=int)
@click.option("--ntest", default=None, help="override the number of testing steps", type=int)
@click.option("--nepochs", default=None, help="override the number of training epochs", type=int)
@click.option("-r", "--recreate", help="force creation of new experiment dir", is_flag=True)
@click.option("-p", "--prefix", default="", help="prefix to put at beginning of training dir name", type=str)
@click.option("--plot-freq", default=None, help="plot detailed validation every N epochs", type=int)
@click.option("--customize", help="customization function", type=str, default=None)
@click.option("--comet-offline", help="log comet-ml experiment locally", is_flag=True)
@click.option("-j", "--jobid", help="log the Slurm job ID in experiments dir", type=str, default=None)
@click.option("-m", "--horovod-enabled", help="Enable multi-node training using Horovod", is_flag=True)
@click.option("-g", "--habana-enabled", help="Enable training on Habana Gaudi", is_flag=True)
@click.option(
    "-b",
    "--benchmark_dir",
    help="dir to save benchmark results. If 'exp_dir' results will be saved in the \
    experiment folder",
    type=str,
    default=None,
)
@click.option("--batch-multiplier", help="batch size per device", type=int, default=None)
@click.option("--num-cpus", help="number of CPU threads to use", type=int, default=1)
@click.option("--seeds", help="set the random seeds", is_flag=True, default=True)
def train(
    config,
    weights,
    ntrain,
    ntest,
    nepochs,
    recreate,
    prefix,
    plot_freq,
    customize,
    comet_offline,
    jobid,
    horovod_enabled,
    habana_enabled,
    benchmark_dir,
    batch_multiplier,
    num_cpus,
    seeds,
):

    if seeds:
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    """Train a model defined by config"""
    config_file_path = config
    config, config_file_stem = parse_config(config, nepochs=nepochs, weights=weights)
    logging.info(f"loaded config file: {config_file_path}")

    if plot_freq:
        config["callbacks"]["plot_freq"] = plot_freq

    if batch_multiplier:
        if config["batching"]["bucket_by_sequence_length"]:
            logging.info(
                "Dynamic batching is enabled, changing batch size multiplier from {} to {}".format(
                    config["batching"]["batch_multiplier"], config["batching"]["batch_multiplier"] * batch_multiplier
                )
            )
            config["batching"]["batch_multiplier"] *= batch_multiplier
        else:
            for key in config["train_test_datasets"]:
                logging.info(
                    "Static batching is enabled, changing batch size for dataset {} from {} to {}".format(
                        key,
                        config["train_test_datasets"][key]["batch_per_gpu"],
                        config["train_test_datasets"][key]["batch_per_gpu"] * batch_multiplier,
                    )
                )
                config["train_test_datasets"][key]["batch_per_gpu"] *= batch_multiplier

    if customize:
        config = customization_functions[customize](config)

    # Decide tf.distribute.strategy depending on number of available GPUs
    horovod_enabled = horovod_enabled or config["setup"]["horovod_enabled"]

    if horovod_enabled:
        num_gpus, num_batches_multiplier = initialize_horovod()
    elif habana_enabled:
        import habana_frameworks.tensorflow as htf

        htf.load_habana_module()
        from habana_frameworks.tensorflow.distribute import HPUStrategy

        logging.info("Using habana_frameworks.tensorflow.distribute.HPUStrategy")
        strategy = HPUStrategy()
        num_gpus = 1
        num_batches_multiplier = 1
    else:
        strategy, num_gpus, num_batches_multiplier = get_strategy(num_cpus=num_cpus)

    outdir = ""
    if not horovod_enabled or hvd.rank() == 0:
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
        shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    try:
        if comet_offline:
            logging.info("Using comet-ml OfflineExperiment, saving logs locally.")

            experiment = OfflineExperiment(
                project_name="particleflow-tf",
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                offline_directory=outdir + "/cometml",
            )
        else:
            logging.info("Using comet-ml Experiment, streaming logs to www.comet.ml.")

            experiment = Experiment(
                project_name="particleflow-tf",
                auto_metric_logging=True,
                auto_param_logging=True,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
            )
    except Exception as e:
        logging.warning("Failed to initialize comet-ml dashboard: {}".format(e))
        experiment = None

    if experiment:
        experiment.set_name(outdir)
        experiment.log_code("mlpf/tfmodel/model.py")
        experiment.log_code("mlpf/tfmodel/utils.py")
        experiment.log_code(config_file_path)

    if jobid is not None:
        with open(f"{outdir}/{jobid}.txt", "w") as f:
            f.write(f"{jobid}\n")

    ds_train, ds_test, ds_val = get_train_test_val_datasets(config, num_batches_multiplier, ntrain, ntest)

    epochs = config["setup"]["num_epochs"]
    total_steps = ds_train.num_steps() * epochs
    logging.info("num_train_steps: {}".format(ds_train.num_steps()))
    logging.info("num_test_steps: {}".format(ds_test.num_steps()))
    logging.info("epochs: {}, total_train_steps: {}".format(epochs, total_steps))

    if horovod_enabled:
        model, optim_callbacks, initial_epoch = model_scope(config, total_steps, weights, horovod_enabled)
    else:
        with strategy.scope():
            model, optim_callbacks, initial_epoch = model_scope(config, total_steps, weights)

    with strategy.scope():
        callbacks = prepare_callbacks(
            config,
            outdir,
            ds_val,
            comet_experiment=experiment,
            horovod_enabled=horovod_enabled,
            benchmark_dir=benchmark_dir,
            num_train_steps=ds_train.num_steps(),
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            train_samples=ds_train.num_samples,
        )

        verbose = 1
        if horovod_enabled:
            callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            callbacks.append(hvd.callbacks.MetricAverageCallback())
            verbose = 1 if hvd.rank() == 0 else 0

            ds_train._num_steps /= hvd.size()
            ds_test._num_steps /= hvd.size()

        callbacks.append(optim_callbacks)

        model.fit(
            ds_train.tensorflow_dataset.repeat(),
            validation_data=ds_test.tensorflow_dataset.repeat(),
            epochs=config["setup"]["num_epochs"],
            callbacks=callbacks,
            steps_per_epoch=ds_train.num_steps(),
            validation_steps=ds_test.num_steps(),
            initial_epoch=initial_epoch,
            verbose=verbose,
        )


@main.command()
@click.help_option("-h", "--help")
@click.option("--train-dir", required=True, help="directory containing a completed training", type=click.Path())
@click.option("--config", help="configuration file", type=click.Path())
@click.option("--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("--customize", help="customization function", type=str, default=None)
@click.option("--nevents", help="override the number of events to evaluate", type=int, default=None)
def evaluate(config, train_dir, weights, customize, nevents):
    """Evaluate the trained model in train_dir"""
    if config is None:
        config = Path(train_dir) / "config.yaml"
        assert config.exists(), "Could not find config file in train_dir, please provide one with -c <path/to/config>"
    config, _ = parse_config(config, weights=weights)

    if customize:
        config = customization_functions[customize](config)

    # disable small graph optimization for onnx export (tf.cond is not well supported by ONNX export)
    if "small_graph_opt" in config["setup"]:
        config["setup"]["small_graph_opt"] = False

    if not weights:
        weights = get_best_checkpoint(train_dir)
        logging.info("Loading best weights that could be found from {}".format(weights))

    model, _, initial_epoch = model_scope(config, 1, weights=weights)

    for dsname in config["validation_datasets"]:
        ds_test = mlpf_dataset_from_config(config["validation_datasets"][0], config, "test", nevents)
        ds_test_tfds = ds_test.tensorflow_dataset.padded_batch(config["validation_batch_size"])
        eval_dir = str(Path(train_dir) / "evaluation" / "epoch_{}".format(initial_epoch) / dsname)
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        eval_model(model, ds_test_tfds, config, eval_dir)

    freeze_model(model, config, train_dir)


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-o", "--outdir", help="output directory", type=click.Path(), default=".")
@click.option("-n", "--figname", help="name of saved figure", type=click.Path(), default="lr_finder.jpg")
@click.option("-l", "--logscale", help="use log scale on y-axis in figure", default=False, is_flag=True)
def find_lr(config, outdir, figname, logscale):
    """Run the Learning Rate Finder to produce a batch loss vs. LR plot from
    which an appropriate LR-range can be determined"""
    config, _ = parse_config(config)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, num_gpus, num_batches_multiplier = get_strategy()

    ds_train, _, _ = get_datasets(config["train_test_datasets"], config, num_batches_multiplier, "train")

    with strategy.scope():
        model, _, _ = model_scope(config, 1)
        max_steps = 200
        lr_finder = LRFinder(max_steps=max_steps)
        callbacks = [lr_finder]

        model.fit(
            ds_train.repeat(),
            epochs=max_steps,
            callbacks=callbacks,
            steps_per_epoch=1,
        )

        lr_finder.plot(save_dir=outdir, figname=figname, log_scale=logscale)


@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", help="training directory", type=click.Path())
@click.option("-d", "--dry_run", help="do not delete anything", is_flag=True, default=False)
def delete_all_but_best_ckpt(train_dir, dry_run):
    """Delete all checkpoint weights in <train_dir>/weights/ except the one with lowest loss in its filename."""
    delete_all_but_best_checkpoint(train_dir, dry_run)


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path(), required=True)
@click.option("-o", "--outdir", help="output dir", type=click.Path(), required=True)
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
@click.option("-r", "--recreate", help="overwrite old hypertune results", is_flag=True, default=False)
def hypertune(config, outdir, ntrain, ntest, recreate):
    config_file_path = config
    config, _ = parse_config(config, ntrain=ntrain, ntest=ntest)

    # Override number of epochs with max_epochs from Hyperband config if specified
    if config["hypertune"]["algorithm"] == "hyperband":
        config["setup"]["num_epochs"] = config["hypertune"]["hyperband"]["max_epochs"]

    strategy, num_gpus, num_batches_multiplier = get_strategy()

    ds_train, ds_test, ds_val = get_train_test_val_datasets(config, num_batches_multiplier, ntrain, ntest)

    model_builder, optim_callbacks = hypertuning.get_model_builder(config, ds_train.num_steps())

    callbacks = prepare_callbacks(
        config,
        outdir,
        ds_val,
    )

    for cb in optim_callbacks:
        callbacks.append(cb)
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=20, monitor="val_loss"))

    tuner = get_tuner(config["hypertune"], model_builder, outdir, recreate, strategy)
    tuner.search_space_summary()

    tuner.search(
        ds_train.repeat(),
        epochs=config["setup"]["num_epochs"],
        validation_data=ds_test.repeat(),
        steps_per_epoch=ds_train.num_steps(),
        validation_steps=ds_test.num_steps(),
        callbacks=[],
    )
    logging.info("Hyperparameter search complete.")
    shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    tuner.results_summary()
    for trial in tuner.oracle.get_best_trials(num_trials=10):
        logging.info(trial.hyperparameters.values, trial.score)


#
# Raytune part
#


def raytune_build_model_and_train(
    config, checkpoint_dir=None, full_config=None, ntrain=None, ntest=None, name=None, seeds=False
):
    from collections import Counter

    from ray import tune
    from ray.tune.integration.keras import TuneReportCheckpointCallback
    from raytune.search_space import set_raytune_search_parameters

    if seeds:
        # Set seeds for reproducibility
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    full_config, config_file_stem = parse_config(full_config)

    if config is not None:
        full_config = set_raytune_search_parameters(search_space=config, config=full_config)

    logging.info("Using comet-ml OfflineExperiment, saving logs locally.")
    experiment = OfflineExperiment(
        project_name="particleflow-tf-gen",
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=False,
        auto_histogram_activation_logging=False,
        offline_directory=tune.get_trial_dir() + "/cometml",
    )

    strategy, num_gpus, num_batches_multiplier = get_strategy()
    ds_train, ds_test, ds_val = get_train_test_val_datasets(config, num_batches_multiplier, ntrain, ntest)

    logging.info("num_train_steps", ds_train.num_steps())
    logging.info("num_test_steps", ds_test.num_steps())
    total_steps = ds_train.num_steps() * full_config["setup"]["num_epochs"]
    logging.info("total_steps", total_steps)

    with strategy.scope():
        weights = get_latest_checkpoint(Path(checkpoint_dir).parent) if (checkpoint_dir is not None) else None
        model, optim_callbacks, initial_epoch = model_scope(full_config, total_steps, weights=weights)

    callbacks = prepare_callbacks(
        full_config,
        tune.get_trial_dir(),
        ds_val,
        comet_experiment=experiment,
        horovod_enabled=False,
    )

    callbacks.append(optim_callbacks)

    tune_report_checkpoint_callback = TuneReportCheckpointCallback(
        metrics=[
            "adam_beta_1",
            "charge_loss",
            "cls_acc_unweighted",
            "cls_loss",
            "cos_phi_loss",
            "energy_loss",
            "eta_loss",
            "learning_rate",
            "loss",
            "pt_loss",
            "sin_phi_loss",
            "val_charge_loss",
            "val_cls_acc_unweighted",
            "val_cls_acc_weighted",
            "val_cls_loss",
            "val_cos_phi_loss",
            "val_energy_loss",
            "val_eta_loss",
            "val_loss",
            "val_pt_loss",
            "val_sin_phi_loss",
        ],
    )

    # To make TuneReportCheckpointCallback continue the numbering of checkpoints correctly
    if weights is not None:
        latest_saved_checkpoint_number = int(Path(weights).name.split("-")[1])
        logging.info("setting TuneReportCheckpointCallback epoch number to {}".format(latest_saved_checkpoint_number))
        tune_report_checkpoint_callback._checkpoint._counter = Counter()
        tune_report_checkpoint_callback._checkpoint._counter["epoch_end"] = latest_saved_checkpoint_number
        tune_report_checkpoint_callback._checkpoint._cp_count = latest_saved_checkpoint_number
    callbacks.append(tune_report_checkpoint_callback)

    try:
        model.fit(
            ds_train.repeat(),
            validation_data=ds_test.repeat(),
            epochs=full_config["setup"]["num_epochs"],
            callbacks=callbacks,
            steps_per_epoch=ds_train.num_steps(),
            validation_steps=ds_test.num_steps(),
            initial_epoch=initial_epoch,
        )
    except tf.errors.ResourceExhaustedError:
        logging.warning("Resource exhausted, skipping this hyperparameter configuration.")
        skiplog_file_path = Path(full_config["raytune"]["local_dir"]) / name / "skipped_configurations.txt"
        lines = ["{}: {}\n".format(item[0], item[1]) for item in config.items()]

        with open(skiplog_file_path, "a") as f:
            f.write("#" * 80 + "\n")
            for line in lines:
                f.write(line)
                logging.warning(line[:-1])
            f.write("#" * 80 + "\n\n")


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-n", "--name", help="experiment name", type=str, default="test_exp")
@click.option("-l", "--local", help="run locally", is_flag=True)
@click.option("--cpus", help="number of cpus per worker", type=int, default=1)
@click.option("--gpus", help="number of gpus per worker", type=int, default=0)
@click.option("--tune_result_dir", help="Tune result dir", type=str, default=None)
@click.option("-r", "--resume", help="resume run from local_dir", is_flag=True)
@click.option("--ntrain", default=None, help="override the number of training steps", type=int)
@click.option("--ntest", default=None, help="override the number of testing steps", type=int)
@click.option("-s", "--seeds", help="set the random seeds", is_flag=True)
def raytune(config, name, local, cpus, gpus, tune_result_dir, resume, ntrain, ntest, seeds):
    import ray
    from ray import tune
    from ray.tune.logger import TBXLoggerCallback
    from raytune.search_space import raytune_num_samples, search_space
    from raytune.utils import get_raytune_schedule, get_raytune_search_alg

    if seeds:
        # Set seeds for reproducibility
        random.seed(1234)
        np.random.seed(1234)
        tf.random.set_seed(1234)

    cfg = load_config(config)
    config_file_path = config

    if tune_result_dir is not None:
        os.environ["TUNE_RESULT_DIR"] = tune_result_dir
    else:
        if isinstance(cfg["raytune"]["local_dir"], type(None)):
            raise TypeError("Please specify a local_dir in the raytune section of the config file.")
        trd = cfg["raytune"]["local_dir"] + "/tune_result_dir"
        os.environ["TUNE_RESULT_DIR"] = trd

    expdir = Path(cfg["raytune"]["local_dir"]) / name
    expdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "mlpf/raytune/search_space.py", str(Path(cfg["raytune"]["local_dir"]) / name / "search_space.py")
    )  # Copy the config file to the train dir for later reference
    shutil.copy(
        config_file_path, str(Path(cfg["raytune"]["local_dir"]) / name / "config.yaml")
    )  # Copy the config file to the train dir for later reference

    ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 1 * 60 * 60  # Avoid timeout errors
    if not local:
        ray.init(address="auto")

    sched = get_raytune_schedule(cfg["raytune"])
    search_alg = get_raytune_search_alg(cfg["raytune"], seeds)

    sync_config = tune.SyncConfig(sync_to_driver=False)

    start = datetime.now()
    analysis = tune.run(
        partial(
            raytune_build_model_and_train, full_config=config_file_path, ntrain=ntrain, ntest=ntest, name=name, seeds=seeds
        ),
        config=search_space,
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        name=name,
        scheduler=sched,
        search_alg=search_alg,
        num_samples=raytune_num_samples,
        local_dir=cfg["raytune"]["local_dir"],
        callbacks=[TBXLoggerCallback()],
        log_to_file=True,
        resume=resume,
        max_failures=2,
        sync_config=sync_config,
        stop=tune.stopper.MaximumIterationStopper(cfg["setup"]["num_epochs"]),
    )
    end = datetime.now()
    logging.info("Total time of tune.run(...): {}".format(end - start))

    logging.info(
        "Best hyperparameters found according to {} were: ".format(cfg["raytune"]["default_metric"]),
        analysis.get_best_config(cfg["raytune"]["default_metric"], cfg["raytune"]["default_mode"]),
    )

    skip = 20
    if skip > cfg["setup"]["num_epochs"]:
        skip = 0
    analysis.default_metric = cfg["raytune"]["default_metric"]
    analysis.default_mode = cfg["raytune"]["default_mode"]
    plot_ray_analysis(analysis, save=True, skip=skip)
    topk_summary_plot_v2(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)
    summarize_top_k(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)

    best_params = analysis.get_best_config(cfg["raytune"]["default_metric"], cfg["raytune"]["default_mode"])
    with open(Path(analysis.get_best_logdir()).parent / "best_parameters.txt", "a") as best_params_file:
        best_params_file.write("Best hyperparameters according to {}\n".format(cfg["raytune"]["default_metric"]))
        for key, val in best_params.items():
            best_params_file.write(("{}: {}\n".format(key, val)))

    with open(Path(analysis.get_best_logdir()).parent / "time.txt", "a") as timefile:
        timefile.write(str(end - start) + "\n")

    num_skipped = count_skipped_configurations(analysis.get_best_logdir())
    logging.info("Number of skipped configurations: {}".format(num_skipped))


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
def count_skipped(exp_dir):
    num_skipped = count_skipped_configurations(exp_dir)
    logging.info("Number of skipped configurations: {}".format(num_skipped))


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
@click.option("-s", "--save", help="save plots in trial dirs", is_flag=True)
@click.option("-k", "--skip", help="skip first values to avoid large losses at start of training", type=int)
@click.option("--metric", help="experiment dir", type=str, default="val_loss")
@click.option("--mode", help="experiment dir", type=str, default="min")
def raytune_analysis(exp_dir, save, skip, mode, metric):
    from ray.tune import ExperimentAnalysis

    experiment_analysis = ExperimentAnalysis(exp_dir, default_metric=metric, default_mode=mode)
    plot_ray_analysis(experiment_analysis, save=save, skip=skip)
    analyze_ray_experiment(exp_dir, default_metric=metric, default_mode=mode)


if __name__ == "__main__":
    main()
