try:
    import comet_ml
except ModuleNotFoundError as e:
    print("comet_ml not found, ignoring")

import sys
import os
import yaml
import json
import datetime
import glob
import random
import platform
import numpy as np
from pathlib import Path
import click
from tqdm import tqdm
import shutil
from functools import partial
import shlex
import subprocess
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
import keras_tuner as kt

from tfmodel.data import Dataset
from tfmodel.datasets import CMSDatasetFactory, DelphesDatasetFactory
from tfmodel.model_setup import (
    make_model,
    configure_model_weights,
    LearningRateLoggingCallback,
    prepare_callbacks,
    FlattenedCategoricalAccuracy,
    SingleClassRecall,
    eval_model,
    freeze_model,
)

from tfmodel.utils import (
    get_lr_schedule,
    get_optimizer,
    create_experiment_dir,
    get_strategy,
    make_weight_function,
    load_config,
    compute_weights_invsqrt,
    compute_weights_none,
    get_train_val_datasets,
    get_dataset_def,
    prepare_val_data,
    set_config_loss,
    get_loss_dict,
    parse_config,
    get_best_checkpoint,
    delete_all_but_best_checkpoint,
    get_tuner,
    get_raytune_schedule,
    get_heptfds_dataset,
    get_datasets
)

from tfmodel.lr_finder import LRFinder
from tfmodel.callbacks import CustomTensorBoard
from tfmodel import hypertuning
from tfmodel.utils_analysis import plot_ray_analysis, analyze_ray_experiment, topk_summary_plot_v2, summarize_top_k

import ray
from ray import tune
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.integration.tensorflow import DistributedTrainableCreator
from ray.tune.logger import TBXLoggerCallback
from ray.tune import Analysis

def customize_gun_sample(config):

    config["dataset"]["classification_loss_coef"] = 0.0
    config["dataset"]["charge_loss_coef"] = 0.0
    config["dataset"]["eta_loss_coef"] = 0.0
    config["dataset"]["sin_phi_loss_coef"] = 0.0
    config["dataset"]["cos_phi_loss_coef"] = 0.0
    config["setup"]["trainable"] = "regression"

    config["training_dataset"] = "cms_pf_single_pi"
    config["testing_dataset"] = "cms_pf_single_pi"
    return config

def customize_pipeline_test(config):
    config["training_datasets"] = [config["training_datasets"][0], ]
    config["testing_datasets"] = [config["testing_datasets"][0], ]
    return config

customization_functions = {
    "gun_sample": customize_gun_sample,
    "pipeline_test": customize_pipeline_test
}

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
def train(config, weights, ntrain, ntest, nepochs, recreate, prefix, plot_freq, customize):

    try:
        from comet_ml import Experiment
        experiment = Experiment(
            project_name="particleflow-tf",
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
        )
    except Exception as e:
        print("Failed to initialize comet-ml dashboard")
        experiment = None


    """Train a model defined by config"""
    config_file_path = config
    config, config_file_stem = parse_config(
        config, nepochs=nepochs, weights=weights
    )

    if plot_freq:
        config["callbacks"]["plot_freq"] = plot_freq

    if customize:
        config = customization_functions[customize](config)

    if recreate or (weights is None):
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
    else:
        outdir = str(Path(weights).parent)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, num_gpus = get_strategy()
    #if "CPU" not in strategy.extended.worker_devices[0]:
    #    nvidia_smi_call = "nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f {}/nvidia_smi_log.csv".format(outdir)
    #    p = subprocess.Popen(shlex.split(nvidia_smi_call))

    ds_train, num_train_steps = get_datasets(config["training_datasets"], config, num_gpus, "train")
    ds_test, num_test_steps = get_datasets(config["testing_datasets"], config, num_gpus, "test")
    ds_val, ds_info = get_heptfds_dataset(config["validation_dataset"], config, num_gpus, "test", config["setup"]["num_events_validation"])

    if ntrain:
        ds_train = ds_train.take(ntrain)
        num_train_steps = ntrain
    if ntest:
        ds_test = ds_test.take(ntest)
        num_test_steps = ntest

    print("num_train_steps", num_train_steps)
    print("num_test_steps", num_test_steps)

    if experiment:
        experiment.set_name(outdir)
        experiment.log_code("mlpf/tfmodel/model.py")
        experiment.log_code("mlpf/tfmodel/utils.py")
        experiment.log_code(config_file_path)

    shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    with strategy.scope():
        lr_schedule, optim_callbacks = get_lr_schedule(config, steps=num_train_steps)
        opt = get_optimizer(config, lr_schedule)

        if config["setup"]["dtype"] == "float16":
            model_dtype = tf.dtypes.float16
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        model = make_model(config, model_dtype)

        # Build the layers after the element and feature dimensions are specified
        model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

        initial_epoch = 0
        if weights:
            # We need to load the weights in the same trainable configuration as the model was set up
            configure_model_weights(model, config["setup"].get("weights_config", "all"))
            model.load_weights(weights, by_name=True)
            initial_epoch = int(weights.split("/")[-1].split("-")[1])
        model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

        config = set_config_loss(config, config["setup"]["trainable"])
        configure_model_weights(model, config["setup"]["trainable"])
        model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

        print("model weights")
        tw_names = [m.name for m in model.trainable_weights]
        for w in model.weights:
            print("layer={} trainable={} shape={} num_weights={}".format(w.name, w.name in tw_names, w.shape, np.prod(w.shape)))

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
            metrics={
                "cls": [
                    FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                    FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                ] + [
                    SingleClassRecall(
                        icls,
                        name="rec_cls{}".format(icls),
                        dtype=tf.float64) for icls in range(config["dataset"]["num_output_classes"])
                ]
            },
        )
        model.summary()

    callbacks = prepare_callbacks(
        config["callbacks"],
        outdir,
        ds_val,
        ds_info,
        comet_experiment=experiment
    )
    callbacks.append(optim_callbacks)

    fit_result = model.fit(
        ds_train.repeat(),
        validation_data=ds_test.repeat(),
        epochs=initial_epoch + config["setup"]["num_epochs"],
        callbacks=callbacks,
        steps_per_epoch=num_train_steps,
        validation_steps=num_test_steps,
        initial_epoch=initial_epoch,
    )


    history_path = Path(outdir) / "history"
    history_path = str(history_path)
    with open("{}/history.json".format(history_path), "w") as fi:
        json.dump(fit_result.history, fi)
    model.save(outdir + "/model_full", save_format="tf")

    print("Training done.")

    #if "CPU" not in strategy.extended.worker_devices[0]:
    #    p.terminate()

@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", required=True, help="directory containing a completed training", type=click.Path())
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("-e", "--evaluation_dir", help="optionally specify evaluation output dir", type=click.Path())
def evaluate(config, train_dir, weights, evaluation_dir):
    """Evaluate the trained model in train_dir"""
    if config is None:
        config = Path(train_dir) / "config.yaml"
        assert config.exists(), "Could not find config file in train_dir, please provide one with -c <path/to/config>"
    config, _ = parse_config(config, weights=weights)

    if evaluation_dir is None:
        eval_dir = str(Path(train_dir) / "evaluation")
    else:
        eval_dir = evaluation_dir

    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    strategy, num_gpus = get_strategy()
    ds_test, _ = get_heptfds_dataset(config["validation_dataset"], config, num_gpus, "test")

    model = make_model(config, model_dtype)
    model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

    # need to load the weights in the same trainable configuration as the model was set up
    configure_model_weights(model, config["setup"].get("weights_config", "all"))
    if weights:
        model.load_weights(weights, by_name=True)
    else:
        weights = get_best_checkpoint(train_dir)
        print("Loading best weights that could be found from {}".format(weights))
        model.load_weights(weights, by_name=True)
    
    eval_model(model, ds_test, config, eval_dir)
    freeze_model(model, config, ds_test.take(1), train_dir)

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
    strategy, num_gpus = get_strategy()

    ds_train, ds_info = get_heptfds_dataset(config["training_dataset"], config, num_gpus, "train", config["setup"]["num_events_train"])
    ds_train = ds_train.take(1)

    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=1e-7)  # This learning rate will be changed by the lr_finder
        if config["setup"]["dtype"] == "float16":
            model_dtype = tf.dtypes.float16
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        model = make_model(config, model_dtype)
        config = set_config_loss(config, config["setup"]["trainable"])

        # Run model once to build the layers
        model.build((1, config["dataset"]["padded_num_elem_size"], config["dataset"]["num_input_features"]))

        configure_model_weights(model, config["setup"]["trainable"])

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights=loss_weights,
            metrics={
                "cls": [
                    FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                    FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                ]
            },
        )
        model.summary()

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

    strategy, num_gpus = get_strategy()
 
    ds_train, ds_info = get_heptfds_dataset(config["training_dataset"], config, num_gpus, "train", config["setup"]["num_events_train"])
    ds_test, _ = get_heptfds_dataset(config["testing_dataset"], config, num_gpus, "test", config["setup"]["num_events_test"])
    ds_val, _ = get_heptfds_dataset(config["validation_dataset"], config, num_gpus, "test", config["setup"]["num_events_validation"])

    num_train_steps = 0
    for _ in ds_train:
        num_train_steps += 1
    num_test_steps = 0
    for _ in ds_test:
        num_test_steps += 1

    model_builder, optim_callbacks = hypertuning.get_model_builder(config, num_train_steps)

    callbacks = prepare_callbacks(
        config["callbacks"],
        outdir,
        ds_val,
        ds_info,
    )

    callbacks.append(optim_callbacks)
    callbacks.append(tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'))

    tuner = get_tuner(config["hypertune"], model_builder, outdir, recreate, strategy)
    tuner.search_space_summary()

    tuner.search(
        ds_train.repeat(),
        epochs=config["setup"]["num_epochs"],
        validation_data=ds_test.repeat(),
        steps_per_epoch=num_train_steps,
        validation_steps=num_test_steps,
        callbacks=callbacks,
    )
    print("Hyperparameter search complete.")
    shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    tuner.results_summary()
    for trial in tuner.oracle.get_best_trials(num_trials=10):
        print(trial.hyperparameters.values, trial.score)


def set_raytune_search_parameters(search_space, config):
    config["parameters"]["combined_graph_layer"]["layernorm"] = search_space["layernorm"]
    config["parameters"]["combined_graph_layer"]["ffn_dist_hidden_dim"] = search_space["ffn_dist_hidden_dim"]
    config["parameters"]["combined_graph_layer"]["distance_dim"] = search_space["distance_dim"]
    config["parameters"]["combined_graph_layer"]["num_node_messages"] = search_space["num_node_messages"]
    config["parameters"]["combined_graph_layer"]["node_message"]["normalize_degrees"] = search_space["normalize_degrees"]
    config["parameters"]["combined_graph_layer"]["node_message"]["output_dim"] = search_space["output_dim"]
    config["parameters"]["num_graph_layers_common"] = search_space["num_graph_layers_common"]
    config["parameters"]["num_graph_layers_energy"] = search_space["num_graph_layers_energy"]
    config["parameters"]["combined_graph_layer"]["dropout"] = search_space["dropout"]
    config["parameters"]["combined_graph_layer"]["bin_size"] = search_space["bin_size"]
    config["parameters"]["combined_graph_layer"]["kernel"]["clip_value_low"] = search_space["clip_value_low"]


    config["setup"]["lr"] = search_space["lr"]
    if isinstance(config["training_datasets"], list):
        training_dataset = config["training_datasets"][0]
    else:
        training_dataset = config["training_datasets"]
    config["datasets"][training_dataset]["batch_per_gpu"] = search_space["batch_size"]

    config["exponentialdecay"]["decay_steps"] = search_space["expdecay_decay_steps"]
    return config


def build_model_and_train(config, checkpoint_dir=None, full_config=None, ntrain=None, ntest=None):
        full_config, config_file_stem = parse_config(full_config)

        if config is not None:
            full_config = set_raytune_search_parameters(search_space=config, config=full_config)

        strategy, num_gpus = get_strategy()

        ds_train, num_train_steps = get_datasets(full_config["training_datasets"], full_config, num_gpus, "train")
        ds_test, num_test_steps = get_datasets(full_config["testing_datasets"], full_config, num_gpus, "test")
        ds_val, ds_info = get_heptfds_dataset(full_config["validation_dataset"], full_config, num_gpus, "test", full_config["setup"]["num_events_validation"])

        if ntrain:
            ds_train = ds_train.take(ntrain)
            num_train_steps = ntrain
        if ntest:
            ds_test = ds_test.take(ntest)
            num_test_steps = ntest

        print("num_train_steps", num_train_steps)
        print("num_test_steps", num_test_steps)

        callbacks = prepare_callbacks(
            full_config["callbacks"],
            tune.get_trial_dir(),
            ds_val,
            ds_info,
        )

        with strategy.scope():
            lr_schedule, optim_callbacks = get_lr_schedule(full_config, steps=num_train_steps)
            callbacks.append(optim_callbacks)
            opt = get_optimizer(full_config, lr_schedule)

            model = make_model(full_config, dtype=tf.dtypes.float32)

            # Run model once to build the layers
            model.build((1, full_config["dataset"]["padded_num_elem_size"], full_config["dataset"]["num_input_features"]))

            full_config = set_config_loss(full_config, full_config["setup"]["trainable"])
            configure_model_weights(model, full_config["setup"]["trainable"])
            model.build((1, full_config["dataset"]["padded_num_elem_size"], full_config["dataset"]["num_input_features"]))

            loss_dict, loss_weights = get_loss_dict(full_config)
            model.compile(
                loss=loss_dict,
                optimizer=opt,
                sample_weight_mode="temporal",
                loss_weights=loss_weights,
                metrics={
                    "cls": [
                        FlattenedCategoricalAccuracy(name="acc_unweighted", dtype=tf.float64),
                        FlattenedCategoricalAccuracy(use_weights=True, name="acc_weighted", dtype=tf.float64),
                    ]
                },
            )
            model.summary()


            callbacks.append(TuneReportCheckpointCallback(
                metrics=[
                    "adam_beta_1",
                    'charge_loss',
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
                ),
            )

            fit_result = model.fit(
                ds_train.repeat(),
                validation_data=ds_test.repeat(),
                epochs=full_config["setup"]["num_epochs"],
                callbacks=callbacks,
                steps_per_epoch=num_train_steps,
                validation_steps=num_test_steps,
            )


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
def raytune(config, name, local, cpus, gpus, tune_result_dir, resume, ntrain, ntest):
    cfg = load_config(config)
    config_file_path = config

    if tune_result_dir is not None:
        os.environ["TUNE_RESULT_DIR"] = tune_result_dir
    else:
        if isinstance(cfg["raytune"]["local_dir"], type(None)):
            raise TypeError("Please specify a local_dir in the raytune section of the config file.")
        trd = cfg["raytune"]["local_dir"] + "/tune_result_dir"
        os.environ["TUNE_RESULT_DIR"] = trd

    ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 24 * 60 * 60  # Avoid timeout errors
    if not local:
        ray.init(address='auto')

    search_space = {
        # Optimizer parameters
        "lr": tune.grid_search(cfg["raytune"]["parameters"]["lr"]),
        "batch_size": tune.grid_search(cfg["raytune"]["parameters"]["batch_size"]),
        "expdecay_decay_steps": tune.grid_search(cfg["raytune"]["parameters"]["expdecay_decay_steps"]),

        # Model parameters
        "layernorm": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["layernorm"]),
        "ffn_dist_hidden_dim": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["ffn_dist_hidden_dim"]),
        "distance_dim": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["distance_dim"]),
        "num_node_messages": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["num_node_messages"]),
        "num_graph_layers_common": tune.grid_search(cfg["raytune"]["parameters"]["num_graph_layers_common"]),
        "num_graph_layers_energy": tune.grid_search(cfg["raytune"]["parameters"]["num_graph_layers_energy"]),
        "dropout": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["dropout"]),
        "bin_size": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["bin_size"]),
        "clip_value_low": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["kernel"]["clip_value_low"]),
        "normalize_degrees": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["node_message"]["normalize_degrees"]),
        "output_dim": tune.grid_search(cfg["raytune"]["parameters"]["combined_graph_layer"]["node_message"]["output_dim"]),
    }

    sched = get_raytune_schedule(cfg["raytune"])

    distributed_trainable = DistributedTrainableCreator(
        partial(build_model_and_train, full_config=config_file_path, ntrain=ntrain, ntest=ntest),
        num_workers=1,  # Number of hosts that each trial is expected to use.
        num_cpus_per_worker=cpus,
        num_gpus_per_worker=gpus,
        num_workers_per_host=1,  # Number of workers to colocate per host. None if not specified.
        timeout_s=24 * 60 * 60,
    )

    analysis = tune.run(
        distributed_trainable,
        config=search_space,
        name=name,
        scheduler=sched,
        num_samples=1,
        local_dir=cfg["raytune"]["local_dir"],
        callbacks=[TBXLoggerCallback()],
        log_to_file=True,
        resume=resume,
        max_failures=10,
    )
    print("Best hyperparameters found were: ", analysis.get_best_config("val_loss", "min"))

    skip = 20
    if skip > cfg["setup"]["num_epochs"]:
        skip = 0
    analysis.default_metric = cfg["raytune"]["default_metric"]
    analysis.default_mode = cfg["raytune"]["default_mode"]
    plot_ray_analysis(analysis, save=True, skip=skip)
    topk_summary_plot_v2(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)
    summarize_top_k(analysis, k=5, save_dir=Path(analysis.get_best_logdir()).parent)

    ray.shutdown()


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
@click.option("-s", "--save", help="save plots in trial dirs", is_flag=True)
@click.option("-k", "--skip", help="skip first values to avoid large losses at start of training", type=int)
@click.option("--metric", help="experiment dir", type=str, default="val_loss")
@click.option("--mode", help="experiment dir", type=str, default="min")
def raytune_analysis(exp_dir, save, skip, mode, metric):
    analysis = Analysis(exp_dir,  default_metric=metric, default_mode=mode)
    plot_ray_analysis(analysis, save=save, skip=skip)
    analyze_ray_experiment(exp_dir, default_metric=metric, default_mode=mode)

@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
def debug_data(config, ntrain, ntest):
    """Train a model defined by config"""
    config, config_file_stem, global_batch_size, n_train, n_test, n_epochs, weights = parse_config(
        config, ntrain, ntest, weights=None,
    )

    dataset_def = get_dataset_def(config)
    ds_train, ds_test, dataset_transform = get_train_val_datasets(config, global_batch_size=1, n_train=n_train, n_test=n_test)

    # cand_counts = np.zeros(8)
    # for data_item in tqdm(ds_train, desc="Counting"):
    #     import pdb; pdb.set_trace()
    #     cand_vals, cand_count = np.unique(np.argmax(data_item[1]['cls'], axis=2), return_counts=True)
    #     cand_counts[cand_vals.astype("int32")] += cand_count
    # print("cand_counts: ", cand_counts)

    dsf = CMSDatasetFactory(config)
    ds_train, _ = dsf.get_dataset(split="train")
    ds_test, _ = dsf.get_dataset(split="test")
    for data_item in tqdm(ds_train, desc="Counting"):
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
