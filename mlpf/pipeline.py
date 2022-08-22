try:
    import comet_ml
except ModuleNotFoundError as e:
    print("comet_ml not found, ignoring")

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    print("hvd not enabled, ignoring")

import sys
import os
import yaml
import json
from datetime import datetime
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
import logging
import pickle

import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
import keras


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
    set_config_loss,
    get_loss_dict,
    parse_config,
    get_best_checkpoint,
    delete_all_but_best_checkpoint,
    get_tuner,
    get_heptfds_dataset,
    get_datasets,
)

from tfmodel.lr_finder import LRFinder
from tfmodel import hypertuning
from tfmodel.utils_analysis import (
    plot_ray_analysis,
    analyze_ray_experiment,
    topk_summary_plot_v2,
    summarize_top_k,
    count_skipped_configurations,
)



def customize_pipeline_test(config):
    #for cms.yaml, keep only ttbar
    if "physical" in config["train_test_datasets"]:
        config["train_test_datasets"]["physical"]["datasets"] = ["cms_pf_ttbar"]
        config["train_test_datasets"] = {"physical": config["train_test_datasets"]["physical"]}
        config["train_test_datasets"]["physical"]["batch_per_gpu"] = 5
        config["validation_datasets"] = ["cms_pf_ttbar"]

    return config

customization_functions = {
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
@click.option("--comet-offline", help="log comet-ml experiment locally", is_flag=True)
def train(config, weights, ntrain, ntest, nepochs, recreate, prefix, plot_freq, customize, comet_offline):

    #tf.debugging.enable_check_numerics()

    """Train a model defined by config"""
    config_file_path = config
    config, config_file_stem = parse_config(
        config, nepochs=nepochs, weights=weights
    )

    if plot_freq:
        config["callbacks"]["plot_freq"] = plot_freq

    if customize:
        config = customization_functions[customize](config)
    

    try:
        if comet_offline:
            print("Using comet-ml OfflineExperiment, saving logs locally.")
            from comet_ml import OfflineExperiment
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
            print("Using comet-ml Experiment, streaming logs to www.comet.ml.")
            from comet_ml import Experiment
            offline_dir = None
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

    # Decide tf.distribute.strategy depending on number of available GPUs
    horovod_enabled = config["setup"]["horovod_enabled"]
    if horovod_enabled:
        num_gpus = initialize_horovod()
    else:
        strategy, num_gpus = get_strategy()
    outdir = ''
    if horovod_enabled:
        if hvd.rank() == 0:
            outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
            if experiment:
                experiment.set_name(outdir)
                experiment.log_code("mlpf/tfmodel/model.py")
                experiment.log_code("mlpf/tfmodel/utils.py")
                experiment.log_code(config_file_path)
            
            shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference
    else:
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
        if experiment:
            experiment.set_name(outdir)
            experiment.log_code("mlpf/tfmodel/model.py")
            experiment.log_code("mlpf/tfmodel/utils.py")
            experiment.log_code(config_file_path)
        
        shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    ds_train, num_train_steps = get_datasets(config["train_test_datasets"], config, num_gpus, "train")
    ds_test, num_test_steps = get_datasets(config["train_test_datasets"], config, num_gpus, "test")
    ds_val, ds_info = get_heptfds_dataset(config["validation_datasets"][0], config, num_gpus, "test", config["setup"]["num_events_validation"])
    ds_val = ds_val.batch(5)

    if ntrain:
        ds_train = ds_train.take(ntrain)
        num_train_steps = ntrain
    if ntest:
        ds_test = ds_test.take(ntest)
        num_test_steps = ntest

    print("num_train_steps", num_train_steps)
    print("num_test_steps", num_test_steps)
    total_steps = num_train_steps * config["setup"]["num_epochs"]
    print("total_steps", total_steps)

    
    if horovod_enabled :
        model,optim_callbacks,initial_epoch = model_scope(config, total_steps, weights, horovod_enabled)
    else:
        with strategy.scope():
            model,optim_callbacks,initial_epoch = model_scope(config, total_steps, weights)

    callbacks = prepare_callbacks(
        config["callbacks"],
        outdir,
        ds_val,
        ds_info,
        comet_experiment=experiment,
        horovod_enabled=config["setup"]["horovod_enabled"]
    )

    verbose = 1
    if horovod_enabled: 
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())
        verbose = 1 if hvd.rank() == 0 else 0

        num_train_steps /= hvd.size()
        num_test_steps /= hvd.size()


    callbacks.append(optim_callbacks)

    
    fit_result = model.fit(
        ds_train.repeat(),
        validation_data=ds_test.repeat(),
        epochs=initial_epoch + config["setup"]["num_epochs"],
        callbacks=callbacks,
        steps_per_epoch=num_train_steps,
        validation_steps=num_test_steps,
        initial_epoch=initial_epoch,
        verbose=verbose
    )

    if horovod_enabled:
        if hvd.rank() == 0:
            model_save(outdir, fit_result, model, weights)
    else:
        model_save(outdir, fit_result, model, weights)

    #if "CPU" not in strategy.extended.worker_devices[0]:
    #    p.terminate()

def model_save(outdir, fit_result, model, weights):
    history_path = Path(outdir) / "history"
    history_path = str(history_path)
    with open("{}/history.json".format(history_path), "w") as fi:
        json.dump(fit_result.history, fi)

    weights = get_best_checkpoint(outdir)
    print("Loading best weights that could be found from {}".format(weights))
    model.load_weights(weights, by_name=True)

    model.save(outdir + "/model_full", save_format="tf")

    print("Training done.")

def model_scope(config, total_steps, weights, horovod_enabled=False):
    lr_schedule, optim_callbacks, lr = get_lr_schedule(config, steps=total_steps)
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
    loaded_opt = None
    
    if weights:
        if lr_schedule:
            raise Exception("Restoring the optimizer state with a learning rate schedule is currently not supported")

        # We need to load the weights in the same trainable configuration as the model was set up
        configure_model_weights(model, config["setup"].get("weights_config", "all"))
        model.load_weights(weights, by_name=True)
        opt_weight_file = weights.replace("hdf5", "pkl").replace("/weights-", "/opt-")
        if os.path.isfile(opt_weight_file):
            loaded_opt = pickle.load(open(opt_weight_file, "rb"))

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

    #Set the optimizer weights
    if loaded_opt:
        def model_weight_setting():
            grad_vars = model.trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
            if model.optimizer.__class__.__module__ == "keras.optimizers.optimizer_v1":
                model.optimizer.optimizer.optimizer.set_weights(loaded_opt["weights"])
            else:
                model.optimizer.set_weights(loaded_opt["weights"])
        try:
            strategy.run(model_weight_setting)
        except Exception as e:
            print(e)

    return model,optim_callbacks,initial_epoch

def initialize_horovod():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    return hvd.size()




@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", required=True, help="directory containing a completed training", type=click.Path())
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
def compute_validation_loss(config, train_dir, weights):
    """Evaluate the trained model in train_dir"""
    if config is None:
        config = Path(train_dir) / "config.yaml"
        assert config.exists(), "Could not find config file in train_dir, please provide one with -c <path/to/config>"
    config, _ = parse_config(config, weights=weights)

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    strategy, num_gpus = get_strategy()
    ds_test, num_test_steps = get_datasets(config["train_test_datasets"], config, num_gpus, "test")

    with strategy.scope():
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

        loss_dict, loss_weights = get_loss_dict(config)
        model.compile(
            loss=loss_dict,
            # sample_weight_mode="temporal",
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

        losses = model.evaluate(
            x=ds_test,
            steps=num_test_steps,
            return_dict=True,
        )
    with open("{}/losses.txt".format(train_dir), "w") as loss_file:
        loss_file.write(json.dumps(losses) + "\n")

@main.command()
@click.help_option("-h", "--help")
@click.option("-t", "--train_dir", required=True, help="directory containing a completed training", type=click.Path())
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
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

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    strategy, num_gpus = get_strategy()
    #physical_devices = tf.config.list_physical_devices('GPU')
    #for dev in physical_devices:
    #    tf.config.experimental.set_memory_growth(dev, True)

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

    iepoch = int(weights.split("/")[-1].split("-")[1])

    for dsname in config["validation_datasets"]:
        ds_test, _ = get_heptfds_dataset(dsname, config, num_gpus, "test", supervised=False)
        if nevents:
            ds_test = ds_test.take(nevents)
        ds_test = ds_test.batch(5)
        eval_dir = str(Path(train_dir) / "evaluation" / "epoch_{}".format(iepoch) / dsname)
        Path(eval_dir).mkdir(parents=True, exist_ok=True)
        eval_model(model, ds_test, config, eval_dir)

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
    strategy, num_gpus = get_strategy()

    ds_train, num_train_steps = get_datasets(config["train_test_datasets"], config, num_gpus, "train")

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
    ds_val, _ = get_heptfds_dataset(config["validation_datasets"][0], config, num_gpus, "test", config["setup"]["num_events_validation"])

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


def build_model_and_train(config, checkpoint_dir=None, full_config=None, ntrain=None, ntest=None, name=None, seeds=False):
        from ray import tune
        from raytune.search_space import set_raytune_search_parameters
        from ray.tune.integration.keras import TuneReportCheckpointCallback
        if seeds:
            # Set seeds for reproducibility
            random.seed(1234)
            np.random.seed(1234)
            tf.random.set_seed(1234)

        full_config, config_file_stem = parse_config(full_config)

        if config is not None:
            full_config = set_raytune_search_parameters(search_space=config, config=full_config)

        strategy, num_gpus = get_strategy()

        ds_train, num_train_steps = get_datasets(full_config["train_test_datasets"], full_config, num_gpus, "train")
        ds_test, num_test_steps = get_datasets(full_config["train_test_datasets"], full_config, num_gpus, "test")
        ds_val, ds_info = get_heptfds_dataset(full_config["validation_datasets"][0], full_config, num_gpus, "test", full_config["setup"]["num_events_validation"])

        if ntrain:
            ds_train = ds_train.take(ntrain)
            num_train_steps = ntrain
        if ntest:
            ds_test = ds_test.take(ntest)
            num_test_steps = ntest

        print("num_train_steps", num_train_steps)
        print("num_test_steps", num_test_steps)
        total_steps = num_train_steps * full_config["setup"]["num_epochs"]
        print("total_steps", total_steps)

        callbacks = prepare_callbacks(
            full_config["callbacks"],
            tune.get_trial_dir(),
            ds_val,
            ds_info,
        )

        callbacks = callbacks[:-1]  # remove the CustomCallback at the end of the list

        with strategy.scope():
            lr_schedule, optim_callbacks = get_lr_schedule(full_config, steps=total_steps)
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

            try:
                fit_result = model.fit(
                    ds_train.repeat(),
                    validation_data=ds_test.repeat(),
                    epochs=full_config["setup"]["num_epochs"],
                    callbacks=callbacks,
                    steps_per_epoch=num_train_steps,
                    validation_steps=num_test_steps,
                )
            except tf.errors.ResourceExhaustedError:
                logging.warning("Resource exhausted, skipping this hyperparameter configuration.")
                skiplog_file_path = Path(full_config["raytune"]["local_dir"]) / name / "skipped_configurations.txt"
                lines = ["{}: {}\n".format(item[0], item[1]) for item in config.items()]

                with open(skiplog_file_path, "a") as f:
                    f.write("#"*80 + "\n")
                    for line in lines:
                        f.write(line)
                        logging.warning(line[:-1])
                    f.write("#"*80 + "\n\n")


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
    from raytune.search_space import search_space, raytune_num_samples
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
    shutil.copy("mlpf/raytune/search_space.py", str(Path(cfg["raytune"]["local_dir"]) / name / "search_space.py"))  # Copy the config file to the train dir for later reference
    shutil.copy(config_file_path, str(Path(cfg["raytune"]["local_dir"]) / name / "config.yaml"))  # Copy the config file to the train dir for later reference

    ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 1 * 60 * 60  # Avoid timeout errors
    if not local:
        ray.init(address='auto')

    sched = get_raytune_schedule(cfg["raytune"])
    search_alg = get_raytune_search_alg(cfg["raytune"], seeds)

    sync_config = tune.SyncConfig(sync_to_driver=False)

    start = datetime.now()
    analysis = tune.run(
        partial(build_model_and_train, full_config=config_file_path, ntrain=ntrain, ntest=ntest, name=name, seeds=seeds),
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
    print("Total time of tune.run(...): {}".format(end - start))

    print("Best hyperparameters found according to {} were: ".format(cfg["raytune"]["default_metric"]),
        analysis.get_best_config(cfg["raytune"]["default_metric"], cfg["raytune"]["default_mode"]))

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
    print("Number of skipped configurations: {}".format(num_skipped))


@main.command()
@click.help_option("-h", "--help")
@click.option("-d", "--exp_dir", help="experiment dir", type=click.Path())
def count_skipped(exp_dir):
    num_skipped = count_skipped_configurations(exp_dir)
    print("Number of skipped configurations: {}".format(num_skipped))


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
