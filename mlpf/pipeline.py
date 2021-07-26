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

import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa
import keras_tuner as kt

from tfmodel.data import Dataset
from tfmodel.model_setup import (
    make_model,
    configure_model_weights,
    LearningRateLoggingCallback,
    prepare_callbacks,
    FlattenedCategoricalAccuracy,
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
    targets_multi_output,
    get_dataset_def,
    prepare_val_data,
    set_config_loss,
    get_loss_dict,
    parse_config,
    get_best_checkpoint,
    delete_all_but_best_checkpoint,
)

from tfmodel.lr_finder import LRFinder
from tfmodel.callbacks import CustomTensorBoard
from tfmodel import hypertuning


@click.group()
@click.help_option("-h", "--help")
def main():
    pass


@main.command()
@click.help_option("-h", "--help")
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-w", "--weights", default=None, help="trained weights to load", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
@click.option("-r", "--recreate", help="force creation of new experiment dir", is_flag=True)
@click.option("-p", "--prefix", default="", help="prefix to put at beginning of training dir name", type=str)
def train(config, weights, ntrain, ntest, recreate, prefix):
    """Train a model defined by config"""
    config_file_path = config
    config, config_file_stem, global_batch_size, n_train, n_test, n_epochs, weights = parse_config(
        config, ntrain, ntest, weights
    )

    dataset_def = get_dataset_def(config)
    ds_train_r, ds_test_r, dataset_transform = get_train_val_datasets(config, global_batch_size, n_train, n_test)
    X_val, ygen_val, ycand_val = prepare_val_data(config, dataset_def, single_file=False)

    if recreate or (weights is None):
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
    else:
        outdir = str(Path(weights).parent)
    shutil.copy(config_file_path, outdir + "/config.yaml")  # Copy the config file to the train dir for later reference

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    # If using more than 1 GPU, we scale the batch size by the number of GPUs
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size
    total_steps = n_epochs * n_train // global_batch_size

    with strategy.scope():
        lr_schedule, optim_callbacks = get_lr_schedule(config, steps=total_steps)
        opt = get_optimizer(config, lr_schedule)

        if config["setup"]["dtype"] == "float16":
            model_dtype = tf.dtypes.float16
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            opt = mixed_precision.LossScaleOptimizer(opt)
        else:
            model_dtype = tf.dtypes.float32

        model = make_model(config, model_dtype)

        # Run model once to build the layers
        print(X_val.shape)
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

        callbacks = prepare_callbacks(
            model,
            outdir,
            X_val[: config["setup"]["batch_size"]],
            ycand_val[: config["setup"]["batch_size"]],
            dataset_transform,
            config["dataset"]["num_output_classes"],
        )
        callbacks.append(optim_callbacks)

        fit_result = model.fit(
            ds_train_r,
            validation_data=ds_test_r,
            epochs=initial_epoch + n_epochs,
            callbacks=callbacks,
            steps_per_epoch=n_train // global_batch_size,
            validation_steps=n_test // global_batch_size,
            initial_epoch=initial_epoch,
        )
        history_path = Path(outdir) / "history"
        history_path = str(history_path)
        with open("{}/history.json".format(history_path), "w") as fi:
            json.dump(fit_result.history, fi)
        model.save(outdir + "/model_full", save_format="tf")

        print("Training done.")

        print("Starting evaluation...")
        eval_dir = Path(outdir) / "evaluation"
        eval_dir.mkdir()
        eval_dir = str(eval_dir)
        # TODO: change to use the evaluate() function below instead of eval_model()
        eval_model(X_val, ygen_val, ycand_val, model, config, eval_dir, global_batch_size)
        print("Evaluation done.")

        freeze_model(model, config, outdir)


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
    config, _, global_batch_size, _, _, _, weights = parse_config(config, weights=weights)
    # Switch off multi-output for the evaluation for backwards compatibility
    config["setup"]["multi_output"] = False

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

    dataset_def = get_dataset_def(config)
    X_val, ygen_val, ycand_val = prepare_val_data(config, dataset_def, single_file=False)

    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size

    with strategy.scope():
        model = make_model(config, model_dtype)

        # Evaluate model once to build the layers
        print(X_val.shape)
        model(tf.cast(X_val[:1], model_dtype))

        # need to load the weights in the same trainable configuration as the model was set up
        configure_model_weights(model, config["setup"].get("weights_config", "all"))
        if weights:
            model.load_weights(weights, by_name=True)
        else:
            weights = get_best_checkpoint(train_dir)
            print("Loading best weights that could be found from {}".format(weights))
            model.load_weights(weights, by_name=True)
        model(tf.cast(X_val[:1], model_dtype))

        model.compile()
        eval_model(X_val, ygen_val, ycand_val, model, config, eval_dir, global_batch_size)
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
    config, _, global_batch_size, n_train, _, _, _ = parse_config(config)
    ds_train_r, _, _ = get_train_val_datasets(config, global_batch_size, n_train, n_test=0)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, maybe_global_batch_size = get_strategy(global_batch_size)

    # If using more than 1 GPU, we scale the batch size by the number of GPUs
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size

    dataset_def = get_dataset_def(config)
    X_val, _, _ = prepare_val_data(config, dataset_def, single_file=True)

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
        model(tf.cast(X_val[:1], model_dtype))

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
            ds_train_r,
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
@click.option("-c", "--config", help="configuration file", type=click.Path())
@click.option("-o", "--outdir", help="output dir", type=click.Path())
@click.option("--ntrain", default=None, help="override the number of training events", type=int)
@click.option("--ntest", default=None, help="override the number of testing events", type=int)
@click.option("-r", "--recreate", help="overwrite old hypertune results", is_flag=True, default=False)
def hypertune(config, outdir, ntrain, ntest, recreate):
    config, _, global_batch_size, n_train, n_test, n_epochs, _ = parse_config(config, ntrain, ntest)

    # Override number of epochs with value from Hyperband config
    cfg_hb = config["hypertune"]["hyperband"]
    n_epochs = cfg_hb["max_epochs"]

    ds_train_r, ds_test_r, _ = get_train_val_datasets(config, global_batch_size, n_train, n_test)

    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size
    total_steps = n_epochs * n_train // global_batch_size

    model_builder, optim_callbacks = hypertuning.get_model_builder(config, total_steps)

    tb = CustomTensorBoard(
            log_dir=outdir + "/tensorboard_logs", histogram_freq=1, write_graph=False, write_images=False,
            update_freq=1,
        )
    # Change the class name of CustomTensorBoard TensorBoard to make keras_tuner recognise it
    tb.__class__.__name__ = "TensorBoard"

    tuner = kt.Hyperband(
        model_builder,
        objective=cfg_hb["objective"],
        max_epochs=cfg_hb["max_epochs"],
        factor=cfg_hb["factor"],
        hyperband_iterations=cfg_hb["iterations"],
        directory=outdir + "/tb",
        project_name="mlpf",
        overwrite=recreate,
        executions_per_trial=cfg_hb["executions_per_trial"],
        distribution_strategy=strategy,
    )

    tuner.search(
        ds_train_r,
        epochs=n_epochs,
        validation_data=ds_test_r,
        steps_per_epoch=n_train // global_batch_size,
        validation_steps=n_test // global_batch_size,
        #callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
        callbacks=[tb] + optim_callbacks,
    )

    tuner.results_summary()
    for trial in tuner.oracle.get_best_trials(num_trials=10):
        print(trial.hyperparameters.values, trial.score)


if __name__ == "__main__":
    main()
