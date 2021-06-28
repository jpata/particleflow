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

import tensorflow as tf
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa

from tfmodel.data import Dataset
from tfmodel.model_setup import (
    targets_multi_output,
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
    create_experiment_dir,
    get_strategy,
    get_weights_func,
    load_config,
    compute_weights_invsqrt,
    compute_weights_none,
)

from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler


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
    """Train a model defined by config
    """
    config_file_stem = Path(config).stem
    config = load_config(config)
    tf.config.run_functions_eagerly(config["tensorflow"]["eager"])

    cds = config["dataset"]
    dataset_def = Dataset(
        num_input_features=int(cds["num_input_features"]),
        num_output_features=int(cds["num_output_features"]),
        padded_num_elem_size=int(cds["padded_num_elem_size"]),
        raw_path=cds.get("raw_path", None),
        raw_files=cds.get("raw_files", None),
        processed_path=cds["processed_path"],
        validation_file_path=cds["validation_file_path"],
        schema=cds["schema"],
    )

    global_batch_size = config["setup"]["batch_size"]
    if "multi_output" in config["setup"]:
        multi_output = config["setup"]["multi_output"]
    else:
        multi_output = True
        config["setup"]["multi_output"] = True
    n_train = config["setup"]["num_events_train"]
    n_test = config["setup"]["num_events_test"]
    if ntrain:
        n_train = ntrain
    if ntest:
        n_test = ntest

    n_epochs = config["setup"]["num_epochs"]
    total_steps = n_epochs * n_train // global_batch_size

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(dataset_def.processed_path))

    random.shuffle(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(
        dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Due to TFRecords format, the length of the dataset is not known beforehand
    num_events = 0
    for i in dataset:
        num_events += 1
    print("dataset loaded, len={}".format(num_events))

    weight_func = get_weights_func(config)
    assert n_train + n_test <= num_events

    # Padded shapes
    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        tf.TensorShape(
            [
                dataset_def.padded_num_elem_size,
            ]
        ),
    )

    ds_train = dataset.take(n_train).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)

    if multi_output:
        dataset_transform = targets_multi_output(config["dataset"]["num_output_classes"])
        ds_train = ds_train.map(dataset_transform)
        ds_test = ds_test.map(dataset_transform)
    else:
        dataset_transform = None

    ds_train_r = ds_train.repeat(n_epochs)
    ds_test_r = ds_test.repeat(n_epochs)

    if weights is None:
        weights = config["setup"]["weights"]

    if recreate or (weights is None):
        outdir = create_experiment_dir(prefix=prefix + config_file_stem + "_", suffix=platform.node())
    else:
        outdir = str(Path(weights).parent)

    # Decide tf.distribute.strategy depending on number of available GPUs
    strategy, maybe_global_batch_size = get_strategy(global_batch_size)

    # If using more than 1 GPU, we scale the batch size by the number of GPUs
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size
    actual_lr = global_batch_size * float(config["setup"]["lr"])

    val_filelist = dataset_def.val_filelist
    if config["setup"]["num_val_files"] > 0:
        val_filelist = val_filelist[: config["setup"]["num_val_files"]]

    Xs = []
    ygens = []
    ycands = []
    for fi in tqdm(val_filelist[:1], desc="Preparing validation data"):
        X, ygen, ycand = dataset_def.prepare_data(fi)
        Xs.append(np.concatenate(X))
        ygens.append(np.concatenate(ygen))
        ycands.append(np.concatenate(ycand))

    assert len(Xs) > 0, "Xs is empty"
    X_val = np.concatenate(Xs)
    ygen_val = np.concatenate(ygens)
    ycand_val = np.concatenate(ycands)

    with strategy.scope():
        lr_schedule, optim_callbacks = get_lr_schedule(config, lr=actual_lr, steps=total_steps)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
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
        model(tf.cast(X_val[:1], model_dtype))

        initial_epoch = 0
        if weights:
            # We need to load the weights in the same trainable configuration as the model was set up
            configure_model_weights(model, config["setup"].get("weights_config", "all"))
            model.load_weights(weights, by_name=True)
            initial_epoch = int(weights.split("/")[-1].split("-")[1])
        model(tf.cast(X_val[:1], model_dtype))

        if config["setup"]["trainable"] == "classification":
            config["dataset"]["pt_loss_coef"] = 0.0
            config["dataset"]["eta_loss_coef"] = 0.0
            config["dataset"]["sin_phi_loss_coef"] = 0.0
            config["dataset"]["cos_phi_loss_coef"] = 0.0
            config["dataset"]["energy_loss_coef"] = 0.0
        elif config["setup"]["trainable"] == "regression":
            config["dataset"]["classification_loss_coef"] = 0.0
            config["dataset"]["charge_loss_coef"] = 0.0

        configure_model_weights(model, config["setup"]["trainable"])
        model(tf.cast(X_val[:1], model_dtype))

        if config["setup"]["classification_loss_type"] == "categorical_cross_entropy":
            cls_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        elif config["setup"]["classification_loss_type"] == "sigmoid_focal_crossentropy":
            cls_loss = tfa.losses.sigmoid_focal_crossentropy
        else:
            raise KeyError("Unknown classification loss type: {}".format(config["setup"]["classification_loss_type"]))

        model.compile(
            loss={
                "cls": cls_loss,
                "charge": getattr(tf.keras.losses, config["dataset"].get("charge_loss", "MeanSquaredError"))(),
                "pt": getattr(tf.keras.losses, config["dataset"].get("pt_loss", "MeanSquaredError"))(),
                "eta": getattr(tf.keras.losses, config["dataset"].get("eta_loss", "MeanSquaredError"))(),
                "sin_phi": getattr(tf.keras.losses, config["dataset"].get("sin_phi_loss", "MeanSquaredError"))(),
                "cos_phi": getattr(tf.keras.losses, config["dataset"].get("cos_phi_loss", "MeanSquaredError"))(),
                "energy": getattr(tf.keras.losses, config["dataset"].get("energy_loss", "MeanSquaredError"))(),
            },
            optimizer=opt,
            sample_weight_mode="temporal",
            loss_weights={
                "cls": config["dataset"]["classification_loss_coef"],
                "charge": config["dataset"]["charge_loss_coef"],
                "pt": config["dataset"]["pt_loss_coef"],
                "eta": config["dataset"]["eta_loss_coef"],
                "sin_phi": config["dataset"]["sin_phi_loss_coef"],
                "cos_phi": config["dataset"]["cos_phi_loss_coef"],
                "energy": config["dataset"]["energy_loss_coef"],
            },
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
        callbacks.append(LearningRateLoggingCallback())
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
@click.option("-e", "--evaluation_dir", help="force creation of new experiment dir", type=click.Path())
def evaluate(config, train_dir, weights, evaluation_dir):
    config = load_config(config)
    # Switch off multi-output for the evaluation for backwards compatibility
    config["setup"]["multi_output"] = False

    if evaluation_dir is None:
        eval_dir = str(Path(train_dir) / "evaluation")
    else:
        eval_dir = evaluation_dir
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    tf.config.run_functions_eagerly(config["tensorflow"]["eager"])

    if weights is None:
        weights = config["setup"]["weights"]

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    cds = config["dataset"]
    dataset_def = Dataset(
        num_input_features=int(cds["num_input_features"]),
        num_output_features=int(cds["num_output_features"]),
        padded_num_elem_size=int(cds["padded_num_elem_size"]),
        raw_path=cds.get("raw_path", None),
        raw_files=cds.get("raw_files", None),
        processed_path=cds["processed_path"],
        validation_file_path=cds["validation_file_path"],
        schema=cds["schema"],
    )

    Xs = []
    ygens = []
    ycands = []
    val_filelist = dataset_def.val_filelist
    if config["setup"]["num_val_files"] > 0:
        val_filelist = val_filelist[: config["setup"]["num_val_files"]]

    for fi in tqdm(val_filelist, desc="Preparing validation data"):
        X, ygen, ycand = dataset_def.prepare_data(fi)
        Xs.append(np.concatenate(X))
        ygens.append(np.concatenate(ygen))
        ycands.append(np.concatenate(ycand))
    assert len(Xs) > 0
    X_val = np.concatenate(Xs)
    ygen_val = np.concatenate(ygens)
    ycand_val = np.concatenate(ycands)

    global_batch_size = config["setup"]["batch_size"]

    strategy, maybe_global_batch_size = get_strategy(global_batch_size)
    if maybe_global_batch_size is not None:
        global_batch_size = maybe_global_batch_size

    with strategy.scope():

        model = make_model(config, model_dtype)

        # Evaluate model once to build the layers
        print(X_val.shape)
        model(tf.cast(X_val[:1], model_dtype))

        if weights:
            # need to load the weights in the same trainable configuration as the model was set up
            configure_model_weights(model, config["setup"].get("weights_config", "all"))
            model.load_weights(weights, by_name=True)
        model(tf.cast(X_val[:1], model_dtype))

        model.compile()
        eval_model(X_val, ygen_val, ycand_val, model, config, eval_dir, global_batch_size)
        freeze_model(model, config, eval_dir)


if __name__ == "__main__":
    main()
