import os
import yaml
from pathlib import Path
import datetime
import platform

import tensorflow as tf

from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler


def load_config(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def create_experiment_dir(prefix=None, suffix=None):
    if prefix is None:
        train_dir = Path("experiments") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        train_dir = Path("experiments") / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    if suffix is not None:
        train_dir = train_dir.with_name(train_dir.name + "." + platform.node())

    train_dir.mkdir(parents=True)
    return str(train_dir)


def get_strategy(global_batch_size):
    try:
        gpus = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]
        num_gpus = len(gpus)
        print("num_gpus=", num_gpus)
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()
            global_batch_size = num_gpus * global_batch_size
        else:
            strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    except Exception as e:
        print("fallback to CPU", e)
        strategy = tf.distribute.OneDeviceStrategy("cpu")
        num_gpus = 0
    return strategy, global_batch_size


def get_lr_schedule(config, lr, steps):
    callbacks = []
    schedule = config["setup"]["lr_schedule"]
    if schedule == "onecycle":
        onecycle_cfg = config["onecycle"]
        lr_schedule = OneCycleScheduler(
            lr_max=lr,
            steps=steps,
            mom_min=onecycle_cfg["mom_min"],
            mom_max=onecycle_cfg["mom_max"],
            warmup_ratio=onecycle_cfg["warmup_ratio"],
            div_factor=onecycle_cfg["div_factor"],
            final_div=onecycle_cfg["final_div"],
        )
        callbacks.append(
            MomentumOneCycleScheduler(
                steps=steps,
                mom_min=onecycle_cfg["mom_min"],
                mom_max=onecycle_cfg["mom_max"],
                warmup_ratio=onecycle_cfg["warmup_ratio"],
            )
        )
    elif schedule == "exponentialdecay":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=steps,
            decay_rate=0.99,
            staircase=True,
        )
    return lr_schedule, callbacks


def compute_weights_invsqrt(X, y, w):
    wn = tf.cast(tf.shape(w)[-1], tf.float32) / tf.sqrt(w)
    wn *= tf.cast(X[:, 0] != 0, tf.float32)
    # wn /= tf.reduce_sum(wn)
    return X, y, wn


def compute_weights_none(X, y, w):
    wn = tf.ones_like(w)
    wn *= tf.cast(X[:, 0] != 0, tf.float32)
    return X, y, wn


def get_weights_func(config):
    sampling = config["setup"]["sample_weights"]
    if sampling == "inverse_sqrt":
        return compute_weights_invsqrt
    elif sampling == "none":
        return compute_weights_none
    else:
        raise ValueError("Only supported weight samplings are 'inverse_sqrt' and 'none'.")
