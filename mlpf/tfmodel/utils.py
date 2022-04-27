import os
import yaml
from pathlib import Path
import datetime
import platform
import random
import glob
import numpy as np
from tqdm import tqdm
import re
import logging

import tensorflow as tf
import tensorflow_addons as tfa

from tfmodel.data import Dataset
from tfmodel.onecycle_scheduler import OneCycleScheduler, MomentumOneCycleScheduler
from tfmodel.datasets import CMSDatasetFactory, DelphesDatasetFactory


def load_config(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def parse_config(config, ntrain=None, ntest=None, nepochs=None, weights=None):
    config_file_stem = Path(config).stem
    config = load_config(config)

    tf.config.run_functions_eagerly(config["tensorflow"]["eager"])
    n_epochs = config["setup"]["num_epochs"]
    
    if ntrain:
        config["setup"]["num_events_train"] = ntrain

    if ntest:
        config["setup"]["num_events_test"] = ntest

    if nepochs:
        config["setup"]["num_epochs"] = nepochs

    if "multi_output" not in config["setup"]:
        config["setup"]["multi_output"] = True

    if weights is None:
        config["setup"]["weights"] = weights

    return config, config_file_stem


def create_experiment_dir(prefix=None, suffix=None):
    if prefix is None:
        train_dir = Path("experiments") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        train_dir = Path("experiments") / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    if suffix is not None:
        train_dir = train_dir.with_name(train_dir.name + "." + platform.node())

    train_dir.mkdir(parents=True)
    print("Creating experiment dir {}".format(train_dir))
    return str(train_dir)


def get_best_checkpoint(train_dir):
    checkpoint_list = list(Path(Path(train_dir) / "weights").glob("weights*.hdf5"))
    # Sort the checkpoints according to the loss in their filenames
    checkpoint_list.sort(key=lambda x: float(re.search("\d+-\d+.\d+", str(x))[0].split("-")[-1]))
    # Return the checkpoint with smallest loss
    return str(checkpoint_list[0])


def delete_all_but_best_checkpoint(train_dir, dry_run):
    checkpoint_list = list(Path(Path(train_dir) / "weights").glob("weights*.hdf5"))
    # Don't remove the checkpoint with smallest loss
    if len(checkpoint_list) == 1:
        raise UserWarning("There is only one checkpoint. No deletion was made.")
    elif len(checkpoint_list) == 0:
        raise UserWarning("Couldn't find any checkpoints. No deletion was made.")
    else:
        # Sort the checkpoints according to the loss in their filenames
        checkpoint_list.sort(key=lambda x: float(re.search("\d+-\d+.\d+", str(x))[0].split("-")[-1]))
        best_ckpt = checkpoint_list.pop(0)
        for ckpt in checkpoint_list:
            if not dry_run:
                ckpt.unlink()

        print("Removed all checkpoints in {} except {}".format(train_dir, best_ckpt))


def get_strategy():
    if isinstance(os.environ.get("CUDA_VISIBLE_DEVICES"), type(None)) or len(os.environ.get("CUDA_VISIBLE_DEVICES")) == 0:
        gpus = [-1]
        print("WARNING: CUDA_VISIBLE_DEVICES variable is empty. \
            If you don't have or intend to use GPUs, this message can be ignored.")
    else:
        gpus = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "-1").split(",")]
    if gpus[0] == -1:
        num_gpus = 0
    else:
        num_gpus = len(gpus)
    print("num_gpus=", num_gpus)
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif num_gpus == 1:
        strategy = tf.distribute.OneDeviceStrategy("gpu:0")
    elif num_gpus == 0:
        print("fallback to CPU")
        strategy = tf.distribute.OneDeviceStrategy("cpu")
        num_gpus = 0
    return strategy, num_gpus


def get_lr_schedule(config, steps):
    lr = float(config["setup"]["lr"])
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
        if config["exponentialdecay"]["decay_steps"] == "epoch":
            decay_steps = int(steps / config["setup"]["num_epochs"])
        else:
            decay_steps = config["exponentialdecay"]["decay_steps"]
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=decay_steps,
            decay_rate=config["exponentialdecay"]["decay_rate"],
            staircase=config["exponentialdecay"]["staircase"],
        )
    elif schedule == "cosinedecay":
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=steps,
        )
    else:
        lr_schedule = None
        callbacks = []
    return lr_schedule, callbacks


def get_optimizer(config, lr_schedule=None):
    if lr_schedule is None:
        lr = float(config["setup"]["lr"])
    else:
        lr = lr_schedule

    if config["setup"]["optimizer"] == "adam":
        cfg_adam = config["optimizer"]["adam"]
        opt = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=cfg_adam["amsgrad"])
        if cfg_adam["pcgrad"]:
            from tfmodel.PCGrad_tf import PCGrad
            opt = PCGrad(opt)
        return opt
    if config["setup"]["optimizer"] == "adamw":
        cfg_adamw = config["optimizer"]["adamw"]
        return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=cfg_adamw["weight_decay"], amsgrad=cfg_adamw["amsgrad"])
    elif config["setup"]["optimizer"] == "sgd":
        cfg_sgd = config["optimizer"]["sgd"]
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=cfg_sgd["momentum"], nesterov=cfg_sgd["nesterov"])
    else:
        raise ValueError("Only 'adam', 'adamw' and 'sgd' are supported optimizers, got {}".format(config["setup"]["optimizer"]))


def get_tuner(cfg_hypertune, model_builder, outdir, recreate, strategy):
    import keras_tuner as kt
    if cfg_hypertune["algorithm"] == "random":
        print("Keras Tuner: Using RandomSearch")
        cfg_rand = cfg_hypertune["random"]
        return kt.RandomSearch(
            model_builder,
            objective=cfg_rand["objective"],
            max_trials=cfg_rand["max_trials"],
            project_name=outdir,
            overwrite=recreate,
        )
    elif cfg_hypertune["algorithm"] == "bayesian":
        print("Keras Tuner: Using BayesianOptimization")
        cfg_bayes = cfg_hypertune["bayesian"]
        return kt.BayesianOptimization(
            model_builder,
            objective=cfg_bayes["objective"],
            max_trials=cfg_bayes["max_trials"],
            num_initial_points=cfg_bayes["num_initial_points"],
            project_name=outdir,
            overwrite=recreate,
        )
    elif cfg_hypertune["algorithm"] == "hyperband":
        print("Keras Tuner: Using Hyperband")
        cfg_hb = cfg_hypertune["hyperband"]
        return kt.Hyperband(
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


def compute_weights_invsqrt(X, y, w):
    wn = tf.cast(tf.shape(w)[-1], tf.float32) / tf.sqrt(w)
    wn *= tf.cast(X[:, 0] != 0, tf.float32)
    # wn /= tf.reduce_sum(wn)
    return X, y, wn


def compute_weights_none(X, y, w):
    wn = tf.ones_like(w)
    wn *= tf.cast(X[:, 0] != 0, tf.float32)
    return X, y, wn


def make_weight_function(config):
    def weight_func(X,y,w):

        w_signal_only = tf.where(y[:, 0]==0, 0.0, 1.0)
        w_signal_only *= tf.cast(X[:, 0]!=0, tf.float32)

        w_none = tf.ones_like(w)
        w_none *= tf.cast(X[:, 0]!=0, tf.float32)

        w_invsqrt = tf.cast(tf.shape(w)[-1], tf.float32)/tf.sqrt(w)
        w_invsqrt *= tf.cast(X[:, 0]!=0, tf.float32)

        w_signal_only_invsqrt = tf.where(y[:, 0]==0, 0.0, tf.cast(tf.shape(w)[-1], tf.float32)/tf.sqrt(w))
        w_signal_only_invsqrt *= tf.cast(X[:, 0]!=0, tf.float32)

        weight_d = {
            "none": w_none,
            "signal_only": w_signal_only,
            "signal_only_inverse_sqrt": w_signal_only_invsqrt,
            "inverse_sqrt": w_invsqrt
        }

        ret_w = {}
        for loss_component, weight_type in config["sample_weights"].items():
            ret_w[loss_component] = weight_d[weight_type]

        return X,y,ret_w
    return weight_func


def targets_multi_output(num_output_classes):
    def func(X, y, w):

        msk = tf.expand_dims(tf.cast(y[:, :, 0]!=0, tf.float32), axis=-1)
        return (
            X,
            {
                "cls": tf.one_hot(tf.cast(y[:, :, 0], tf.int32), num_output_classes),
                "charge": y[:, :, 1:2]*msk,
                "pt": y[:, :, 2:3]*msk,
                "eta": y[:, :, 3:4]*msk,
                "sin_phi": y[:, :, 4:5]*msk,
                "cos_phi": y[:, :, 5:6]*msk,
                "energy": y[:, :, 6:7]*msk,
            },
            w,
        )

    return func

def get_dataset_def(config):
    cds = config["dataset"]

    return Dataset(
        num_input_features=int(cds["num_input_features"]),
        num_output_features=int(cds["num_output_features"]),
        padded_num_elem_size=int(cds["padded_num_elem_size"]),
        schema=cds["schema"],
    )


def get_train_val_datasets(config, global_batch_size, n_train, n_test, repeat=True):
    dataset_def = get_dataset_def(config)

    tfr_files = sorted(glob.glob(dataset_def.processed_path))
    if len(tfr_files) == 0:
        raise Exception("Could not find any files in {}".format(dataset_def.processed_path))

    random.shuffle(tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files).map(
        dataset_def.parse_tfr_element, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Due to TFRecords format, the length of the dataset is not known beforehand
    num_events = 0
    for _ in dataset:
        num_events += 1
    print("dataset loaded, len={}".format(num_events))

    weight_func = make_weight_function(config)
    assert n_train + n_test <= num_events

    # Padded shapes
    ps = (
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_input_features]),
        tf.TensorShape([dataset_def.padded_num_elem_size, dataset_def.num_output_features]),
        {
            "cls": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "charge": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "energy": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "pt": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "eta": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "sin_phi": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
            "cos_phi": tf.TensorShape([dataset_def.padded_num_elem_size, ]),
        }
    )

    ds_train = dataset.take(n_train).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)
    ds_test = dataset.skip(n_train).take(n_test).map(weight_func).padded_batch(global_batch_size, padded_shapes=ps)

    if config["setup"]["multi_output"]:
        dataset_transform = targets_multi_output(config["dataset"]["num_output_classes"])
        ds_train = ds_train.map(dataset_transform)
        ds_test = ds_test.map(dataset_transform)
    else:
        dataset_transform = None

    return ds_train, ds_test, dataset_transform

def prepare_val_data(config, dataset_def, single_file=False):
    if single_file:
        val_filelist = dataset_def.val_filelist[:1]
    else:
        val_filelist = dataset_def.val_filelist
    if config["setup"]["num_val_files"] > 0:
        val_filelist = val_filelist[: config["setup"]["num_val_files"]]

    Xs = []
    ygens = []
    ycands = []
    for fi in tqdm(val_filelist, desc="Preparing validation data"):
        X, ygen, ycand = dataset_def.prepare_data(fi)
        Xs.append(np.concatenate(X))
        ygens.append(np.concatenate(ygen))
        ycands.append(np.concatenate(ycand))

    assert len(Xs) > 0, "Xs is empty"
    X_val = np.concatenate(Xs)
    ygen_val = np.concatenate(ygens)
    ycand_val = np.concatenate(ycands)

    return X_val, ygen_val, ycand_val


def get_heptfds_dataset(dataset_name, config, num_gpus, split, num_events=None, supervised=True):
    cds = config["dataset"]

    if cds['schema'] == "cms":
        dsf = CMSDatasetFactory(config)
    elif cds['schema'] == "delphes":
        dsf = DelphesDatasetFactory(config)
    else:
        raise ValueError("Only supported datasets are 'cms' and 'delphes'.")

    ds, ds_info = dsf.get_dataset(dataset_name, config["datasets"][dataset_name], split)

    if not (num_events is None):
        ds = ds.take(num_events)

    if supervised:
        ds = ds.map(dsf.get_map_to_supervised())

    return ds, ds_info

def load_and_interleave(dataset_names, config, num_gpus, split, batch_size):
    datasets = []
    steps = []
    for ds_name in dataset_names:
        ds, _ = get_heptfds_dataset(ds_name, config, num_gpus, split)
        num_steps = ds.cardinality().numpy()
        assert(num_steps > 0)
        print("Loaded {}:{} with {} steps".format(ds_name, split, num_steps))

        datasets.append(ds)
        steps.append(num_steps)

    #Now interleave elements from the datasets randomly
    ids = 0
    indices = []
    for ds, num_steps in zip(datasets, steps):
        indices += num_steps*[ids]
        ids += 1
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)

    ds = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    bs = batch_size
    if num_gpus>1:
        bs = bs*num_gpus
    ds = ds.batch(bs)
    return ds

#Load multiple datasets and mix them together
def get_datasets(datasets_to_interleave, config, num_gpus, split):
    datasets = []
    steps = []
    for joint_dataset_name in datasets_to_interleave.keys():
        ds_conf = datasets_to_interleave[joint_dataset_name]
        if ds_conf["datasets"] is None:
            logging.warning("No datasets in {} list.".format(joint_dataset_name))
        else:
            interleaved_ds = load_and_interleave(ds_conf["datasets"], config, num_gpus, split, ds_conf["batch_per_gpu"])
            num_steps = 0
            for elem in interleaved_ds:
                num_steps += 1
            print("Interleaved joint dataset {} with {} steps".format(joint_dataset_name, num_steps))
            datasets.append(interleaved_ds)
            steps.append(num_steps)
    
    ids = 0
    indices = []
    for ds, num_steps in zip(datasets, steps):
        indices += num_steps*[ids]
        ids += 1
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)
    ds = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    num_steps = 0
    for elem in ds:
        num_steps += 1

    print("Final dataset with {} steps".format(num_steps))
    return ds, num_steps

def set_config_loss(config, trainable):
    if trainable == "classification":
        config["dataset"]["pt_loss_coef"] = 0.0
        config["dataset"]["eta_loss_coef"] = 0.0
        config["dataset"]["sin_phi_loss_coef"] = 0.0
        config["dataset"]["cos_phi_loss_coef"] = 0.0
        config["dataset"]["energy_loss_coef"] = 0.0
    elif trainable == "regression":
        config["dataset"]["classification_loss_coef"] = 0.0
        config["dataset"]["charge_loss_coef"] = 0.0
        config["dataset"]["pt_loss_coef"] = 0.0
        config["dataset"]["eta_loss_coef"] = 0.0
        config["dataset"]["sin_phi_loss_coef"] = 0.0
        config["dataset"]["cos_phi_loss_coef"] = 0.0
    elif trainable == "all":
        pass
    return config


def get_class_loss(config):
    if config["setup"]["classification_loss_type"] == "categorical_cross_entropy":
        cls_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=config["setup"].get("classification_label_smoothing", 0.0))
    elif config["setup"]["classification_loss_type"] == "sigmoid_focal_crossentropy":
        cls_loss = tfa.losses.sigmoid_focal_crossentropy
    else:
        raise KeyError("Unknown classification loss type: {}".format(config["setup"]["classification_loss_type"]))
    return cls_loss


def get_loss_from_params(input_dict):
    input_dict = input_dict.copy()
    loss_type = input_dict.pop("type")
    loss_cls = getattr(tf.keras.losses, loss_type)
    return loss_cls(**input_dict)

# class MyLoss(tf.keras.losses.Loss):
#   def call(self, y_true, y_pred):
#       import pdb;pdb.set_trace()
#       return tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)


def get_loss_dict(config):
    cls_loss = get_class_loss(config)

    default_loss = {"type": "MeanSquaredError"}
    loss_dict = {
        "cls": cls_loss,
        "charge": get_loss_from_params(config["dataset"].get("charge_loss", default_loss)),
        "pt": get_loss_from_params(config["dataset"].get("pt_loss", default_loss)),
        "eta": get_loss_from_params(config["dataset"].get("eta_loss", default_loss)),
        "sin_phi": get_loss_from_params(config["dataset"].get("sin_phi_loss", default_loss)),
        "cos_phi": get_loss_from_params(config["dataset"].get("cos_phi_loss", default_loss)),
        "energy": get_loss_from_params(config["dataset"].get("energy_loss", default_loss)),
        "sum_energy": tf.keras.losses.MeanSquaredError(),
        "sum_pt": tf.keras.losses.MeanSquaredError(),
    }
    loss_weights = {
        "cls": config["dataset"]["classification_loss_coef"],
        "charge": config["dataset"]["charge_loss_coef"],
        "pt": config["dataset"]["pt_loss_coef"],
        "eta": config["dataset"]["eta_loss_coef"],
        "sin_phi": config["dataset"]["sin_phi_loss_coef"],
        "cos_phi": config["dataset"]["cos_phi_loss_coef"],
        "energy": config["dataset"]["energy_loss_coef"],
        "sum_energy": config["dataset"]["sum_energy_loss_coef"],
        "sum_pt": config["dataset"]["sum_pt_loss_coef"],
    }
    return loss_dict, loss_weights
