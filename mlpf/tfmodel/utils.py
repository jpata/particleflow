import datetime
import logging
import os
import platform
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from tfmodel.datasets import CMSDatasetFactory, DelphesDatasetFactory
from tfmodel.onecycle_scheduler import MomentumOneCycleScheduler, OneCycleScheduler


@tf.function
def histogram_2d(mask, eta, phi, weights_px, weights_py, eta_range, phi_range, nbins, bin_dtype=tf.float32):
    eta_bins = tf.histogram_fixed_width_bins(eta, eta_range, nbins=nbins, dtype=bin_dtype)
    phi_bins = tf.histogram_fixed_width_bins(phi, phi_range, nbins=nbins, dtype=bin_dtype)

    # create empty histograms
    hist_px = tf.zeros((nbins, nbins), dtype=weights_px.dtype)
    hist_py = tf.zeros((nbins, nbins), dtype=weights_py.dtype)
    indices = tf.transpose(tf.stack([eta_bins, phi_bins]))

    indices_masked = tf.boolean_mask(indices, mask)
    weights_px_masked = tf.boolean_mask(weights_px, mask)
    weights_py_masked = tf.boolean_mask(weights_py, mask)

    hist_px = tf.tensor_scatter_nd_add(hist_px, indices=indices_masked, updates=weights_px_masked)
    hist_py = tf.tensor_scatter_nd_add(hist_py, indices=indices_masked, updates=weights_py_masked)
    hist_pt = tf.sqrt(hist_px**2 + hist_py**2)
    return hist_pt


@tf.function
def batched_histogram_2d(mask, eta, phi, w_px, w_py, x_range, y_range, nbins, bin_dtype=tf.float32):
    return tf.map_fn(
        lambda a: histogram_2d(a[0], a[1], a[2], a[3], a[4], x_range, y_range, nbins, bin_dtype),
        (mask, eta, phi, w_px, w_py),
        fn_output_signature=tf.TensorSpec(
            [nbins, nbins],
            dtype=tf.float32,
        ),
    )


def load_config(config_file_path):
    with open(config_file_path, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def parse_config(config, ntrain=None, ntest=None, nepochs=None, weights=None):
    config_file_stem = Path(config).stem
    config = load_config(config)

    tf.config.run_functions_eagerly(config["tensorflow"]["eager"])

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
        print(
            "WARNING: CUDA_VISIBLE_DEVICES variable is empty. \
            If you don't have or intend to use GPUs, this message can be ignored."
        )
    else:
        gpus = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "-1").split(",")]
    if gpus[0] == -1:
        num_gpus = 0
    else:
        num_gpus = len(gpus)
    print("num_gpus=", num_gpus)
    if num_gpus > 1:
        print("Using tf.distribute.MirroredStrategy()")
        strategy = tf.distribute.MirroredStrategy()
    elif num_gpus == 1:
        print("Using tf.distribute.OneDeviceStrategy('gpu:0')")
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
    return lr_schedule, callbacks, lr


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
        raise ValueError(
            "Only 'adam', 'adamw' and 'sgd' are supported optimizers, got {}".format(config["setup"]["optimizer"])
        )


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


def targets_multi_output(num_output_classes):
    def func(X, y, w):

        msk = tf.expand_dims(tf.cast(y[:, :, 0] != 0, tf.float32), axis=-1)
        return (
            X,
            {
                "cls": tf.one_hot(tf.cast(y[:, :, 0], tf.int32), num_output_classes),
                "charge": y[:, :, 1:2] * msk,
                "pt": y[:, :, 2:3] * msk,
                "eta": y[:, :, 3:4] * msk,
                "sin_phi": y[:, :, 4:5] * msk,
                "cos_phi": y[:, :, 5:6] * msk,
                "energy": y[:, :, 6:7] * msk,
            },
            w,
        )

    return func


def get_heptfds_dataset(dataset_name, config, num_gpus, split, num_events=None, supervised=True):
    cds = config["dataset"]

    if cds["schema"] == "cms":
        dsf = CMSDatasetFactory(config)
    elif cds["schema"] == "delphes":
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
    total_num_steps = 0
    for ds_name in dataset_names:
        ds, _ = get_heptfds_dataset(ds_name, config, num_gpus, split)
        num_steps = ds.cardinality().numpy()
        total_num_steps += num_steps
        assert num_steps > 0
        print("Loaded {}:{} with {} steps".format(ds_name, split, num_steps))

        datasets.append(ds)
        steps.append(num_steps)

    # Now interleave elements from the datasets randomly
    ids = 0
    indices = []
    for ds, num_steps in zip(datasets, steps):
        indices += num_steps * [ids]
        ids += 1
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)

    ds = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    bs = batch_size
    if not config["setup"]["horovod_enabled"]:
        if num_gpus > 1:
            bs = bs * num_gpus
    ds = ds.batch(bs)

    total_num_steps = total_num_steps // bs
    # num_steps = 0
    # for _ in ds:
    #    num_steps += 1
    # assert(total_num_steps == num_steps)
    return ds, total_num_steps


# Load multiple datasets and mix them together
def get_datasets(datasets_to_interleave, config, num_gpus, split):
    datasets = []
    steps = []
    for joint_dataset_name in datasets_to_interleave.keys():
        ds_conf = datasets_to_interleave[joint_dataset_name]
        if ds_conf["datasets"] is None:
            logging.warning("No datasets in {} list.".format(joint_dataset_name))
        else:
            interleaved_ds, num_steps = load_and_interleave(
                ds_conf["datasets"], config, num_gpus, split, ds_conf["batch_per_gpu"]
            )
            print("Interleaved joint dataset {} with {} steps".format(joint_dataset_name, num_steps))
            datasets.append(interleaved_ds)
            steps.append(num_steps)

    ids = 0
    indices = []
    total_num_steps = 0
    for ds, num_steps in zip(datasets, steps):
        indices += num_steps * [ids]
        total_num_steps += num_steps
        ids += 1
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)
    ds = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)
    # num_steps = 0
    # for elem in ds:
    #    num_steps += 1
    # assert(total_num_steps == num_steps)

    print("Final dataset with {} steps".format(total_num_steps))
    return ds, total_num_steps


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
        cls_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, label_smoothing=config["setup"].get("classification_label_smoothing", 0.0)
        )
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


# batched version of https://github.com/VinAIResearch/DSW/blob/master/gsw.py#L19
@tf.function
def sliced_wasserstein_loss(y_true, y_pred, num_projections=1000):

    # take everything but the jet_idx
    y_true = y_true[..., :5]
    y_pred = y_pred[..., :5]

    # create normalized random basis vectors
    theta = tf.random.normal((num_projections, y_true.shape[-1]))
    theta = theta / tf.sqrt(tf.reduce_sum(theta**2, axis=1, keepdims=True))

    # project the features with the random basis
    A = tf.linalg.matmul(y_true, theta, False, True)
    B = tf.linalg.matmul(y_pred, theta, False, True)

    A_sorted = tf.sort(A, axis=-2)
    B_sorted = tf.sort(B, axis=-2)

    ret = tf.math.sqrt(tf.reduce_sum(tf.math.pow(A_sorted - B_sorted, 2), axis=[-1, -2]))
    return ret


@tf.function
def hist_2d_loss(y_true, y_pred):

    eta_true = y_true[..., 2]
    eta_pred = y_pred[..., 2]

    sin_phi_true = y_true[..., 3]
    sin_phi_pred = y_pred[..., 3]

    pt_true = y_true[..., 0]
    pt_pred = y_pred[..., 0]

    px_true = pt_true * y_true[..., 4]
    py_true = pt_true * y_true[..., 3]
    px_pred = pt_pred * y_pred[..., 4]
    py_pred = pt_pred * y_pred[..., 3]

    mask = eta_true != 0.0

    # bin in (eta, sin_phi), as calculating phi=atan2(sin_phi, cos_phi)
    # introduces a numerical instability which can lead to NaN.
    pt_hist_true = batched_histogram_2d(
        mask,
        eta_true,
        sin_phi_true,
        px_true,
        py_true,
        tf.cast([-6.0, 6.0], tf.float32),
        tf.cast([-1.0, 1.0], tf.float32),
        20,
    )

    pt_hist_pred = batched_histogram_2d(
        mask,
        eta_pred,
        sin_phi_pred,
        px_pred,
        py_pred,
        tf.cast([-6.0, 6.0], tf.float32),
        tf.cast([-1.0, 1.0], tf.float32),
        20,
    )

    mse = tf.math.sqrt(tf.reduce_mean((pt_hist_true - pt_hist_pred) ** 2, axis=[-1, -2]))
    return mse


@tf.function
def jet_reco(px, py, jet_idx, max_jets):

    tf.debugging.assert_shapes(
        [
            (px, ("N")),
            (py, ("N")),
            (jet_idx, ("N")),
        ]
    )

    jet_idx_capped = tf.where(jet_idx <= max_jets, jet_idx, 0)

    jet_px = tf.zeros(
        [
            max_jets,
        ],
        dtype=px.dtype,
    )
    jet_py = tf.zeros(
        [
            max_jets,
        ],
        dtype=py.dtype,
    )

    jet_px_new = tf.tensor_scatter_nd_add(jet_px, indices=tf.expand_dims(jet_idx_capped, axis=-1), updates=px)
    jet_py_new = tf.tensor_scatter_nd_add(jet_py, indices=tf.expand_dims(jet_idx_capped, axis=-1), updates=py)

    jet_pt = tf.math.sqrt(jet_px_new**2 + jet_py_new**2)

    return jet_pt


@tf.function
def batched_jet_reco(px, py, jet_idx, max_jets):
    tf.debugging.assert_shapes(
        [
            (px, ("B", "N")),
            (py, ("B", "N")),
            (jet_idx, ("B", "N")),
        ]
    )

    return tf.map_fn(
        lambda a: jet_reco(a[0], a[1], a[2], max_jets),
        (px, py, jet_idx),
        fn_output_signature=tf.TensorSpec(
            [
                max_jets,
            ],
            dtype=tf.float32,
        ),
    )


#y_true: [nbatch, nptcl, 5] array of true particle properties.
#y_pred: [nbatch, nptcl, 5] array of predicted particle properties
#last dim corresponds to [pt, energy, eta, sin_phi, cos_phi, gen_jet_idx]
#max_jets: integer of the max number of jets to consider
#returns: dict of true and predicted jet pts.  
@tf.function
def compute_jet_pt(y_true, y_pred, max_jets=201):
    y = {}
    y["true"] = y_true
    y["pred"] = y_pred
    jet_pt = {}

    jet_idx = tf.cast(y["true"][..., 5], dtype=tf.int32)
    for typ in ["true", "pred"]:
        px = y[typ][..., 0] * y[typ][..., 4]
        py = y[typ][..., 0] * y[typ][..., 3]
        jet_pt[typ] = batched_jet_reco(px, py, jet_idx, max_jets)
    return jet_pt


@tf.function
def gen_jet_mse_loss(y_true, y_pred):

    jet_pt = compute_jet_pt(y_true, y_pred)
    mse = tf.math.sqrt(tf.reduce_mean((jet_pt["true"] - jet_pt["pred"]) ** 2, axis=[-1, -2]))
    return mse


@tf.function
def gen_jet_logcosh_loss(y_true, y_pred):

    jet_pt = compute_jet_pt(y_true, y_pred)
    loss = tf.keras.losses.log_cosh(jet_pt["true"], jet_pt["pred"])
    return loss


def get_loss_dict(config):
    cls_loss = get_class_loss(config)

    default_loss = {"type": "MeanSquaredError"}
    loss_dict = {
        "cls": cls_loss,
        "charge": get_loss_from_params(config["loss"].get("charge_loss", default_loss)),
        "pt": get_loss_from_params(config["loss"].get("pt_loss", default_loss)),
        "eta": get_loss_from_params(config["loss"].get("eta_loss", default_loss)),
        "sin_phi": get_loss_from_params(config["loss"].get("sin_phi_loss", default_loss)),
        "cos_phi": get_loss_from_params(config["loss"].get("cos_phi_loss", default_loss)),
        "energy": get_loss_from_params(config["loss"].get("energy_loss", default_loss)),
    }
    loss_weights = {
        "cls": config["loss"]["classification_loss_coef"],
        "charge": config["loss"]["charge_loss_coef"],
        "pt": config["loss"]["pt_loss_coef"],
        "eta": config["loss"]["eta_loss_coef"],
        "sin_phi": config["loss"]["sin_phi_loss_coef"],
        "cos_phi": config["loss"]["cos_phi_loss_coef"],
        "energy": config["loss"]["energy_loss_coef"],
    }

    if config["loss"]["event_loss"] != "none":
        loss_weights["pt_e_eta_phi"] = config["loss"]["event_loss_coef"]

    if config["loss"]["event_loss"] == "sliced_wasserstein":
        loss_dict["pt_e_eta_phi"] = sliced_wasserstein_loss

    if config["loss"]["event_loss"] == "hist_2d":
        loss_dict["pt_e_eta_phi"] = hist_2d_loss

    if config["loss"]["event_loss"] == "gen_jet_mse":
        loss_dict["pt_e_eta_phi"] = gen_jet_mse_loss

    if config["loss"]["event_loss"] == "gen_jet_logcosh":
        loss_dict["pt_e_eta_phi"] = gen_jet_logcosh_loss

    return loss_dict, loss_weights
