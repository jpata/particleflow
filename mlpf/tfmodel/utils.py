import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    logging.warning("horovod not found, ignoring")


import tensorflow as tf
import yaml
from tensorflow.keras import mixed_precision

# from tfmodel.datasets import CMSDatasetFactory, DelphesDatasetFactory
from tfmodel.datasets.BaseDatasetFactory import (
    MLPFDataset,
    get_map_to_supervised,
    interleave_datasets,
    mlpf_dataset_from_config,
)
from tfmodel.model_setup import configure_model_weights, make_model
from tfmodel.onecycle_scheduler import (
    MomentumOneCycleScheduler,
    OneCycleScheduler,
)


@tf.function
def histogram_2d(
    mask,
    eta,
    phi,
    weights_px,
    weights_py,
    eta_range,
    phi_range,
    nbins,
    bin_dtype=tf.float32,
):
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
def batched_histogram_2d(
    mask,
    eta,
    phi,
    w_px,
    w_py,
    x_range,
    y_range,
    nbins,
    bin_dtype=tf.float32,
):
    return tf.map_fn(
        lambda a: histogram_2d(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            x_range,
            y_range,
            nbins,
            bin_dtype,
        ),
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

    if weights is not None:
        config["setup"]["weights"] = weights

    return config, config_file_stem


def get_best_checkpoint(train_dir):
    checkpoint_list = list(Path(Path(train_dir) / "weights").glob("weights*.hdf5"))
    # Sort the checkpoints according to the loss in their filenames
    checkpoint_list.sort(key=lambda x: float(re.search(r"\d+-\d+.\d+", str(x.name))[0].split("-")[-1]))
    # Return the checkpoint with smallest loss
    return str(checkpoint_list[0])


def get_latest_checkpoint(train_dir):
    checkpoint_list = list(Path(Path(train_dir) / "weights").glob("weights*.hdf5"))
    # Sort the checkpoints according to the epoch number in their filenames
    checkpoint_list.sort(key=lambda x: int(re.search(r"\d+-\d+.\d+", str(x.name))[0].split("-")[0]))
    # Return the checkpoint with highest epoch number
    return str(checkpoint_list[-1])


def delete_all_but_best_checkpoint(train_dir, dry_run):
    checkpoint_list = list(Path(Path(train_dir) / "weights").glob("weights*.hdf5"))
    # Don't remove the checkpoint with smallest loss
    if len(checkpoint_list) == 1:
        raise UserWarning("There is only one checkpoint. No deletion was made.")
    elif len(checkpoint_list) == 0:
        raise UserWarning("Couldn't find any checkpoints. No deletion was made.")
    else:
        # Sort the checkpoints according to the loss in their filenames
        checkpoint_list.sort(key=lambda x: float(re.search(r"\d+-\d+.\d+", str(x))[0].split("-")[-1]))
        best_ckpt = checkpoint_list.pop(0)
        for ckpt in checkpoint_list:
            if not dry_run:
                ckpt.unlink()

        logging.info("Removed all checkpoints in {} except {}".format(train_dir, best_ckpt))


def _get_num_gpus(envvar="CUDA_VISIBLE_DEVICES"):
    env = os.environ[envvar]
    gpus = [int(x) for x in env.split(",")]
    if len(gpus) == 1 and gpus[0] == -1:
        num_gpus = 0
    else:
        num_gpus = len(gpus)
    return num_gpus, gpus


def get_num_gpus():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        num_gpus, gpus = _get_num_gpus("CUDA_VISIBLE_DEVICES")
    elif "ROCR_VISIBLE_DEVICES" in os.environ:
        num_gpus, gpus = _get_num_gpus("ROCR_VISIBLE_DEVICES")
    elif "HABANA_VISIBLE_DEVICES" in os.environ:
        num_gpus, gpus = _get_num_gpus("HABANA_VISIBLE_DEVICES")
    else:
        logging.warning(
            "CUDA/ROC variable is empty. \
            If you don't have or intend to use GPUs, this message can be ignored."
        )
        num_gpus = 0
        gpus = []
    return num_gpus, gpus


# retrieves the appropriate tf.distribute strategy for single-node training
def get_singlenode_strategy(num_cpus=None):
    # Always use the correct number of threads that were requested
    if num_cpus == 1:
        logging.warning("num_cpus==1, using explicitly only one CPU thread")

    if num_cpus:
        os.environ["OMP_NUM_THREADS"] = str(num_cpus)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpus)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(num_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
        tf.config.threading.set_intra_op_parallelism_threads(num_cpus)

    num_gpus, gpus = get_num_gpus()

    if num_gpus > 1:
        # multiple GPUs selected
        # logging.info("Attempting to use multiple GPUs with tf.distribute.MultiWorkerMirroredStrategy()...")
        # communication_options = tf.distribute.experimental.CommunicationOptions(
        #    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
        # )
        # strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
        logging.info("Attempting to use multiple GPUs with tf.distribute.MirroredStrategy()...")
        strategy = tf.distribute.MirroredStrategy()
    elif num_gpus == 1:
        # single GPU
        logging.info("Using a single GPU with tf.distribute.OneDeviceStrategy()")
        strategy = tf.distribute.OneDeviceStrategy(f"gpu:{gpus[0]}")
    else:
        logging.info("Fallback to CPU, using tf.distribute.OneDeviceStrategy('cpu')")
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    # tf.distribute strategies require the batch size to be increased by the number of devices,
    # as the data of a single batch gets split across all the devices
    num_batches_multiplier = 1
    if num_gpus > 1:
        num_batches_multiplier = num_gpus
        logging.info(f"Multiple GPUs detected, num_batches_multiplier={num_batches_multiplier}")

    return strategy, num_gpus, num_batches_multiplier


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
        logging.info("not using LR schedule")
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
        return opt
    elif config["setup"]["optimizer"] == "adamw":
        cfg_adam = config["optimizer"]["adamw"]
        return tf.keras.optimizers.AdamW(learning_rate=lr, amsgrad=cfg_adam["amsgrad"])
    elif config["setup"]["optimizer"] == "lion":
        return tf.keras.optimizers.Lion(learning_rate=lr)
    elif config["setup"]["optimizer"] == "sgd":
        cfg_sgd = config["optimizer"]["sgd"]
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=cfg_sgd["momentum"],
            nesterov=cfg_sgd["nesterov"],
        )
    else:
        raise ValueError("Only 'adam', 'adamw', 'sgd', 'lion' are supported optimizers, got {}".format(config["setup"]["optimizer"]))


def get_tuner(cfg_hypertune, model_builder, outdir, recreate, strategy):
    import keras_tuner as kt

    if cfg_hypertune["algorithm"] == "random":
        logging.info("Keras Tuner: Using RandomSearch")
        cfg_rand = cfg_hypertune["random"]
        return kt.RandomSearch(
            model_builder,
            objective=cfg_rand["objective"],
            max_trials=cfg_rand["max_trials"],
            project_name=outdir,
            overwrite=recreate,
        )
    elif cfg_hypertune["algorithm"] == "bayesian":
        logging.info("Keras Tuner: Using BayesianOptimization")
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
        logging.info("Keras Tuner: Using Hyperband")
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


def load_and_interleave(
    joint_dataset_name,
    dataset_names,
    config,
    num_batches_multiplier,
    split,
    batch_size,
    max_events,
    horovod_enabled,
):

    cachedir = config["cache"]

    datasets = [
        mlpf_dataset_from_config(
            ds_name,
            config,
            split,
            max_events=max_events,
            horovod_enabled=horovod_enabled,
        )
        for ds_name in dataset_names
    ]
    ds = interleave_datasets(joint_dataset_name, split, datasets)
    tensorflow_dataset = ds.tensorflow_dataset.map(get_map_to_supervised(config), num_parallel_calls=tf.data.AUTOTUNE)

    # use dynamic batching depending on the sequence length
    if config["batching"]["bucket_by_sequence_length"]:
        if "combined_graph_layer" in config["parameters"]:
            bin_size = config["parameters"]["combined_graph_layer"]["bin_size"]
        else:
            bin_size = 256

        # generate (max_elems, batch_size) pairs
        # start from (bin_size+1, max_n*bin_size), and step down
        # this number was tuned for an A100,
        # and means we can treat at most events with 60*bin_size elements in it (60*256 = 15360)
        max_n = 60
        bucket_batch_sizes = [(bin_size * n + 1, int(max_n * bin_size / (n * bin_size))) for n in range(1, max_n + 1)]
        bucket_boundaries = [int(x[0]) for x in bucket_batch_sizes[:-1]]

        # increase batch sizes for number of gpus and with the overall batch multiplier
        bucket_batch_sizes = [max(int(x[1] * num_batches_multiplier * config["batching"]["batch_multiplier"]), 1) for x in bucket_batch_sizes]
        logging.info("Batching {}:{} with bucket_by_sequence_length".format(ds.name, ds.split))
        logging.info("bucket_boundaries={}".format(bucket_boundaries))
        logging.info("bucket_batch_sizes={}".format(bucket_batch_sizes))
        tensorflow_dataset = tensorflow_dataset.bucket_by_sequence_length(
            # length is determined by the number of elements in the input set
            element_length_func=lambda X, y, mask: tf.shape(X)[0],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            pad_to_bucket_boundary=True,
            drop_remainder=True,
        )
        num_steps = None
    # use fixed-size batching
    else:
        bs = batch_size

        # Multiply batch size by number of GPUs for MirroredStrategy
        if not config["setup"]["horovod_enabled"]:
            if num_batches_multiplier > 1:
                bs = bs * num_batches_multiplier
        bs = int(bs)
        assert bs > 0
        logging.info("Batching {}:{} with padded_batch, batch_size={}".format(ds.name, ds.split, bs))

        # For padded_batch, either pad each batch of events to the largest event in each batch (if event_pad_size = None)
        # or pad all events to a fixed size for all batches (if event_pad_size>0).
        # Note that in the latter case, all events must be smaller
        # than the pad size, otherwise, tfds will throw an error.
        padded_shapes = list(tensorflow_dataset.element_spec)
        event_pad_size = config["train_test_datasets"][joint_dataset_name]["event_pad_size"]
        if event_pad_size == -1:
            event_pad_size = None
        padded_shapes[0] = (event_pad_size, padded_shapes[0].shape[1])
        padded_shapes[1] = {k: (event_pad_size, v.shape[1]) for k, v in padded_shapes[1].items()}
        padded_shapes[2] = {k: (event_pad_size, v.shape[1]) for k, v in padded_shapes[2].items()}
        padded_shapes = tuple(padded_shapes)

        num_steps = ds.num_samples // bs
        tensorflow_dataset = tensorflow_dataset.padded_batch(bs, padded_shapes=padded_shapes, drop_remainder=True)

    ds = MLPFDataset(
        ds.name,
        split,
        tensorflow_dataset,
        ds.num_samples,
    )

    if not (num_steps is None):
        ds._num_steps = num_steps

    statefile = f"{cachedir}/{ds.name}_{ds.split}.json"
    if config["batching"]["bucket_by_sequence_length"] and os.path.isfile(statefile):
        ds.load_state(statefile)
    logging.info("Dataset {} after batching, {} steps, {} samples".format(ds.name, ds.num_steps(), ds.num_samples))
    if config["batching"]["bucket_by_sequence_length"]:
        ds.save_state(statefile)
    return ds


# Load multiple datasets and mix them together
def get_datasets(
    datasets_to_interleave,
    config,
    num_batches_multiplier,
    split,
    max_events=None,
    horovod_enabled=False,
):
    datasets = []
    for joint_dataset_name in datasets_to_interleave.keys():
        ds_conf = datasets_to_interleave[joint_dataset_name]
        if ds_conf["datasets"] is None:
            logging.warning("No datasets in {} list.".format(joint_dataset_name))
        else:
            ds = load_and_interleave(
                joint_dataset_name,
                ds_conf["datasets"],
                config,
                num_batches_multiplier,
                split,
                ds_conf["batch_per_gpu"],
                max_events=max_events,
                horovod_enabled=horovod_enabled,
            )
            datasets.append(ds)

    ds = interleave_datasets("all", split, datasets)

    # Interleaved dataset does not support FILE based sharding
    # explicitly switch to DATA sharding to avoid a lengthy warning
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds.tensorflow_dataset = ds.tensorflow_dataset.with_options(options)

    logging.info("Final dataset with {} steps".format(ds.num_steps()))
    return ds


def set_config_loss(config, trainable):
    if trainable == "classification":
        config["dataset"]["pt_loss_coef"] = 0.0
        config["dataset"]["event_loss_coef"] = 0.0
        config["dataset"]["eta_loss_coef"] = 0.0
        config["dataset"]["sin_phi_loss_coef"] = 0.0
        config["dataset"]["cos_phi_loss_coef"] = 0.0
        config["dataset"]["energy_loss_coef"] = 0.0
    elif trainable == "regression":
        config["dataset"]["classification_loss_coef"] = 0.0
        config["dataset"]["charge_loss_coef"] = 0.0
    elif trainable == "all":
        pass
    return config


def get_loss_from_params(input_dict):
    input_dict = input_dict.copy()
    loss_type = input_dict.pop("type")
    if loss_type == "SigmoidFocalCrossEntropy":
        from .tfa import SigmoidFocalCrossEntropy

        loss_cls = SigmoidFocalCrossEntropy
    else:
        loss_cls = getattr(tf.keras.losses, loss_type)
    return loss_cls(**input_dict, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)


# batched version of https://github.com/VinAIResearch/DSW/blob/master/gsw.py#L19
@tf.function
def sliced_wasserstein_loss(y_true_pt_e_eta_phi, y_pred_pt_e_eta_phi, num_projections=200):
    # mask of true genparticles
    # msk_pid = y_true_pt_e_eta_phi[..., 6:7]

    # take (pt, energy, eta, sin_phi, cos_phi) as defined in BaseDatasetFactory.py
    y_true = y_true_pt_e_eta_phi[..., :5]
    y_pred = y_pred_pt_e_eta_phi[..., :5]

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
    mask = tf.cast(y_true[:, :, 6] != 0, tf.float32)

    eta_true = y_true[..., 2]
    eta_pred = y_pred[..., 2]

    sin_phi_true = y_true[..., 3]
    sin_phi_pred = y_pred[..., 3]

    cos_phi_true = y_true[..., 4]
    cos_phi_pred = y_pred[..., 4]

    # note that calculating phi=atan2(sin_phi, cos_phi)
    # introduces a numerical instability which can lead to NaN.
    phi_true = tf.math.atan2(sin_phi_true, cos_phi_true) * mask
    phi_pred = tf.math.atan2(sin_phi_pred, cos_phi_pred) * mask

    pt_true = y_true[..., 0]
    pt_pred = y_pred[..., 0]

    px_true = pt_true * y_true[..., 4]
    py_true = pt_true * y_true[..., 3]
    px_pred = pt_pred * y_pred[..., 4]
    py_pred = pt_pred * y_pred[..., 3]

    pt_hist_true = batched_histogram_2d(
        mask,
        eta_true,
        phi_true,
        px_true,
        py_true,
        tf.cast([-6.0, 6.0], tf.float32),
        tf.cast([-4.0, 4.0], tf.float32),
        20,
    )

    pt_hist_pred = batched_histogram_2d(
        mask,
        eta_pred,
        phi_pred,
        px_pred,
        py_pred,
        tf.cast([-6.0, 6.0], tf.float32),
        tf.cast([-4.0, 4.0], tf.float32),
        20,
    )

    mse = tf.math.sqrt(tf.reduce_mean((pt_hist_true - pt_hist_pred) ** 2, axis=[-1, -2]))
    return mse


@tf.function
def jet_reco(px, py, jet_idx, max_jets):
    # tf.debugging.assert_shapes(
    #    [
    #        (px, ("N")),
    #        (py, ("N")),
    #        (jet_idx, ("N")),
    #    ]
    # )

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
    # tf.debugging.assert_shapes(
    #    [
    #        (px, ("B", "N")),
    #        (py, ("B", "N")),
    #        (jet_idx, ("B", "N")),
    #    ]
    # )

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


# y_true: [nbatch, nptcl, 5] array of true particle properties.
# y_pred: [nbatch, nptcl, 5] array of predicted particle properties
# last dim corresponds to [pt, energy, eta, sin_phi, cos_phi, gen_jet_idx]
# max_jets: integer of the max number of jets to consider
# returns: dict of true and predicted jet pts.
@tf.function
def compute_jet_pt(y_true, y_pred, max_jets=201):
    y = {}
    y["true"] = y_true
    y["pred"] = y_pred
    jet_pt = {}

    jet_idx = tf.cast(y["true"][..., 5], dtype=tf.int32)

    # mask the predicted particles in cases where there was no true particle
    msk = tf.cast(y_true[:, :, 6] != 0, tf.float32)
    for typ in ["true", "pred"]:
        px = y[typ][..., 0] * y[typ][..., 4] * msk
        py = y[typ][..., 0] * y[typ][..., 3] * msk
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
    cls_loss = get_loss_from_params(config["loss"].get("cls_loss"))

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

    if config["loss"]["met_loss"] != "none":
        loss_weights["met"] = config["loss"]["met_loss_coef"]

    if config["loss"]["event_loss"] == "sliced_wasserstein":
        loss_dict["pt_e_eta_phi"] = sliced_wasserstein_loss

    if config["loss"]["event_loss"] == "hist_2d":
        loss_dict["pt_e_eta_phi"] = hist_2d_loss

    if config["loss"]["event_loss"] == "gen_jet_mse":
        loss_dict["pt_e_eta_phi"] = gen_jet_mse_loss

    if config["loss"]["event_loss"] == "gen_jet_logcosh":
        loss_dict["pt_e_eta_phi"] = gen_jet_logcosh_loss

    if config["loss"]["met_loss"] != "none":
        loss_dict["met"] = get_loss_from_params(config["loss"].get("met_loss", default_loss))

    return loss_dict, loss_weights


# get the datasets for training, testing and validation
def get_train_test_val_datasets(config, num_batches_multiplier, ntrain=None, ntest=None, horovod_enabled=False):
    ds_train = get_datasets(
        config["train_test_datasets"],
        config,
        num_batches_multiplier,
        "train",
        max_events=ntrain,
        horovod_enabled=horovod_enabled,
    )
    ds_test = get_datasets(
        config["train_test_datasets"],
        config,
        num_batches_multiplier,
        "test",
        max_events=ntest,
        horovod_enabled=horovod_enabled,
    )
    ds_val = mlpf_dataset_from_config(
        config["validation_dataset"],
        config,
        "test",
        max_events=config["validation_num_events"],
        horovod_enabled=horovod_enabled,
    )
    ds_val.tensorflow_dataset = ds_val.tensorflow_dataset.padded_batch(config["validation_batch_size"])

    return ds_train, ds_test, ds_val


def model_scope(config, total_steps, weights=None, horovod_enabled=False):
    lr_schedule, optim_callbacks, lr = get_lr_schedule(config, steps=total_steps)
    opt = get_optimizer(config, lr_schedule)

    if config["setup"]["dtype"] == "float16":
        model_dtype = tf.dtypes.float16
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    elif config["setup"]["dtype"] == "bfloat16":
        model_dtype = tf.dtypes.bfloat16
        policy = mixed_precision.Policy("mixed_bfloat16")
        mixed_precision.set_global_policy(policy)
        opt = mixed_precision.LossScaleOptimizer(opt)
    else:
        model_dtype = tf.dtypes.float32

    model = make_model(config, model_dtype)

    # Build the layers after the element and feature dimensions are specified
    model.build((1, None, config["dataset"]["num_input_features"]))

    initial_epoch = 0
    loaded_opt = None

    if weights:
        # We need to load the weights in the same trainable configuration as the model was set up
        configure_model_weights(model, config["setup"].get("weights_config", "all"))
        model.load_weights(weights, by_name=True)

        logging.info("using checkpointed model weights from: {}".format(weights))
        opt_weight_file = weights.replace("hdf5", "pkl").replace("/weights-", "/opt-")
        if os.path.isfile(opt_weight_file):
            loaded_opt = pickle.load(open(opt_weight_file, "rb"))
            logging.info("using checkpointed optimizer weights from: {}".format(opt_weight_file))

            def model_weight_setting():
                grad_vars = model.trainable_weights
                logging.info("grad_vars={}".format(len(grad_vars)))
                if loaded_opt:
                    logging.info("setting optimizer state")
                    opt.build(grad_vars)
                    try:
                        opt.load_own_variables(loaded_opt["weights"])
                    except Exception as e:
                        logging.error("could not restore optimizer with load_own_variables: {}".format(e))
                        # TF 2.12 compatibility
                        for i, variable in enumerate(opt.variables):
                            variable.assign(loaded_opt["weights"][str(i)])

            logging.info("distributing optimizer state")
            strategy = tf.distribute.get_strategy()
            strategy.run(model_weight_setting)

        initial_epoch = int(weights.split("/")[-1].split("-")[1])

    logging.info("setting model weights")
    config = set_config_loss(config, config["setup"]["trainable"])
    configure_model_weights(model, config["setup"]["trainable"])

    if horovod_enabled:
        # wrap the Keras optimizer with Horovod's Distributed Optimizer.
        # https://horovod.readthedocs.io/en/latest/keras.html
        opt = hvd.DistributedOptimizer(opt)

    if not horovod_enabled or hvd.rank() == 0:
        logging.info("model weights follow")
        tw_names = [m.name for m in model.trainable_weights]
        for w in model.weights:
            logging.info("layer={} trainable={} shape={} num_weights={}".format(w.name, w.name in tw_names, w.shape, np.prod(w.shape)))

    loss_dict, loss_weights = get_loss_dict(config)

    logging.info("compiling model")
    model.compile(
        loss=loss_dict,
        optimizer=opt,
        sample_weight_mode="temporal",
        loss_weights=loss_weights,
    )

    if not horovod_enabled or hvd.rank() == 0:
        model.summary()

    return model, optim_callbacks, initial_epoch


def initialize_horovod(habana_enabled=False):
    hvd.init()
    if not habana_enabled:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    # in horovod, we don't need to increase batch size, as the dataset is sharded
    num_batches_multiplier = 1

    return hvd.size(), num_batches_multiplier
