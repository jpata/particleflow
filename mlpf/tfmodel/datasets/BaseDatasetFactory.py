import logging

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


def unpack_target(y, num_output_classes, config):
    if config["dataset"]["schema"] == "cms" or config["dataset"]["schema"] == "delphes":
        return unpack_target_cms(y, num_output_classes, config)
    elif config["dataset"]["schema"] == "clic":
        return unpack_target_clic(y, num_output_classes, config)
    else:
        raise Exception("Unknown schema: {}".format(config["dataset"]["schema"]))


# Unpacks a flat target array along the feature axis to a feature dict
# the feature order is defined in the data prep stage
def unpack_target_cms(y, num_output_classes, config):
    msk_pid = tf.cast(y[..., 0:1] != 0, tf.float32)

    pt = y[..., 2:3] * msk_pid
    energy = y[..., 6:7] * msk_pid
    eta = y[..., 3:4] * msk_pid
    sin_phi = y[..., 4:5] * msk_pid
    cos_phi = y[..., 5:6] * msk_pid

    ret = {
        "cls": tf.one_hot(tf.cast(y[..., 0], tf.int32), num_output_classes),
        "charge": y[..., 1:2],
        "pt": pt,
        "eta": eta,
        "sin_phi": sin_phi,
        "cos_phi": cos_phi,
        "energy": energy,
    }

    if config["loss"]["event_loss"] != "none":
        jet_idx = y[..., 7:8] * msk_pid
        pt_e_eta_phi = tf.concat([pt, energy, eta, sin_phi, cos_phi, jet_idx, msk_pid], axis=-1)
        ret["pt_e_eta_phi"] = pt_e_eta_phi

    if config["loss"]["met_loss"] != "none":
        px = pt * cos_phi
        py = pt * sin_phi
        met = tf.sqrt(tf.reduce_sum(px, axis=-2) ** 2 + tf.reduce_sum(py, axis=-2) ** 2)
        ret["met"] = met

    return ret


def unpack_target_clic(y, num_output_classes, config):
    msk_pid = tf.cast(y[..., 0:1] != 0, tf.float32)

    px = y[..., 2:3] * msk_pid
    py = y[..., 3:4] * msk_pid
    pz = y[..., 4:5] * msk_pid
    energy = y[..., 5:6] * msk_pid

    pt = tf.math.sqrt(px**2 + py**2) * msk_pid
    p = tf.math.sqrt(px**2 + py**2 + pz**2) * msk_pid

    cos_theta = tf.math.divide_no_nan(pz, p)
    theta = tf.math.acos(cos_theta)
    eta = -tf.math.log(tf.math.tan(theta / 2.0)) * msk_pid

    sin_phi = tf.math.divide_no_nan(py, pt) * msk_pid
    cos_phi = tf.math.divide_no_nan(px, pt) * msk_pid

    ret = {
        "cls": tf.one_hot(tf.cast(y[..., 0], tf.int32), num_output_classes),
        "charge": y[..., 1:2],
        "pt": pt,
        "eta": eta,
        "sin_phi": sin_phi,
        "cos_phi": cos_phi,
        "energy": energy,
    }

    if config["loss"]["event_loss"] != "none":
        jet_idx = y[..., 6:7] * msk_pid
        pt_e_eta_phi = tf.concat([pt, energy, eta, sin_phi, cos_phi, jet_idx, msk_pid], axis=-1)
        ret["pt_e_eta_phi"] = pt_e_eta_phi * msk_pid

    if config["loss"]["met_loss"] != "none":
        px = pt * cos_phi
        py = pt * sin_phi
        met = tf.sqrt(tf.reduce_sum(px, axis=-2) ** 2 + tf.reduce_sum(py, axis=-2) ** 2)
        ret["met"] = met

    return ret


def mlpf_dataset_from_config(dataset_name, full_config, split, max_events=None):
    dataset_config = full_config["datasets"][dataset_name]
    tf_dataset = tfds.load(
        "{}:{}".format(dataset_name, dataset_config["version"]),
        split=split,
        as_supervised=False,
        data_dir=dataset_config["data_dir"],
        with_info=False,
        shuffle_files=False,
        download=False,
    )
    if max_events:
        tf_dataset = tf_dataset.take(max_events)
    num_samples = tf_dataset.cardinality().numpy()
    logging.info("Loaded {}:{} with {} samples".format(dataset_name, split, num_samples))
    return MLPFDataset(dataset_name, split, tf_dataset, num_samples)


def get_map_to_supervised(config):
    target_particles = config["dataset"]["target_particles"]
    num_output_classes = config["dataset"]["num_output_classes"]
    assert target_particles in [
        "gen",
        "cand",
    ], "Target particles has to be 'cand' or 'gen'."

    def func(data_item):
        X = data_item["X"]
        y = data_item["y{}".format(target_particles)]

        # mask to keep only nonzero (not zero-padded due to batching) elements
        msk_elems = tf.cast(X[..., 0:1] != 0, tf.float32)

        # mask to keep only nonzero (not zero-padded due to object condensation) target particles
        msk_signal = tf.cast(y[..., 0:1] != 0, tf.float32)

        target = unpack_target(y, num_output_classes, config)

        cls_weights = msk_elems
        reg_weights = msk_elems * msk_signal

        if config["dataset"]["cls_weight_by_pt"]:
            cls_weights *= X[..., 1:2]

        if config["dataset"]["reg_weight_by_pt"]:
            reg_weights *= X[..., 1:2]

        # inputs: X
        # targets: dict by classification (cls) and regression feature columns
        # weights: dict of weights for each target
        return (
            X,
            target,
            {
                "cls": cls_weights,
                "charge": cls_weights,
                "pt": reg_weights,
                "eta": reg_weights,
                "sin_phi": reg_weights,
                "cos_phi": reg_weights,
                "energy": reg_weights,
            },
        )

    return func


def interleave_datasets(joint_dataset_name, split, datasets):
    indices = []
    num_steps_total = 0
    for ids, ds in enumerate(datasets):
        steps = ds.num_steps()
        num_steps_total += steps
        indices += steps * [ids]
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)
    interleaved_tensorflow_dataset = tf.data.experimental.choose_from_datasets(
        [ds.tensorflow_dataset for ds in datasets], choice_dataset
    )

    ds = MLPFDataset(
        joint_dataset_name,
        split,
        interleaved_tensorflow_dataset,
        sum([ds.num_samples for ds in datasets]),
    )
    ds._num_steps = num_steps_total
    logging.info(
        "Interleaved joint dataset {}:{} with {} steps, {} samples".format(ds.name, ds.split, ds.num_steps(), ds.num_samples)
    )
    return ds


class MLPFDataset:
    def __init__(self, name, split, tensorflow_dataset, num_samples):
        self.name = name
        self.split = split
        self.tensorflow_dataset = tensorflow_dataset
        self.num_samples = num_samples
        self._num_steps = None

    def num_steps(self):
        card = self.tensorflow_dataset.cardinality().numpy()
        if card > 0:
            logging.info("Number of steps in {}:{} is known from cardinality: {}".format(self.name, self.split, card))
            return card
        else:
            if self._num_steps is None:
                logging.info("Checking the number of steps in {}:{}".format(self.name, self.split))
                # In case dynamic batching was applied, we don't know the number of steps for the dataset
                # compute it using https://stackoverflow.com/a/61019377
                self._num_steps = (
                    self.tensorflow_dataset.map(
                        lambda *args: 1,
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .reduce(tf.constant(0), lambda x, _: x + 1)
                    .numpy()
                )
                assert self._num_steps > 0
            return self._num_steps
