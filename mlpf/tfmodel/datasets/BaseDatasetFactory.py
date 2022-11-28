import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm


# Unpacks a flat target array along the feature axis to a feature dict
# the feature order is defined in the data prep stage (postprocessing2.py)
def unpack_target(y, num_output_classes, config):
    msk_pid = tf.cast(y[..., 0:1] != 0, tf.float32)

    pt = y[..., 2:3] * msk_pid
    energy = y[..., 6:7] * msk_pid
    eta = y[..., 3:4] * msk_pid
    sin_phi = y[..., 4:5] * msk_pid
    cos_phi = y[..., 5:6] * msk_pid
    jet_idx = y[..., 7:8] * msk_pid

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
        pt_e_eta_phi = tf.concat([pt, energy, eta, sin_phi, cos_phi, jet_idx], axis=-1)
        ret["pt_e_eta_phi"] = pt_e_eta_phi

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
    assert target_particles in ["gen", "cand"], "Target particles has to be 'cand' or 'gen'."

    def func(data_item):
        X = data_item["X"]
        y = data_item["y{}".format(target_particles)]

        # mask to keep only nonzero (not zero-padded within the batch) elements
        msk_elems = tf.cast(X[..., 0:1] != 0, tf.float32)

        # mask to keep only nonzero (not zero-padded due to object condensation) target particles
        # also mask particles where true energy is 0 (FIXME: check this in postprocessing.py!)
        msk_signal = tf.cast(y[..., 0:1] != 0, tf.float32) * tf.cast(y[..., 6:7] > 0, tf.float32)

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
    for ids, ds in enumerate(datasets):
        indices += ds.num_steps() * [ids]
    indices = np.array(indices, np.int64)
    np.random.shuffle(indices)

    choice_dataset = tf.data.Dataset.from_tensor_slices(indices)
    interleaved_tensorflow_dataset = tf.data.experimental.choose_from_datasets(
        [ds.tensorflow_dataset for ds in datasets], choice_dataset
    )

    ds = MLPFDataset(joint_dataset_name, split, interleaved_tensorflow_dataset, sum([ds.num_samples for ds in datasets]))
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
                isteps = 0
                logging.info("Checking the number of steps in {}:{}".format(self.name, self.split))
                for elem in tqdm.tqdm(self.tensorflow_dataset):
                    isteps += 1
                total_num_steps = isteps
                self._num_steps = total_num_steps
            return self._num_steps
