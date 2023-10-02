import logging

import numpy as np
import tensorflow as tf
import json

import tensorflow_datasets as tfds

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError:
    pass


# Unpacks a flat target array along the feature axis to a feature dict
# the feature order is defined in the data prep stage
def unpack_target(y, num_output_classes, config):
    msk_pid = tf.cast(y[..., 0:1] != 0, tf.float32)

    pt = y[..., 2:3] * msk_pid
    energy = y[..., 6:7] * msk_pid
    eta = y[..., 3:4] * msk_pid
    sin_phi = y[..., 4:5] * msk_pid
    cos_phi = y[..., 5:6] * msk_pid

    type_as_int = tf.cast(y[..., 0], tf.int32)

    # JP: in cms_pf_single_proton, charge is sometimes 2. so I added clip here currently
    charge_as_int = tf.clip_by_value(tf.cast(y[..., 1] + 1, tf.int32), 0, 2)

    tf.debugging.assert_greater_equal(charge_as_int, 0, message="charge", summarize=100)
    tf.debugging.assert_less_equal(charge_as_int, 2, message="charge", summarize=100)

    tf.debugging.assert_greater_equal(type_as_int, 0, message="targettype", summarize=100)
    tf.debugging.assert_less_equal(type_as_int, num_output_classes, message="targettype", summarize=100)

    tf.debugging.assert_less_equal(tf.math.abs(pt), 1e5)
    tf.debugging.assert_less_equal(tf.math.abs(eta), 1e5)
    tf.debugging.assert_less_equal(tf.math.abs(sin_phi), 1e5)
    tf.debugging.assert_less_equal(tf.math.abs(cos_phi), 1e5)
    tf.debugging.assert_less_equal(tf.math.abs(energy), 1e5)

    ret = {
        "cls": tf.one_hot(type_as_int, num_output_classes),
        "charge": tf.one_hot(charge_as_int, 3),
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


def my_getitem(self, vals):
    tf.print(
        "reading dataset {}:{} from disk in slice {}, total={}".format(self.dataset_info.name, self.split, vals, len(self))
    )
    records = self.data_source.__getitems__(vals)
    return [self.dataset_info.features.deserialize_example_np(record, decoders=self.decoders) for record in records]


def mlpf_dataset_from_config(dataset_name, full_config, split, max_events=None, horovod_enabled=False):
    dataset_config = full_config["datasets"][dataset_name]

    def yield_from_ds():
        for elem in dss:
            yield {"X": elem["X"], "ygen": elem["ygen"], "ycand": elem["ycand"]}

    # def get_from_ds(i):
    #    elem = dss[i]
    #    return {"X": elem["X"], "ygen": elem["ygen"], "ycand": elem["ycand"]}

    # when the dataset is saved with file_format=array_record, we cannot do tfds.load, but instead must do the following
    ds_builder = tfds.builder("{}:{}".format(dataset_name, dataset_config["version"]), data_dir=dataset_config["data_dir"])
    dss = ds_builder.as_data_source(split)
    num_samples = len(dss)

    # hack to prevent a warning from tfds about accessing sequences of indices
    dss.__class__.__getitems__ = my_getitem

    output_signature = {k: tf.TensorSpec(shape=(None, v.shape[1])) for (k, v) in dss.dataset_info.features.items()}

    # Note 2023-09-09
    # from_generator uses tf.numpy_function, which creates issues with parallelization.
    # This means that in an IO-bound loop over the dataset, the performance will be somewhat limited
    # using range().map(get_from_ds) would be better, but currently the internals of array_record do not support jit.
    tf_dataset = tf.data.Dataset.from_generator(yield_from_ds, output_signature=output_signature)
    # tf_dataset = tf.data.Dataset.range(len(dss)).map(get_from_ds, num_parallel_calls=tf.data.AUTOTUNE)

    if max_events:
        tf_dataset = tf_dataset.take(max_events)
        num_samples = max_events

    if horovod_enabled:
        tf_dataset = tf_dataset.shard(num_shards=hvd.size(), index=hvd.rank())

    logging.info("Loaded {}:{} with {} samples".format(dataset_name, split, num_samples))
    ds = MLPFDataset(dataset_name, split, tf_dataset, num_samples)
    ds._num_steps = num_samples
    return ds


def get_map_to_supervised(config):
    target_particles = config["dataset"]["target_particles"]
    num_output_classes = config["dataset"]["num_output_classes"]
    num_input_classes = config["dataset"]["num_input_classes"]
    assert target_particles in [
        "gen",
        "cand",
    ], "Target particles has to be 'cand' or 'gen'."

    def func(data_item):
        X = data_item["X"]
        y = data_item["y{}".format(target_particles)]

        tf.debugging.assert_greater_equal(X[..., 0], 0.0, message="X", summarize=100)
        tf.debugging.assert_less_equal(X[..., 0], float(num_input_classes), message="X", summarize=100)

        X = tf.where(tf.math.is_inf(X), tf.zeros_like(X), X)
        X = tf.where(tf.math.is_nan(X), tf.zeros_like(X), X)

        X = tf.clip_by_value(X, -1e6, 1e6)

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
    def __init__(
        self,
        name: str,
        split: str,
        tensorflow_dataset: tf.data.Dataset,
        num_samples: int,
    ):

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
                self._num_steps = self.tensorflow_dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()

                assert self._num_steps > 0
            return self._num_steps

    def save_state(self, outfile):
        logging.info(f"saving state to {outfile}")
        with open(outfile, "w") as fi:
            json.dump(
                {
                    "num_samples": int(self.num_samples),
                    "num_steps": int(self._num_steps),
                    "name": self.name,
                    "split": self.split,
                },
                fi,
            )

    def load_state(self, infile):
        logging.info(f"loading state from {infile}")
        with open(infile, "r") as fi:
            data = json.load(fi)
            assert self.name == data["name"]
            assert self.split == data["split"]
            assert self.num_samples == data["num_samples"]
            self._num_steps = data["num_steps"]
