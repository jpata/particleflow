import tensorflow as tf
import tensorflow_datasets as tfds
import heptfds


class BaseDatasetFactory:
    def __init__(self, config):
        self.cfg = config

    def get_map_to_supervised(self):
        target_particles = self.cfg["dataset"]["target_particles"]
        num_output_classes = self.cfg["dataset"]["num_output_classes"]
        assert target_particles in ["gen", "cand"], "Target particles has to be 'cand' or 'gen'."
        def func(data_item):
            X = data_item["X"]
            y = data_item["y{}".format(target_particles)]
            return (
                X,
                {
                    "cls": tf.one_hot(tf.cast(y[:, :, 0], tf.int32), num_output_classes),
                    "charge": y[:, :, 1:2],
                    "pt": y[:, :, 2:3],
                    "eta": y[:, :, 3:4],
                    "sin_phi": y[:, :, 4:5],
                    "cos_phi": y[:, :, 5:6],
                    "energy": y[:, :, 6:7],
                },
            )
        return func
    
    def get_dataset(self, split, max_examples_per_split=None):
        raise NotImplementedError