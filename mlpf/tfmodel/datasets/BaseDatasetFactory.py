import tensorflow as tf
import tensorflow_datasets as tfds
import heptfds

#Unpacks a flat target array along the feature axis to a feature dict
#the feature order is defined in the data prep stage (postprocessing2.py)
def unpack_target(y, num_output_classes):
    return {
        "cls": tf.one_hot(tf.cast(y[..., 0], tf.int32), num_output_classes),
        "charge": y[..., 1:2],
        "pt": y[..., 2:3],
        "eta": y[..., 3:4],
        "sin_phi": y[..., 4:5],
        "cos_phi": y[..., 5:6],
        "energy": y[..., 6:7],
        "sum_energy": tf.reduce_sum(y[..., 6:7], axis=-2),
        "sum_pt": tf.reduce_sum(y[..., 2:3], axis=-2)
    }

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

            #mask to keep only nonzero elements
            msk_elems = tf.cast(X[:, 0:1]!=0, tf.float32)

            #mask to keep only nonzero target particles
            msk_signal = tf.cast(y[:, 0:1]!=0, tf.float32)

            #inputs: X
            #targets: dict by classification (cls) and regression feature columns
            #weights: dict of weights for each target
            return (
                X, unpack_target(y, num_output_classes),
                {
                    "cls": msk_elems,
                    "charge": msk_elems*msk_signal,
                    "pt": msk_elems*msk_signal,
                    "eta": msk_elems*msk_signal,
                    "sin_phi": msk_elems*msk_signal,
                    "cos_phi": msk_elems*msk_signal,
                    "energy": msk_elems*msk_signal,
                    "sum_energy": 1.0,
                    "sum_pt": 1.0
                }
            )
        return func
    
    def get_dataset(self, split, max_examples_per_split=None):
        raise NotImplementedError
