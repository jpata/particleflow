import tensorflow as tf


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

    return ret


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

            # mask to keep only nonzero elements
            msk_elems = tf.cast(X[:, 0:1] != 0, tf.float32)

            # mask to keep only nonzero target particles
            msk_signal = tf.cast(y[:, 0:1] != 0, tf.float32)

            target = unpack_target(y, num_output_classes, self.cfg)

            # inputs: X
            # targets: dict by classification (cls) and regression feature columns
            # weights: dict of weights for each target
            return (
                X,
                target,
                {
                    "cls": msk_elems,
                    "charge": msk_elems * msk_signal,
                    "pt": msk_elems * msk_signal,
                    "eta": msk_elems * msk_signal,
                    "sin_phi": msk_elems * msk_signal,
                    "cos_phi": msk_elems * msk_signal,
                    "energy": msk_elems * msk_signal,
                },
            )

        return func

    def get_dataset(self, split, max_examples_per_split=None):
        raise NotImplementedError
