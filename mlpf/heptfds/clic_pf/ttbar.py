from pathlib import Path

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from mlpf.data_clic.postprocessing import prepare_data_clic

_DESCRIPTION = """
CLIC dataset with ttbar
"""

_CITATION = """
"""

PADDED_NUM_ELEM_SIZE = 300

# these labels are for tracks from track_as_array
X_FEATURES_TRK = ["type", "px", "py", "pz", "nhits", "d0", "z0"]

# these labels are for clusters from cluster_as_array
X_FEATURES_CL = ["type", "x", "y", "z", "nhits_ecal", "nhits_hcal", "energy"]

Y_FEATURES = ["type", "charge", "px", "py", "pz"]


def split_sample(path, pad_size, test_frac=0.8):
    files = sorted(list(path.glob("*.json.bz2")))
    print("Found {} files in {}".format(files, path))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0
    return {"train": generate_examples(files_train, pad_size), "test": generate_examples(files_test, pad_size)}


def generate_examples(files, pad_size):
    for fi in files:
        ret = prepare_data_clic(fi)
        for iev, (X, ycand, ygen) in enumerate(ret):
            X = X[:pad_size]
            X = np.pad(X, [(0, pad_size - X.shape[0]), (0, 0)])
            ygen = ygen[:pad_size]
            ygen = np.pad(ygen, [(0, pad_size - ygen.shape[0]), (0, 0)])
            ycand = ycand[:pad_size]
            ycand = np.pad(ycand, [(0, pad_size - ycand.shape[0]), (0, 0)])

            yield str(fi) + "_" + str(iev), {
                "X": X.astype(np.float32),
                "ygen": ygen.astype(np.float32),
                "ycand": ycand.astype(np.float32),
            }


class ClicTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, 7), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, 5), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, 5), dtype=tf.float32),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK, x_features_cluster=X_FEATURES_CL, y_features=Y_FEATURES
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return split_sample(Path("data/clic/gev380ee_pythia6_ttbar_rfull201/raw"), PADDED_NUM_ELEM_SIZE)

    def _generate_examples(self, files):
        return generate_examples(files, PADDED_NUM_ELEM_SIZE)
