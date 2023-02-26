from pathlib import Path

import tensorflow as tf
from utils_edm import (
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

import tensorflow_datasets as tfds

_DESCRIPTION = """
CLIC EDM4HEP dataset with ttbar
"""

_CITATION = """
"""


class ClicEdmTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "200k generated",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/clic_edm4hep_2023_02_21/ ./
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(
                        shape=(
                            None,
                            max(len(X_FEATURES_TRK), len(X_FEATURES_CL)),
                        ),
                        dtype=tf.float32,
                    ),
                    "ygen": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return split_sample(Path(path / "p8_ee_tt_ecm380/"))

    def _generate_examples(self, files):
        return generate_examples(files)
