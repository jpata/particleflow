from pathlib import Path

import tensorflow as tf
from utils_edm import (
    X_FEATURES_CH,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample_several,
)

import tensorflow_datasets as tfds

_DESCRIPTION = """
CLIC EDM4HEP dataset with single electron with raw hits
"""

_CITATION = """
"""


class ClicEdmSingleElectronHitsPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.2.0")
    RELEASE_NOTES = {
        "1.1.0": "Remove track referencepoint feature",
        "1.2.0": "Keep all interacting genparticels"
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    FIXME
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
                            max(len(X_FEATURES_TRK), len(X_FEATURES_CH)),
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
                x_features_calohit=X_FEATURES_CH,
                y_features=Y_FEATURES,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return split_sample_several([Path(path / "e-/"), Path(path / "e+/")])

    def _generate_examples(self, files):
        return generate_examples(files)
