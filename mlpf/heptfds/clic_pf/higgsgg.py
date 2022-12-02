from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from clic_utils import (
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

_DESCRIPTION = """
CLIC dataset with Higgs->gg
"""

_CITATION = """
"""


class ClicHiggsGgPf(tfds.core.GeneratorBasedBuilder):
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
                    "X": tfds.features.Tensor(shape=(None, max(len(X_FEATURES_TRK), len(X_FEATURES_CL))), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(None, 6), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, 6), dtype=tf.float32),
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
        return split_sample(Path("data/clic/gev380ee_pythia6_higgs_gamgam_full201/"))

    def _generate_examples(self, files):
        return generate_examples(files)
