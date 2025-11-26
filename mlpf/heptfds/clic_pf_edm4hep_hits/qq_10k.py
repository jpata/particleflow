from pathlib import Path

import numpy as np
from utils_edm import (
    X_FEATURES_CH,
    X_FEATURES_TRK,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

import tensorflow_datasets as tfds

from qq import _DESCRIPTION, _CITATION


class ClicEdmQqHitsPf10k(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.7.0")
    RELEASE_NOTES = {
        "1.5.0": "Regenerate with ARRAY_RECORD",
        "1.7.0": "Update track features",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the raw input files in ROOT EDM4HEP format, please see the citation above.

    The processed tensorflow_dataset can also be downloaded from: https://zenodo.org/record/8414225
    """

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(ClicEdmQqHitsPf10k, self).__init__(*args, **kwargs)

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
                        dtype=np.float32,
                    ),
                    "ygen": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
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
        return split_sample(Path(path / "p8_ee_qq_ecm380/"), max_files=100)

    def _generate_examples(self, files):
        return generate_examples(files)
