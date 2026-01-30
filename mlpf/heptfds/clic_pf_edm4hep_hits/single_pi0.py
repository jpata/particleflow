import os
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

_DESCRIPTION = """
CLIC EDM4HEP dataset with single pi0 with raw hits.
  - X: reconstructed tracks and calorimeter hits, variable number N per event
  - ygen: stable generator particles, zero-padded to N per event
  - ycand: baseline particle flow particles, zero-padded to N per event
"""

_CITATION = """
Pata, Joosep, Wulff, Eric, Duarte, Javier, Mokhtar, Farouk, Zhang, Mengke, Girone, Maria, & Southwick, David. (2023).
Simulated datasets for detector and particle flow reconstruction: CLIC detector (1.1) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.8260741
"""


class ClicEdmSinglePi0HitsPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "1.7.0"))
    RELEASE_NOTES = {
        "1.1.0": "Remove track referencepoint feature",
        "1.2.0": "Keep all interacting genparticles",
        "1.5.0": "Regenerate with ARRAY_RECORD",
        "1.7.0": "Update track features",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the raw input files in ROOT EDM4HEP format, please see the citation above.

    The processed tensorflow_dataset can also be downloaded from:
    FIXME
    """

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(ClicEdmSinglePi0HitsPf, self).__init__(*args, **kwargs)

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
        return split_sample(Path(path / "pi0/"))

    def _generate_examples(self, files):
        return generate_examples(files)
