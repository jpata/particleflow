from pathlib import Path

import tensorflow_datasets as tfds
import numpy as np

from utils_delphes import X_FEATURES, Y_FEATURES
from utils_delphes import split_sample, generate_examples

_DESCRIPTION = """
Dataset generated with Delphes.

TTbar events with PU~200.
"""

_CITATION = """
https://zenodo.org/record/4559324#.YTs853tRVH4
"""


class DelphesTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.2.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Do not pad events to the same size",
        "1.2.0": "Regenerate with ARRAY_RECORD",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download from https://zenodo.org/record/4559324#.YTs853tRVH4
    """

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(DelphesTtbarPf, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=np.float32),
                    "ygen": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                }
            ),
            supervised_keys=None,
            homepage="https://zenodo.org/record/4559324#.YTs853tRVH4",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = Path(dl_manager.manual_dir)
        return split_sample(Path(path / "pythia8_ttbar/raw"))

    def _generate_examples(self, path):
        return generate_examples(path)
