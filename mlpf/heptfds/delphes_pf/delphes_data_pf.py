from pathlib import Path

from utils_delphes import prepare_data_delphes, X_FEATURES, Y_FEATURES
import tensorflow_datasets as tfds
import numpy as np

_DESCRIPTION = """
Dataset generated with Delphes.

TTbar and QCD events with PU~200.
"""

_CITATION = """
https://zenodo.org/record/4559324#.YTs853tRVH4
"""


class DelphesDataPf(tfds.core.GeneratorBasedBuilder):
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
        super(DelphesDataPf, self).__init__(*args, **kwargs)

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
        return {
            "train": self._generate_examples(path / "pythia8_ttbar"),
            "test": self._generate_examples(path / "pythia8_qcd"),
        }

    def _generate_examples(self, path):
        for fi in list(path.glob("*.pkl.bz2")):
            Xs, ygens, ycands = prepare_data_delphes(str(fi))
            for iev in range(len(Xs)):
                yield str(fi) + "_" + str(iev), {
                    "X": Xs[iev],
                    "ygen": ygens[iev],
                    "ycand": ycands[iev],
                }
