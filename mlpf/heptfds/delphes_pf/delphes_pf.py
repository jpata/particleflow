from pathlib import Path
import tensorflow as tf
import tqdm

import tensorflow_datasets as tfds

from delphes_utils import prepare_data_delphes, X_FEATURES, Y_FEATURES

_DESCRIPTION = """
Dataset generated with Delphes.

TTbar and QCD events with PU~200.
"""

# TODO(delphes_pf): BibTeX citation
_CITATION = """
https://zenodo.org/record/4559324#.YTs853tRVH4
"""


class DelphesPf(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for delphes_pf dataset."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Do not pad events to the same size",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download from https://zenodo.org/record/4559324#.YTs853tRVH4
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(delphes_pf): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("X", "ygen"),  # Set to `None` to disable
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
        """Yields examples."""
        for fi in tqdm.tqdm(list(path.glob("*.pkl.bz2"))):
            Xs, ygens, ycands = prepare_data_delphes(str(fi))
            for iev in range(len(Xs)):
                yield str(fi) + "_" + str(iev), {
                    "X": Xs[iev],
                    "ygen": ygens[iev],
                    "ycand": ycands[iev],
                }
