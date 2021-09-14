"""CMS PF SinglePi dataset."""

from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

from heptfds import cms_utils

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

SingleElectron events.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""

PADDED_NUM_ELEM_SIZE = 320

class CmsPfSingleElectron(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_singlepi dataset."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/SingleElectronFlatPt1To100_pythia8_cfi data/
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cms_pf): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, len(X_FEATURES)), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(PADDED_NUM_ELEM_SIZE, len(Y_FEATURES)), dtype=tf.float32),
                }
            ),
            supervised_keys=("X", "ycand"),
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        sample_dir = "SingleElectronFlatPt1To100_pythia8_cfi"
        files = sorted(list((path/sample_dir/"raw").glob("*.pkl*")))
        idx_split = int(0.8*len(files))
        files_train = files[:idx_split]
        files_test= files[idx_split:]
        return {"train": self._generate_examples(files_train), "test": self._generate_examples(files_test)}

    def _generate_examples(self, files):

        for fi in files:
            X, ygen, ycand = cms_utils.prepare_data_cms(str(fi), PADDED_NUM_ELEM_SIZE)
            for ii in range(X[0].shape[0]):
                x = X[0][ii]
                yg = ygen[0][ii]
                yc = ycand[0][ii]
                yield str(fi) + "_" + str(ii), {
                    "X": x,
                    "ygen": yg,
                    "ycand": yc,
                }

