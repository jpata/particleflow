"""CMS PF SinglePi dataset."""

from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

from heptfds import cms_utils

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

SinglePi0 events.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""

PADDED_NUM_ELEM_SIZE = 320

class CmsPfSinglePi0(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_singlepi0 dataset."""

    VERSION = tfds.core.Version("1.2.0")
    RELEASE_NOTES = {
        "1.1.0": "Initial release",
        "1.2.0": "12_1_0_pre3 generation, add corrected energy, cluster flags, 20k events",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/SinglePi0E10_pythia8_cfi data/
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
        sample_dir = "SinglePi0E10_pythia8_cfi"
        return cms_utils.split_sample(path/sample_dir/"raw", PADDED_NUM_ELEM_SIZE)

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files, PADDED_NUM_ELEM_SIZE)
