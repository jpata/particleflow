"""CMS PF TTbar dataset."""

from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

from heptfds import cms_utils

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

TTbar events with PU~55 in a Run3 setup.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""

PADDED_NUM_ELEM_SIZE = 6400

class CmsPfTtbar(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf dataset."""

    VERSION = tfds.core.Version("1.3.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Add muon type, fix electron GSF association",
        "1.2.0": "12_1_0_pre3 generation, add corrected energy, cluster flags, 20k events",
        "1.3.0": "12_2_0_pre2 generation with updated caloparticle/trackingparticle",
        "1.3.1": "Remove PS again"
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    mkdir -p data
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi data/
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
        sample_dir = "TTbar_14TeV_TuneCUETP8M1_cfi"
        return cms_utils.split_sample(path/sample_dir/"raw", PADDED_NUM_ELEM_SIZE)

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files, PADDED_NUM_ELEM_SIZE)
