"""CMS PF SinglePi dataset."""
import cms_utils
import tensorflow as tf
import tensorflow_datasets as tfds

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

SinglePi events.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""


class CmsPfSinglePi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_singlepi dataset."""

    VERSION = tfds.core.Version("1.5.1")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Add muon type, fix electron GSF association",
        "1.2.0": "12_1_0_pre3 generation, add corrected energy, cluster flags, 20k events",
        "1.4.0": "Add genjet information",
        "1.5.0": "Without padding",
        "1.5.1": "Remove outlier caps",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/SinglePiMinusFlatPt0p7To1000_cfi data/
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cms_pf): Specifies the tfds.core.DatasetInfo object
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
            supervised_keys=("X", "ycand"),
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        sample_dir = "SinglePiMinusFlatPt0p7To1000_cfi"
        return cms_utils.split_sample(path / sample_dir / "raw")

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files)
