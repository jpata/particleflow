"""CMS PF TTbar dataset."""
import cms_utils
import tensorflow as tf

import tensorflow_datasets as tfds

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

SMS-T1tttt events with PU~55 in a Run3 setup.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""


class CmsPfSmsT1tttt(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf dataset."""

    VERSION = tfds.core.Version("1.7.0")
    RELEASE_NOTES = {
        "1.6.0": "Regenerate with ARRAY_RECORD",
        "1.7.0": "Add cluster shape vars",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress \
        lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/cms/cms_pf_sms_t1tttt \
        ~/tensorflow_datasets/
    """

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CmsPfSmsT1tttt, self).__init__(*args, **kwargs)

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
        sample_dir = "SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi"
        return cms_utils.split_sample(path / sample_dir / "raw")

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files)
