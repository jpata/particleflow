"""CMS PF TTbar dataset."""
import cms_utils
import tensorflow as tf
import tensorflow_datasets as tfds

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

Ztautau, all-hadronic events with PU 55-75 in a Run3 setup.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""


class CmsPfZttNopu(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_ztt_nopu dataset."""

    VERSION = tfds.core.Version("2.5.0")
    RELEASE_NOTES = {
        "2.5.0": "Remove neutrinos from genjets, split to 10",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/cms/cms_pf_ztt ~/tensorflow_datasets/
    """

    # create configs 1 ... NUM_SPLITS + 1 that allow to parallelize the dataset building
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, cms_utils.NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CmsPfZttNopu, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=tf.float32),
                    "ytarget": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "genmet": tfds.features.Scalar(dtype=tf.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
                }
            ),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        sample_dir = "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi"
        return cms_utils.split_sample(path / sample_dir / "raw", self.builder_config, num_splits=cms_utils.NUM_SPLITS)

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files)
