"""CMS PF TTbar dataset."""
import cms_utils
import tensorflow as tf
import tensorflow_datasets as tfds

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

TTbar events with PU~55 in a Run3 setup.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""


class CmsPfTtbar(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for cms_pf_ttbar dataset."""

    VERSION = tfds.core.Version("2.5.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Add muon type, fix electron GSF association",
        "1.2.0": "12_1_0_pre3 generation, add corrected energy, cluster flags, 20k events",
        "1.3.0": "12_2_0_pre2 generation with updated caloparticle/trackingparticle",
        "1.3.1": "Remove PS again",
        "1.4.0": "Add gen jet index information",
        "1.5.0": "No padding",
        "1.5.1": "Remove outlier caps",
        "1.6.0": "Regenerate with ARRAY_RECORD",
        "1.7.0": "Add cluster shape vars",
        "1.7.1": "Increase stats to 400k events",
        "1.8.0": "Add ispu, genjets, genmet; disable genjet_idx; improved merging",
        "2.0.0": "New truth def based primarily on CaloParticles",
        "2.1.0": "Additional stats",
        "2.2.0": "Split CaloParticles along tracks",
        "2.3.0": "Increase stats",
        "2.4.0": "Add gp_to_track, gp_to_cluster, jet_idx",
        "2.5.0": "Remove neutrinos from genjets, split to 10",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/cms/cms_pf_ttbar ~/tensorflow_datasets/
    """

    # create configs 1 ... NUM_SPLITS + 1 that allow to parallelize the dataset building
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, cms_utils.NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CmsPfTtbar, self).__init__(*args, **kwargs)

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
        sample_dir = "TTbar_14TeV_TuneCUETP8M1_cfi"
        return cms_utils.split_sample(path / sample_dir / "raw", self.builder_config, num_splits=cms_utils.NUM_SPLITS)

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files)
