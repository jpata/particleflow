"""CMS PF QCD 13p6 dataset."""
import os

import cms_utils
import numpy as np
import tensorflow_datasets as tfds

X_FEATURES = cms_utils.X_FEATURES
Y_FEATURES = cms_utils.Y_FEATURES

_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.
QCD events with PU 55~75 in a Run3 setup, 13.6 TeV.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""


class CmsPfQcd13p6(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_qcd dataset."""

    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "2.8.0"))
    RELEASE_NOTES = {
        "2.6.0": "First version",
        "2.8.0": "Add Pythia",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/cms/cms_pf_qcd ~/tensorflow_datasets/
    """

    # create configs 1 ... NUM_SPLITS + 1 that allow to parallelize the dataset building
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, cms_utils.NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CmsPfQcd13p6, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=np.float32),
                    "ytarget": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "genmet": tfds.features.Scalar(dtype=np.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                }
            ),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        sample_dir = "QCDForPF_13p6TeV_TuneCUETP8M1_cfi"
        return cms_utils.split_sample(path / sample_dir / "raw", self.builder_config, num_splits=cms_utils.NUM_SPLITS)

    def _generate_examples(self, files):
        return cms_utils.generate_examples(files)
