from pathlib import Path

import tensorflow as tf
from utils_edm import X_FEATURES_CL, X_FEATURES_TRK, Y_FEATURES, generate_examples, split_sample, NUM_SPLITS

import tensorflow_datasets as tfds

_DESCRIPTION = """
CLIC EDM4HEP dataset with Z->tautau
  - X: reconstructed tracks and clusters, variable number N per event
  - ygen: stable generator particles, zero-padded to N per event
  - ycand: baseline particle flow particles, zero-padded to N per event
"""

_CITATION = """
"""


class ClicEdmZTautauPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.5.0")
    RELEASE_NOTES = {
        "1.3.0": "First version",
        "1.4.0": "Fix ycand matching",
        "1.5.0": "Regenerate with ARRAY_RECORD",
        "2.1.0": "Add ispu, genjets, genmet; disable genjet_idx; truth def not based on gp.status==1",
        "2.3.0": "Fix target/truth momentum, st=1 more inclusive: PR352",
        "2.5.0": "Use 10 splits, skip 2.4.0 to unify with CMS datasets",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    For the raw input files in ROOT EDM4HEP format, please see the citation above.

    The processed tensorflow_dataset can also be downloaded from:
    rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/clic_edm4hep/ ./
    """

    # create configs 1 ... NUM_SPLITS + 1 that allow to parallelize the dataset building
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(ClicEdmZTautauPf, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(
                        shape=(
                            None,
                            max(len(X_FEATURES_TRK), len(X_FEATURES_CL)),
                        ),
                        dtype=tf.float32,
                    ),
                    "ytarget": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=tf.float32),
                    "genmet": tfds.features.Scalar(dtype=tf.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=tf.float32),
                }
            ),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return split_sample(Path(path / "p8_ee_Z_Ztautau_ecm380/"), self.builder_config, num_splits=NUM_SPLITS)

    def _generate_examples(self, files):
        return generate_examples(files)
