from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from mlpf.heptfds.edm4hep_utils.utils_pf import (
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
)

from mlpf.heptfds.edm4hep_utils.utils_ild import generate_examples, split_sample

_DESCRIPTION = """
ILD@FCC-ee EDM4HEP dataset with ee -> qq at 365 GeV.
  - X: reconstructed tracks and clusters, variable number N per event
  - ytarget: visible generator particles assigned and merged onto PF elements, zero-padded to N per event
  - ycand: baseline particle flow particles, zero-padded to N per event
"""

_CITATION = ""


class IldEdmQqPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {"0.1.0": "ILD_FCCee_v02 qualification with fitted Clupatra tracks and Pandora clusters/PFOs"}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Run the ILD generation and postprocessing recipe; pass its post directory as manual_dir."
    # One configuration, with train and test drawn from independent generation jobs.
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, 2)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(IldEdmQqPf, self).__init__(*args, **kwargs)

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
                        dtype=np.float32,
                    ),
                    "ytarget": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                    "genmet": tfds.features.Scalar(dtype=np.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
                detector="ILD_FCCee_v02",
                track_source="clupatra_kaltest",
                target_policy="ild-direct-hit-majority-v1",
                candidate_source="pandora",
                qualification="experimental; see docs/datasets/ild.md",
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return split_sample(Path(path / "p8_ee_qq_ecm365"), self.builder_config)

    def _generate_examples(self, files):
        return generate_examples(files)
