"""IDEA ttbar track/cluster dataset using temporary truth-seeded tracks."""

from pathlib import Path
import os

import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.edm4hep_utils.utils_idea import (
    CANDIDATE_SOURCE,
    TRACK_SOURCE,
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    find_idea_parquets,
    generate_examples,
    split_event_references,
)

_DESCRIPTION = """
IDEA_o1_v03 ee -> ttbar at 365 GeV, for MLPF pipeline validation.
Tracks are truth-seeded proxies and ycand is a target oracle. Neither should
be interpreted as reconstructed-physics performance.
"""


class IdeaEdmTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "0.1.0"))
    RELEASE_NOTES = {"0.1.0": "Initial pipeline-validation dataset with explicit proxy provenance."}
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name="1")]
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Point --manual_dir at a directory containing the IDEA parquet files."

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(
                        shape=(None, max(len(X_FEATURES_TRK), len(X_FEATURES_CL))),
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
            metadata=tfds.core.MetadataDict(
                x_features_track=X_FEATURES_TRK,
                x_features_cluster=X_FEATURES_CL,
                y_features=Y_FEATURES,
                track_source=TRACK_SOURCE,
                candidate_source=CANDIDATE_SOURCE,
                suitable_for_physics=False,
            ),
        )

    def _split_generators(self, dl_manager):
        files = find_idea_parquets(Path(dl_manager.manual_dir), process_name="p8_ee_ttbar_ecm365")
        return {name: self._generate_examples(refs) for name, refs in split_event_references(files).items()}

    def _generate_examples(self, references):
        yield from generate_examples(references)
