"""MAIA ttbar sample in the shared EDM4hep track/cluster tensor schema."""
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.edm4hep_utils.utils_pf import (
    NUM_SPLITS, N_X_FEATURES, N_Y_FEATURES, X_FEATURES_TRK, X_FEATURES_CL,
    Y_FEATURES, generate_examples, split_sample,
)


class MaiaEdmTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "MAIA ttbar track/cluster dataset with shared target building."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download https://uaf-3.t2.ucsd.edu/~atuna/muoncollider/data/mlpf/ttbar/v04/ttbar_reco_10000.slcio.edm4hep.root
    and convert with mlpf.data.key4hep.postprocessing --detector maia.
    Supply converted parquet files directly in manual_dir. Config 10 supports
    two files with disjoint events (one train and one test file).
    """
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="MAIA muon-collider ttbar: reconstructed tracks/clusters and particle-flow targets/candidates.",
            features=tfds.features.FeaturesDict({
                "X": tfds.features.Tensor(shape=(None, N_X_FEATURES), dtype=np.float32),
                "ytarget": tfds.features.Tensor(shape=(None, N_Y_FEATURES), dtype=np.float32),
                "ycand": tfds.features.Tensor(shape=(None, N_Y_FEATURES), dtype=np.float32),
                "genmet": tfds.features.Scalar(dtype=np.float32),
                "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
            }),
            homepage="https://github.com/jpata/particleflow",
            metadata=tfds.core.MetadataDict(x_features_track=X_FEATURES_TRK, x_features_cluster=X_FEATURES_CL, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager):
        return split_sample(Path(dl_manager.manual_dir), self.builder_config, num_splits=NUM_SPLITS)

    def _generate_examples(self, files):
        return generate_examples(files)
