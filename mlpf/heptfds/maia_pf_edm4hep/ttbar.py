"""MAIA ttbar sample in the shared EDM4hep track/cluster tensor schema."""

from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from mlpf.conf import EDM4HEP
from mlpf.heptfds.edm4hep_utils.utils_pf import (
    NUM_SPLITS,
    N_X_FEATURES,
    N_Y_FEATURES,
    X_FEATURES_CL,
    Y_FEATURES,
    generate_examples,
    split_sample,
)

# MAIA tracks carry their calorimeter entry (eta_ecal, phi_ecal) after the shared track columns
X_FEATURES_TRK = EDM4HEP.DETECTORS["maia"].track_feature_names()


class MaiaEdmTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.3.0")
    RELEASE_NOTES = {
        "1.0.0": "MAIA ttbar track/cluster dataset with shared target building.",
        "1.1.0": "Add num_muon_hits and energy_muon cluster features (#511); input width 17 -> 19, regenerate",
        "1.2.0": "Track eta_ecal/phi_ecal, cluster eta/phi from the centroid, majority-energy target assignment (#546); regenerate",
        "1.3.0": "MuColl v3.2 reconstruction (MAIAConfig): SiTracks selected from AllTracks, dEdx 0 (no dQdx), tracker hit system/side/layer "
        "from the sim CellIDEncoding, muon hits from MuonBarrelHits/MuonEndcapHits; regenerate",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Download https://uaf-3.t2.ucsd.edu/~atuna/muoncollider/data/mlpf/57b1d7f2/ttbar_reco_100000.edm4hep.root
    (MuColl v3.2) and convert with mlpf.data.key4hep.postprocessing --detector maia. MuColl v2.9 files,
    such as .../ttbar/v04/ttbar_reco_10000.slcio.edm4hep.root (up to 1.2.0), convert with --detector maia_v29.
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
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, N_X_FEATURES), dtype=np.float32),
                    "ytarget": tfds.features.Tensor(shape=(None, N_Y_FEATURES), dtype=np.float32),
                    "ycand": tfds.features.Tensor(shape=(None, N_Y_FEATURES), dtype=np.float32),
                    "genmet": tfds.features.Scalar(dtype=np.float32),
                    "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                    "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                }
            ),
            homepage="https://github.com/jpata/particleflow",
            metadata=tfds.core.MetadataDict(x_features_track=X_FEATURES_TRK, x_features_cluster=X_FEATURES_CL, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager):
        return split_sample(Path(dl_manager.manual_dir), self.builder_config, num_splits=NUM_SPLITS)

    def _generate_examples(self, files):
        return generate_examples(files)
