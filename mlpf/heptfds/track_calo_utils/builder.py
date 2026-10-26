"""Common ArrayRecord builder for the track/calo view."""

import numpy as np
import tensorflow_datasets as tfds

from .utils import CLASS_LABELS, X_FEATURES, Y_FEATURES, feature_availability, generate_examples, split_files


class TrackCaloBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Independent reconstructed-track/calorimeter-hit targets; common 25-column input schema."}
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Convert source data using --track-calo-targets. Place event-disjoint Parquet
    directly in manual_dir (sorted file split 90/10), or in train/ and test/.
    Use config 10 for small samples. Do not split alternate views of an event
    across training and test. Legacy-only Parquet is rejected.
    """
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(i)) for i in range(1, 11)]
    DETECTOR = None

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"{self.DETECTOR}: reconstructed tracks plus calorimeter hits, with independent PF targets.",
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
            metadata=tfds.core.MetadataDict(
                detector=self.DETECTOR,
                input_type="track_calo",
                x_features=X_FEATURES,
                y_features=Y_FEATURES,
                class_labels=CLASS_LABELS,
                element_types={"0": "padding", "1": "reconstructed_track", "2": "calorimeter_hit"},
                feature_availability=feature_availability(self.DETECTOR),
                target_assignment="track_calo_v1",
                input_schema="track_calo_25_v1",
                has_reco_candidates=False,
                units={"momentum_energy": "GeV", "position_D0_Z0_radius": "mm", "omega": "1/mm", "time": "ns"},
            ),
        )

    def _split_generators(self, dl_manager):
        return {name: self._generate_examples(files) for name, files in split_files(dl_manager.manual_dir, self.builder_config.name).items()}

    def _generate_examples(self, files):
        yield from generate_examples(files, self.DETECTOR)
