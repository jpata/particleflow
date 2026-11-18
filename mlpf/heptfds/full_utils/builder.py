"""Common ArrayRecord builder for the full reconstruction view."""

import os

import numpy as np
import tensorflow_datasets as tfds

from mlpf.data.full_targets import ACCOUNTING_FEATURES, ELEMENT_KINDS, INSTANCE_FEATURES, LABEL_FEATURES, ORIGIN_NAMES

from .utils import CLASS_LABELS, X_FEATURES, Y_FEATURES, generate_examples, split_files


class FullBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "1.0.0"))
    RELEASE_NOTES = {
        "1.0.0": "Tracks, clusters, tracker and calorimeter hits with one cluster-independent target table and "
        "per-element particle, instance and origin labels. Replaces the *_hits and *_track_calo views."
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Convert source data with the current key4hep postprocessor. Place event-disjoint
    Parquet of one physics sample directly in manual_dir (sorted file split 90/10),
    or in train/ and test/. Use config 10 for small samples.
    """
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(i)) for i in range(1, 11)]
    DETECTOR = None
    PROCESS = None

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self):
        features = {}
        for kind in ELEMENT_KINDS:
            features[f"X_{kind}"] = tfds.features.Tensor(shape=(None, len(X_FEATURES[kind])), dtype=np.float32)
            features[f"ylabel_{kind}"] = tfds.features.Tensor(shape=(None, len(LABEL_FEATURES)), dtype=np.float32)
        features.update(
            {
                "ytarget_particles": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                "target_instances": tfds.features.Tensor(shape=(None, len(INSTANCE_FEATURES)), dtype=np.float32),
                "energy_accounting": tfds.features.Tensor(shape=(len(ACCOUNTING_FEATURES),), dtype=np.float32),
                "genmet": tfds.features.Scalar(dtype=np.float32),
                "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=f"{self.DETECTOR} {self.PROCESS}: tracks, clusters, tracker hits and calorimeter hits with "
            "cluster-independent particle targets and per-element truth labels.",
            features=tfds.features.FeaturesDict(features),
            homepage="https://github.com/jpata/particleflow",
            metadata=tfds.core.MetadataDict(
                detector=self.DETECTOR,
                sample=self.PROCESS,
                input_type="full",
                element_kinds=list(ELEMENT_KINDS),
                x_features=X_FEATURES,
                y_features=Y_FEATURES,
                label_features=LABEL_FEATURES,
                instance_features=INSTANCE_FEATURES,
                accounting_features=ACCOUNTING_FEATURES,
                origin_codes={str(code): name for code, name in ORIGIN_NAMES.items()},
                class_labels=CLASS_LABELS,
                target_assignment="full_v1",
                units={"momentum_energy": "GeV", "position_D0_Z0_radius": "mm", "omega": "1/mm", "time": "ns"},
            ),
        )

    def _split_generators(self, dl_manager):
        return {name: self._generate_examples(files) for name, files in split_files(dl_manager.manual_dir, self.builder_config.name).items()}

    def _generate_examples(self, files):
        yield from generate_examples(files)
