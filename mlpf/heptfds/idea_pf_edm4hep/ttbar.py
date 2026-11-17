"""IDEA ttbar track/cluster dataset using GGTF/GenFit tracks."""

from pathlib import Path
import os

import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.edm4hep_utils.utils_idea import (
    CANDIDATE_SOURCE,
    TRACK_SOURCE,
    PROXY_TRACK_SOURCE,
    TRACK_COLLECTION,
    TRACK_QUALITY_POLICY,
    TRACK_TRUTH_POLICY,
    TRACK_TRUTH_MIN_PURITY,
    X_FEATURES_CL,
    X_FEATURES_TRK,
    Y_FEATURES,
    find_idea_parquets,
    generate_examples,
    split_event_references,
)

_DESCRIPTION = """
IDEA_o1_v03 ee -> ttbar at 365 GeV, for MLPF pipeline validation.
Versions 0.4.0 and later use quality-selected GGTF/GenFit tracks and hit-majority truth
labels. ycand is still a target oracle, not an independent physics baseline.
Versions below 0.4.0 use truth-seeded proxy tracks.
"""


class IdeaEdmTtbarPf(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "0.5.0"))
    RELEASE_NOTES = {
        "0.5.0": "Regenerated with key4hep-sim v1.4.0 and stable Key4hep 2026-04-08 for the 50k-event campaign.",
        "0.4.0": "November 2026 upstream simulation/reconstruction and dual-readout patches; measured GGTF/GenFit tracks with seed-orientation and fitted-charge fixes, quality cuts and hit-majority labels; regenerate reconstruction, Parquet and TFDS.",
        "0.1.0": "Initial pipeline-validation dataset with explicit proxy provenance.",
        "0.2.0": "Add num_muon_hits and energy_muon cluster features (#511); input width 17 -> 19, regenerate",
        "0.2.1": "Compute cluster shower widths from associated calorimeter cells.",
        "0.3.0": "Track phi, D0 and Z0 as the perigee relative to the origin (CLIC/CLD convention; was phi at the production vertex for secondaries), radiusOfInnermostHit from the AtFirstHit state.",
    }
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
                track_source=TRACK_SOURCE if self.version >= tfds.core.Version("0.4.0") else PROXY_TRACK_SOURCE,
                track_collection=TRACK_COLLECTION if self.version >= tfds.core.Version("0.4.0") else "TracksFromGenParticles",
                track_quality_policy=TRACK_QUALITY_POLICY if self.version >= tfds.core.Version("0.4.0") else "legacy_proxy",
                track_truth_policy=TRACK_TRUTH_POLICY if self.version >= tfds.core.Version("0.4.0") else "legacy_proxy",
                track_truth_min_purity=TRACK_TRUTH_MIN_PURITY if self.version >= tfds.core.Version("0.4.0") else 0.0,
                candidate_source=CANDIDATE_SOURCE,
                suitable_for_physics=False,
            ),
        )

    def _split_generators(self, dl_manager):
        files = find_idea_parquets(Path(dl_manager.manual_dir), process_name="p8_ee_ttbar_ecm365")
        return {name: self._generate_examples(refs) for name, refs in split_event_references(files).items()}

    def _generate_examples(self, references):
        source = TRACK_SOURCE if self.version >= tfds.core.Version("0.4.0") else PROXY_TRACK_SOURCE
        yield from generate_examples(references, expected_track_source=source)
