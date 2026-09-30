import os
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from mlpf.heptfds.colliderml_hits.utils import NUM_SPLITS, X_FEATURES, Y_FEATURES, generate_examples, split_sample

_DESCRIPTION = """
ColliderML Release 1 (ttbar, no pileup) converted for MLPF training.
Hits view: raw tracker + calorimeter hits carry the input elements; ytarget rows live on the
hits (one exclusive representative hit per target, particle_number marks on related hits).
"""

_CITATION = """
Elitez, Murnane, Gessinger et al. (2025). ColliderML: The First Release of an
OpenDataDetector High-Luminosity Physics Benchmark Dataset. arXiv:2512.15230.
DOI 10.57967/hf/7269. CC-BY-4.0.
"""


class CollidermlTtbarHits(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "1.0.0"))
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    DESCRIPTION = _DESCRIPTION
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    ColliderML release 1 is downloaded on Ceph under
    /mnt/ceph/users/ewulff/data/colliderml/CERN__ColliderML-Release-1/ at pinned revision
    64c3d2f112df3d5d20979d22da7cfdff13e10c4b. The intermediate MLPF parquet is produced by
    `python3 -m mlpf.data.colliderml.postprocessing --input <source_dir> --outpath <dir>`
    which this builder consumes from `dl_manager.manual_dir` via tfds's standard manual-dir
    convention.
    """

    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(group)) for group in range(1, NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super(CollidermlTtbarHits, self).__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.DESCRIPTION,
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
        return split_sample(Path(dl_manager.manual_dir), self.builder_config, num_splits=NUM_SPLITS)

    def _generate_examples(self, files):
        return generate_examples(files)
