"""CMS Phase-2 (TICL, v2 target) ttbar 200PU dataset (isPU=-1 inputs dropped)."""
import os
import cms_phase2_utils
import numpy as np
import tensorflow_datasets as tfds

X_FEATURES = cms_phase2_utils.X_FEATURES
from mlpf.conf import Y_FEATURES  # noqa: E402

_DESCRIPTION = "CMS Phase-2 HGCAL/TICL NanoAOD, v2 target (tracks + CLUE3D tracksters). ttbar 200 PU (isPU=-1 inputs dropped)."
_CITATION = ""
# layout: <manual_dir>/ttbar_200pu/*.pkl  (manual_dir = data/cms/phase2/offline/pkl, v2 ticl_ pkls).
# Override the subdir with PHASE2_PKL_SUBDIR if the pkls live one level deeper.
_SAMPLE_DIR = os.path.normpath(os.path.join("ttbar_200pu", os.environ.get("PHASE2_PKL_SUBDIR", ".")))
_CLASS_NAME = "CmsPfPhase2Ttbar200pu"


class CmsPfPhase2Ttbar200pu(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf_phase2_ttbar_200pu."""

    VERSION = tfds.core.Version(os.environ.get("TFDS_VERSION", "1.0.0"))
    RELEASE_NOTES = {"1.0.0": "First Phase-2 200PU v2 target (isPU=-1 inputs dropped)"}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Point --manual_dir at data/cms/phase2/offline/pkl"
    BUILDER_CONFIGS = [tfds.core.BuilderConfig(name=str(g)) for g in range(1, cms_phase2_utils.NUM_SPLITS + 1)]

    def __init__(self, *args, **kwargs):
        kwargs["file_format"] = tfds.core.FileFormat.ARRAY_RECORD
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "X": tfds.features.Tensor(shape=(None, len(X_FEATURES)), dtype=np.float32),
                "ytarget": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                "ycand": tfds.features.Tensor(shape=(None, len(Y_FEATURES)), dtype=np.float32),
                "genmet": tfds.features.Scalar(dtype=np.float32),
                "genjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "targetjets": tfds.features.Tensor(shape=(None, 4), dtype=np.float32),
                "pythia": tfds.features.Tensor(shape=(None, 5), dtype=np.float32),
            }),
            homepage="https://github.com/jpata/particleflow",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES, y_features=Y_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return cms_phase2_utils.split_sample(dl_manager.manual_dir / _SAMPLE_DIR, self.builder_config)

    def _generate_examples(self, files):
        return cms_phase2_utils.generate_examples(files)
