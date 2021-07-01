"""cms_pf dataset."""

from pathlib import Path
import numpy as np
import pickle
import bz2
import tensorflow as tf
import tensorflow_datasets as tfds

from numpy.lib.recfunctions import append_fields


# TODO(cms_pf): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Dataset generated with CMSSW and full detector sim.

TTbar events with PU~55 in a Run3 setup.
"""

# TODO(cms_pf): BibTeX citation
_CITATION = """
"""

CMS_PF_CLASS_NAMES = ["none" "charged hadron", "neutral hadron", "hfem", "hfhad", "photon", "electron", "muon"]

ELEM_LABELS_CMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# ch.had, n.had, HFEM, HFHAD, gamma, ele, mu
CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13]
PADDED_NUM_ELEM_SIZE = 6400


class CmsPf(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cms_pf dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Ask jpata for the data and place it in <your_dir>. Then build the dataset using 
    `tfds build <path_to_heptfds>/heptfds/cms_pf --manual_dir <your_dir>`. Alternatively,
    load the dataset using tfds.load() and give the argument
    `download_and_prepare_kwargs={"download_config": download_config}` where `download_config`
    is a tfds.download.DownloadConfig with manual_dir set to <your_dir>.
    """


    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(cms_pf): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(1, 6400, 15), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(1, 6400, 7), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(1, 6400, 7), dtype=tf.float32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("X", "ycand"),  # Set to `None` to disable
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        return {"train": self._generate_examples(path / "raw"), "test": self._generate_examples(path / "val")}

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(cms_pf): Yields (key, example) tuples from the dataset
        for fi in path.glob("*.pkl.bz2"):
            X, ygen, ycand = self.prepare_data_cms(str(fi))
            #   yield str(fi), {
            #       'X': X,
            #       'ygen': ygen,
            #       'ycand': ycand,
            #   }
            for ii in range(X[0].shape[0]):
                x = [X[0][ii]]
                yg = [ygen[0][ii]]
                yc = [ycand[0][ii]]
                yield str(fi) + "_" + str(ii), {
                    "X": x,
                    "ygen": yg,
                    "ycand": yc,
                    # 'X': np.expand_dims(X[ii], 0),
                    # 'ygen': np.expand_dims(ygen[ii], 0),
                    # 'ycand': np.expand_dims(ycand[ii], 0),
                }

    def prepare_data_cms(self, fn):
        Xs = []
        ygens = []
        ycands = []

        if fn.endswith(".pkl"):
            data = pickle.load(open(fn, "rb"), encoding="iso-8859-1")
        elif fn.endswith(".pkl.bz2"):
            data = pickle.load(bz2.BZ2File(fn, "rb"))

        for event in data:
            Xelem = event["Xelem"]
            ygen = event["ygen"]
            ycand = event["ycand"]

            # remove PS from inputs, they don't seem to be very useful
            msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3)

            Xelem = Xelem[~msk_ps]
            ygen = ygen[~msk_ps]
            ycand = ycand[~msk_ps]

            Xelem = append_fields(
                Xelem, "typ_idx", np.array([ELEM_LABELS_CMS.index(int(i)) for i in Xelem["typ"]], dtype=np.float32)
            )
            ygen = append_fields(
                ygen, "typ_idx", np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32)
            )
            ycand = append_fields(
                ycand,
                "typ_idx",
                np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32),
            )

            Xelem_flat = np.stack(
                [
                    Xelem[k].view(np.float32).data
                    for k in [
                        "typ_idx",
                        "pt",
                        "eta",
                        "phi",
                        "e",
                        "layer",
                        "depth",
                        "charge",
                        "trajpoint",
                        "eta_ecal",
                        "phi_ecal",
                        "eta_hcal",
                        "phi_hcal",
                        "muon_dt_hits",
                        "muon_csc_hits",
                    ]
                ],
                axis=-1,
            )
            ygen_flat = np.stack(
                [
                    ygen[k].view(np.float32).data
                    for k in [
                        "typ_idx",
                        "charge",
                        "pt",
                        "eta",
                        "sin_phi",
                        "cos_phi",
                        "e",
                    ]
                ],
                axis=-1,
            )
            ycand_flat = np.stack(
                [
                    ycand[k].view(np.float32).data
                    for k in [
                        "typ_idx",
                        "charge",
                        "pt",
                        "eta",
                        "sin_phi",
                        "cos_phi",
                        "e",
                    ]
                ],
                axis=-1,
            )

            # take care of outliers
            Xelem_flat[np.isnan(Xelem_flat)] = 0
            Xelem_flat[np.abs(Xelem_flat) > 1e4] = 0
            ygen_flat[np.isnan(ygen_flat)] = 0
            ygen_flat[np.abs(ygen_flat) > 1e4] = 0
            ycand_flat[np.isnan(ycand_flat)] = 0
            ycand_flat[np.abs(ycand_flat) > 1e4] = 0

            X = Xelem_flat[:PADDED_NUM_ELEM_SIZE]
            X = np.pad(X, [(0, PADDED_NUM_ELEM_SIZE - X.shape[0]), (0, 0)])

            ygen = ygen_flat[:PADDED_NUM_ELEM_SIZE]
            ygen = np.pad(ygen, [(0, PADDED_NUM_ELEM_SIZE - ygen.shape[0]), (0, 0)])

            ycand = ycand_flat[:PADDED_NUM_ELEM_SIZE]
            ycand = np.pad(ycand, [(0, PADDED_NUM_ELEM_SIZE - ycand.shape[0]), (0, 0)])

            X = np.expand_dims(X, 0)
            ygen = np.expand_dims(ygen, 0)
            ycand = np.expand_dims(ycand, 0)

            Xs.append(X)
            ygens.append(ygen)
            ycands.append(ycand)

        X = [np.concatenate(Xs)]
        ygen = [np.concatenate(ygens)]
        ycand = [np.concatenate(ycands)]

        return X, ygen, ycand
