"""delphes_pf dataset."""

import bz2
import os
import pickle
import resource
from pathlib import Path

import numpy as np
import tensorflow as tf
import tqdm

import tensorflow_datasets as tfds

# Increase python's soft limit on number of open files to accomodate tensorflow_datasets sharding
# https://github.com/tensorflow/datasets/issues/1441
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


_DESCRIPTION = """
Dataset generated with Delphes.

TTbar and QCD events with PU~200.
"""

# TODO(delphes_pf): BibTeX citation
_CITATION = """
"""

DELPHES_CLASS_NAMES = ["none" "charged hadron", "neutral hadron", "hfem", "hfhad", "photon", "electron", "muon"]

# based on delphes/ntuplizer.py
X_FEATURES = [
    "typ_idx",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "e",
    "eta_outer",
    "sin_phi_outer",
    "cos_phi_outer",
    "charge",
    "is_gen_muon",
    "is_gen_electron",
]


class DelphesPf(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for delphes_pf dataset."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.1.0": "Do not pad events to the same size",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(delphes_pf): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "X": tfds.features.Tensor(shape=(None, 12), dtype=tf.float32),
                    "ygen": tfds.features.Tensor(shape=(None, 7), dtype=tf.float32),
                    "ycand": tfds.features.Tensor(shape=(None, 7), dtype=tf.float32),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("X", "ygen"),  # Set to `None` to disable
            homepage="",
            citation=_CITATION,
            metadata=tfds.core.MetadataDict(x_features=X_FEATURES),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        delphes_dir = dl_manager.download_dir / "delphes_pf"
        if delphes_dir.exists():
            print("INFO: Data already exists. Please delete {} if you want to download data again.".format(delphes_dir))
        else:
            get_delphes_from_zenodo(download_dir=dl_manager.download_dir / "delphes_pf")

        ttbar_dir = delphes_dir / "pythia8_ttbar/raw"
        qcd_dir = delphes_dir / "pythia8_qcd/val"

        if not ttbar_dir.exists():
            ttbar_dir.mkdir(parents=True)
            for ttbar_file in delphes_dir.glob("*ttbar*.pkl.bz2"):
                ttbar_file.rename(ttbar_dir / ttbar_file.name)
        if not qcd_dir.exists():
            qcd_dir.mkdir(parents=True)
            for qcd_file in delphes_dir.glob("*qcd*.pkl.bz2"):
                qcd_file.rename(qcd_dir / qcd_file.name)

        return {
            "train": self._generate_examples(delphes_dir / "pythia8_ttbar/raw"),
            "test": self._generate_examples(delphes_dir / "pythia8_qcd/val"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for fi in tqdm.tqdm(list(path.glob("*.pkl.bz2"))):
            Xs, ygens, ycands = self.prepare_data_delphes(str(fi))
            for iev in range(len(Xs)):
                yield str(fi) + "_" + str(iev), {
                    "X": Xs[iev],
                    "ygen": ygens[iev],
                    "ycand": ycands[iev],
                }

    def prepare_data_delphes(self, fname):

        if fname.endswith(".pkl"):
            data = pickle.load(open(fname, "rb"))
        elif fname.endswith(".pkl.bz2"):
            data = pickle.load(bz2.BZ2File(fname, "rb"))
        else:
            raise Exception("Unknown file: {}".format(fname))

        # make all inputs and outputs the same size with padding
        Xs = []
        ygens = []
        ycands = []
        for i in range(len(data["X"])):
            X = data["X"][i].astype(np.float32)
            ygen = data["ygen"][i].astype(np.float32)
            ycand = data["ycand"][i].astype(np.float32)
            Xs.append(X)
            ygens.append(ygen)
            ycands.append(ycand)

        return Xs, ygens, ycands


def get_delphes_from_zenodo(download_dir="."):
    # url = 'https://zenodo.org/record/4559324'
    zenodo_doi = "10.5281/zenodo.4559324"
    print("Downloading data from {} to {}".format(zenodo_doi, download_dir))
    os.system("zenodo_get -d {} -o {}".format(zenodo_doi, download_dir))
    return Path(download_dir)
