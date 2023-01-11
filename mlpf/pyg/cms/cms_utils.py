import bz2
import pickle

import numpy as np
import torch
from numpy.lib.recfunctions import append_fields
from torch_geometric.data import Data

"""Based on https://github.com/jpata/hep_tfds/blob/master/heptfds/cms_utils.py#L10"""

# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
ELEM_LABELS_CMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ELEM_NAMES_CMS = [
    "NONE",
    "TRACK",
    "PS1",
    "PS2",
    "ECAL",
    "HCAL",
    "GSF",
    "BREM",
    "HFEM",
    "HFHAD",
    "SC",
    "HO",
]

# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13, 15]
CLASS_NAMES_CMS_LATEX = [
    "none",
    "chhad",
    "nhad",
    "HFEM",
    "HFHAD",
    r"$\gamma$",
    r"$e^\pm$",
    r"$\mu^\pm$",
    r"$\tau$",
]
CLASS_NAMES_CMS = [
    "none",
    "chhad",
    "nhad",
    "HFEM",
    "HFHAD",
    "gamma",
    "ele",
    "mu",
    "tau",
]

CLASS_NAMES_LONG_CMS = [
    "none" "charged hadron",
    "neutral hadron",
    "hfem",
    "hfhad",
    "photon",
    "electron",
    "muon",
    "tau",
]

CMS_PF_CLASS_NAMES = [
    "none" "charged hadron",
    "neutral hadron",
    "hfem",
    "hfhad",
    "photon",
    "electron",
    "muon",
]

X_FEATURES_CMS = [
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
    "muon_type",
    "px",
    "py",
    "pz",
    "deltap",
    "sigmadeltap",
    "gsf_electronseed_trkorecal",
    "gsf_electronseed_dnn1",
    "gsf_electronseed_dnn2",
    "gsf_electronseed_dnn3",
    "gsf_electronseed_dnn4",
    "gsf_electronseed_dnn5",
    "num_hits",
    "cluster_flags",
    "corr_energy",
    "corr_energy_err",
    "vx",
    "vy",
    "vz",
    "pterror",
    "etaerror",
    "phierror",
    "lambd",
    "lambdaerror",
    "theta",
    "thetaerror",
]

Y_FEATURES = [
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "e",
]


def prepare_data_cms(fn):
    """
    Takes as input a bz2 file that contains the cms raw information, and returns a list of PyG Data() objects.
    Each element of the list looks like this ~ Data(x=[#, 41], ygen=[#, 6], ygen_id=[#, 9], ycand=[#, 6], ycand_id=[#, 9])

    Args
        raw_file_name: raw parquet data file.
    Returns
        list of Data() objects.
    """

    batched_data = []

    data = pickle.load(bz2.BZ2File(fn, "rb"))

    for event in data:
        Xelem = event["Xelem"]
        ygen = event["ygen"]
        ycand = event["ycand"]

        # remove PS and BREM from inputs
        msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3) | (Xelem["typ"] == 7)

        Xelem = Xelem[~msk_ps]
        ygen = ygen[~msk_ps]
        ycand = ycand[~msk_ps]

        Xelem = append_fields(
            Xelem,
            "typ_idx",
            np.array(
                [ELEM_LABELS_CMS.index(int(i)) for i in Xelem["typ"]],
                dtype=np.float32,
            ),
        )
        ygen = append_fields(
            ygen,
            "typ_idx",
            np.array(
                [CLASS_LABELS_CMS.index(abs(int(i))) for i in ygen["typ"]],
                dtype=np.float32,
            ),
        )
        ycand = append_fields(
            ycand,
            "typ_idx",
            np.array(
                [CLASS_LABELS_CMS.index(abs(int(i))) for i in ycand["typ"]],
                dtype=np.float32,
            ),
        )

        Xelem_flat = np.stack(
            [Xelem[k].view(np.float32).data for k in X_FEATURES_CMS],
            axis=-1,
        )
        ygen_flat = np.stack(
            [ygen[k].view(np.float32).data for k in Y_FEATURES],
            axis=-1,
        )
        ycand_flat = np.stack(
            [ycand[k].view(np.float32).data for k in Y_FEATURES],
            axis=-1,
        )

        # take care of outliers
        Xelem_flat[np.isnan(Xelem_flat)] = 0
        Xelem_flat[np.abs(Xelem_flat) > 1e4] = 0

        ygen_flat[np.isnan(ygen_flat)] = 0
        ygen_flat[np.abs(ygen_flat) > 1e4] = 0

        ycand_flat[np.isnan(ycand_flat)] = 0
        ycand_flat[np.abs(ycand_flat) > 1e4] = 0

        d = Data(
            x=torch.tensor(Xelem_flat),
            ygen=torch.tensor(ygen_flat[:, 1:]),
            ygen_id=torch.tensor(ygen_flat[:, 0]).long(),
            ycand=torch.tensor(ycand_flat[:, 1:]),
            ycand_id=torch.tensor(ycand_flat[:, 0]).long(),
        )
        batched_data.append(d)

    return batched_data
