from configparser import MAX_INTERPOLATION_DEPTH
import pickle
import bz2

import numpy as np
from numpy.lib.recfunctions import append_fields
import fastjet
import vector
import awkward as ak


# https://github.com/ahlinist/cmssw/blob/1df62491f48ef964d198f574cdfcccfd17c70425/DataFormats/ParticleFlowReco/interface/PFBlockElement.h#L33
ELEM_LABELS_CMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ELEM_NAMES_CMS = ["NONE", "TRACK", "PS1", "PS2", "ECAL", "HCAL", "GSF", "BREM", "HFEM", "HFHAD", "SC", "HO"]

# https://github.com/cms-sw/cmssw/blob/master/DataFormats/ParticleFlowCandidate/src/PFCandidate.cc#L254
CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13]
CLASS_NAMES_CMS = ["none", "ch.had", "n.had", "HFHAD", "HFEM", "gamma", "ele", "mu"]
CLASS_NAMES_LONG_CMS = ["none" "charged hadron", "neutral hadron", "hfem", "hfhad", "photon", "electron", "muon"]

X_FEATURES = [
    "typ_idx", "pt", "eta", "phi", "e",
    "layer", "depth", "charge", "trajpoint", 
    "eta_ecal", "phi_ecal", "eta_hcal", "phi_hcal", "muon_dt_hits", "muon_csc_hits", "muon_type",
    "px", "py", "pz", "deltap", "sigmadeltap",
    "gsf_electronseed_trkorecal",
    "gsf_electronseed_dnn1",
    "gsf_electronseed_dnn2",
    "gsf_electronseed_dnn3",
    "gsf_electronseed_dnn4",
    "gsf_electronseed_dnn5",
    "num_hits", "cluster_flags", "corr_energy",
    "corr_energy_err", "vx", "vy", "vz", "pterror", "etaerror", "phierror", "lambd", "lambdaerror", "theta", "thetaerror"
]
               
Y_FEATURES = [
    "typ_idx",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "e",
    "jet_idx"
]

def prepare_data_cms(fn, padded_num_elem_size):
    Xs = []
    ygens = []
    ycands = []

    # prepare jet definition and min jet pt for clustering gen jets 
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    min_jet_pt = 5.0 # GeV

    if fn.endswith(".pkl"):
        data = pickle.load(open(fn, "rb"), encoding="iso-8859-1")
    elif fn.endswith(".pkl.bz2"):
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
            Xelem, "typ_idx", np.array([ELEM_LABELS_CMS.index(int(i)) for i in Xelem["typ"]], dtype=np.float32)
        )
        ygen = append_fields(
            ygen, "typ_idx", np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32)
        )
        ygen = append_fields(
            ygen, "jet_idx", np.zeros(ygen["typ"].shape, dtype=np.float32)
        )
        ycand = append_fields(
            ycand,
            "typ_idx",
            np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32),
        )
        ycand = append_fields(
            ycand, "jet_idx", np.zeros(ycand["typ"].shape, dtype=np.float32)
        )

        Xelem_flat = np.stack(
            [
                Xelem[k].view(np.float32).data
                for k in X_FEATURES
            ],
            axis=-1,
        )
        ygen_flat = np.stack(
            [
                ygen[k].view(np.float32).data
                for k in Y_FEATURES
            ],
            axis=-1,
        )
        ycand_flat = np.stack(
            [
                ycand[k].view(np.float32).data
                for k in Y_FEATURES
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

        X = Xelem_flat[:padded_num_elem_size]
        X = np.pad(X, [(0, padded_num_elem_size - X.shape[0]), (0, 0)])

        ygen = ygen_flat[:padded_num_elem_size]
        ygen = np.pad(ygen, [(0, padded_num_elem_size - ygen.shape[0]), (0, 0)])

        ycand = ycand_flat[:padded_num_elem_size]
        ycand = np.pad(ycand, [(0, padded_num_elem_size - ycand.shape[0]), (0, 0)])

        X = np.expand_dims(X, 0)
        ygen = np.expand_dims(ygen, 0)
        ycand = np.expand_dims(ycand, 0)

        # prepare gen candidates for clustering
        cls_id = ygen[:, :, 0]
        valid = (cls_id != 0)
        # save mapping of index after masking -> index before masking as numpy array
        # inspired from: https://stackoverflow.com/a/1044443
        cumsum = np.cumsum(valid) - 1
        index_mapping = np.nonzero(np.r_[1, np.diff(cumsum)[:-1]])[0]

        pt = ak.from_iter([y[m] for y, m in zip(ygen[:, :, Y_FEATURES.index("pt")], valid)])
        eta = ak.from_iter([y[m] for y, m in zip(ygen[:, :, Y_FEATURES.index("eta")], valid)])
        phi = np.arctan2(ygen[:, :, Y_FEATURES.index("sin_phi")], ygen[:, :, Y_FEATURES.index("cos_phi")])
        phi = ak.from_iter([y[m] for y, m in zip(phi, valid)])
        e = ak.from_iter([y[m] for y, m in zip(ygen[:, :, Y_FEATURES.index("e")], valid)])
        vec = vector.arr({"pt": pt, "eta": eta, "phi": phi, "e": e})

        # cluster jets, sort jet indices in descending order by pt
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets = cluster.inclusive_jets(min_pt=min_jet_pt)
        sorted_jet_idx = ak.argsort(jets.pt, axis=-1, ascending=False).to_list()[0]
        # retrieve corresponding indices of constituents
        constituent_idx = cluster.constituent_index(min_pt=min_jet_pt).to_list()[0]

        # add index information to ygen and ycand
        # index jets in descending order by pt starting from 1:
        # 0 is null (unclustered),
        # 1 is 1st highest-pt jet,
        # 2 is 2nd highest-pt jet, ...
        for jet_idx in sorted_jet_idx:
            jet_constituents = [index_mapping[idx] for idx in constituent_idx[jet_idx]] # map back to constituent index *before* masking
            ygen[0, jet_constituents, Y_FEATURES.index("jet_idx")] = jet_idx + 1 # jet index starts from 1
            ycand[0, jet_constituents, Y_FEATURES.index("jet_idx")] = jet_idx + 1

        Xs.append(X)
        ygens.append(ygen)
        ycands.append(ycand)

    X = [np.concatenate(Xs)]
    ygen = [np.concatenate(ygens)]
    ycand = [np.concatenate(ycands)]

    return X, ygen, ycand

def split_sample(path, pad_size, test_frac=0.8):
    files = sorted(list(path.glob("*.pkl*")))
    print("Found {} files in {}".format(files, path))
    assert(len(files)>0)
    idx_split = int(test_frac*len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert(len(files_train)>0)
    assert(len(files_test)>0)
    return {"train": generate_examples(files_train, pad_size), "test": generate_examples(files_test, pad_size)}

def generate_examples(files, pad_size):
    """Yields examples."""

    for fi in files:
        X, ygen, ycand = prepare_data_cms(str(fi), pad_size)
        for ii in range(X[0].shape[0]):
            x = X[0][ii]
            yg = ygen[0][ii]
            yc = ycand[0][ii]
            yield str(fi) + "_" + str(ii), {
                "X": x,
                "ygen": yg,
                "ycand": yc,
            }

