import bz2
import pickle
import datetime

import awkward as ak
import numpy as np

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

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
CLASS_LABELS_CMS = [0, 211, 130, 1, 2, 22, 11, 13]
CLASS_NAMES_CMS = [
    "none",
    "ch.had",
    "n.had",
    "HFHAD",
    "HFEM",
    "gamma",
    "ele",
    "mu",
]
CLASS_NAMES_LONG_CMS = [
    "none",
    "charged hadron",
    "neutral hadron",
    "hfem",
    "hfhad",
    "photon",
    "electron",
    "muon",
]

X_FEATURES = [
    "typ_idx",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
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
    "time",
    "timeerror",
    "etaerror1",
    "etaerror2",
    "etaerror3",
    "etaerror4",
    "phierror1",
    "phierror2",
    "phierror3",
    "phierror4",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]

Y_FEATURES = [
    "typ_idx",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "ispu",
    "generatorStatus",
    "simulatorStatus",
    "cp_to_track",
    "cp_to_cluster",
    "jet_idx",
]

# split each dataset into equal parts for faster building
NUM_SPLITS = 10


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    if pdgid in [22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def prepare_data_cms(fn):
    Xs = []
    ytargets = []
    ycands = []
    genmets = []
    genjets = []
    targetjets = []
    ypythias = []

    try:
        if fn.endswith(".pkl"):
            data = pickle.load(open(fn, "rb"), encoding="iso-8859-1")
        elif fn.endswith(".pkl.bz2"):
            data = pickle.load(bz2.BZ2File(fn, "rb"))
    except Exception as e:
        print("Could not open file {}: {}".format(fn, e))
        return Xs, ytargets, ycands, genmets, genjets, targetjets

    for event in data:
        Xelem = event["Xelem"]
        ytarget = event["ytarget"]
        ycand = event["ycand"]
        genmet = event["genmet"][0][0]
        genjet = event["genjet"]
        targetjet = event["targetjet"]
        ypythia = event["pythia"]

        # remove PS and BREM from inputs
        msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3) | (Xelem["typ"] == 7)

        Xelem = ak.Array(Xelem[~msk_ps])
        ytarget = ak.Array(ytarget[~msk_ps])
        ycand = ak.Array(ycand[~msk_ps])

        Xelem["sin_phi"] = np.sin(Xelem["phi"])
        Xelem["cos_phi"] = np.cos(Xelem["phi"])
        Xelem["typ_idx"] = np.array([ELEM_LABELS_CMS.index(int(i)) for i in Xelem["typ"]], dtype=np.float32)
        pids_remapped = [map_pdgid_to_candid(abs(int(pid)), q) for (pid, q) in zip(ytarget["pid"], ytarget["charge"])]
        ytarget["typ_idx"] = np.array([CLASS_LABELS_CMS.index(pid) for pid in pids_remapped], dtype=np.float32)
        ycand["typ_idx"] = np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ycand["pid"]], dtype=np.float32)

        Xelem_flat = ak.to_numpy(
            np.stack(
                [Xelem[k] for k in X_FEATURES],
                axis=-1,
            )
        )
        ytarget_flat = ak.to_numpy(
            np.stack(
                [ytarget[k] for k in Y_FEATURES],
                axis=-1,
            )
        )
        ycand_flat = ak.to_numpy(
            np.stack(
                [ycand[k] for k in Y_FEATURES],
                axis=-1,
            )
        )

        X = Xelem_flat
        ycand = ycand_flat
        ytarget = ytarget_flat

        Xs.append(X)
        ytargets.append(ytarget)
        ycands.append(ycand)
        genmets.append(genmet)
        genjets.append(genjet)
        targetjets.append(targetjet)
        ypythias.append(ypythia)

    return Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias


def split_list(lst, x):
    # Calculate the size of each sublist (except potentially the last)
    sublist_size = len(lst) // x

    # Create x-1 sublists of equal size
    result = [lst[i * sublist_size : (i + 1) * sublist_size] for i in range(x - 1)]

    # Add the remaining elements to the last sublist
    result.append(lst[(x - 1) * sublist_size :])

    return result


def split_sample(path, builder_config, num_splits=NUM_SPLITS, train_frac=0.9):
    files = sorted(list(path.glob("*.pkl*")))
    print("Found {} files in {}".format(len(files), path))
    assert len(files) > 0
    idx_test = int(train_frac * len(files))
    files_train = files[:idx_test]
    files_test = files[idx_test:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split = split_list(files_test, num_splits)
    assert len(files_train_split[split_index]) > 0
    assert len(files_test_split[split_index]) > 0
    return {
        "train": generate_examples(files_train_split[split_index]),
        "test": generate_examples(files_test_split[split_index]),
    }


def generate_examples(files):
    """Yields examples."""

    for fi in files:
        print(datetime.datetime.now(), "started reading file", fi)
        Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias = prepare_data_cms(str(fi))
        if len(Xs) == 0:
            print("Error: file {} is broken".format(fi))
        for ii in range(len(Xs)):
            x = Xs[ii].astype(np.float32)
            yg = ytargets[ii].astype(np.float32)
            yc = ycands[ii].astype(np.float32)
            gm = genmets[ii].astype(np.float32)
            gj = genjets[ii].astype(np.float32)
            tj = targetjets[ii].astype(np.float32)
            yp = ypythias[ii].astype(np.float32)

            uniqs, counts = np.unique(yg[:, 0], return_counts=True)
            yield str(fi) + "_" + str(ii), {"X": x, "ytarget": yg, "ycand": yc, "genmet": gm, "genjets": gj, "targetjets": tj, "pythia": yp}
