import bz2
import pickle
import tqdm

import awkward as ak
import fastjet
import numpy as np
import vector

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
    "none" "charged hadron",
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
    "e",
    "jet_idx",
]


def prepare_data_cms(fn, with_jet_idx=True):
    Xs = []
    ygens = []
    ycands = []

    # prepare jet definition and min jet pt for clustering gen jets
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    min_jet_pt = 5.0  # GeV

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

        Xelem = ak.Array(Xelem[~msk_ps])
        ygen = ak.Array(ygen[~msk_ps])
        ycand = ak.Array(ycand[~msk_ps])

        Xelem["sin_phi"] = np.sin(Xelem["phi"])
        Xelem["cos_phi"] = np.cos(Xelem["phi"])
        Xelem["typ_idx"] = np.array([ELEM_LABELS_CMS.index(int(i)) for i in Xelem["typ"]], dtype=np.float32)
        ygen["typ_idx"] = np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32)
        ycand["typ_idx"] = np.array([CLASS_LABELS_CMS.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32)

        if with_jet_idx:
            ygen["jet_idx"] = np.zeros(len(ygen["typ"]), dtype=np.float32)
            ycand["jet_idx"] = np.zeros(len(ycand["typ"]), dtype=np.float32)

        Xelem_flat = ak.to_numpy(
            np.stack(
                [Xelem[k] for k in X_FEATURES],
                axis=-1,
            )
        )
        ygen_flat = ak.to_numpy(
            np.stack(
                [ygen[k] for k in Y_FEATURES],
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
        ygen = ygen_flat

        if with_jet_idx:
            # prepare gen candidates for clustering
            cls_id = ygen[..., 0]
            valid = cls_id != 0
            # save mapping of index after masking -> index before masking as numpy array
            # inspired from:
            # https://stackoverflow.com/questions/432112/1044443#comment54747416_1044443
            cumsum = np.cumsum(valid) - 1
            _, index_mapping = np.unique(cumsum, return_index=True)

            pt = ygen[valid, Y_FEATURES.index("pt")]
            eta = ygen[valid, Y_FEATURES.index("eta")]
            phi = np.arctan2(
                ygen[valid, Y_FEATURES.index("sin_phi")],
                ygen[valid, Y_FEATURES.index("cos_phi")],
            )
            e = ygen[valid, Y_FEATURES.index("e")]
            vec = vector.awk(ak.zip({"pt": pt, "eta": eta, "phi": phi, "e": e}))

            # cluster jets, sort jet indices in descending order by pt
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
            jets = vector.awk(cluster.inclusive_jets(min_pt=min_jet_pt))
            sorted_jet_idx = ak.argsort(jets.pt, axis=-1, ascending=False).to_list()
            # retrieve corresponding indices of constituents
            constituent_idx = cluster.constituent_index(min_pt=min_jet_pt).to_list()

            # add index information to ygen and ycand
            # index jets in descending order by pt starting from 1:
            # 0 is null (unclustered),
            # 1 is 1st highest-pt jet,
            # 2 is 2nd highest-pt jet, ...
            for jet_idx in sorted_jet_idx:
                jet_constituents = [
                    index_mapping[idx] for idx in constituent_idx[jet_idx]
                ]  # map back to constituent index *before* masking
                ygen[jet_constituents, Y_FEATURES.index("jet_idx")] = jet_idx + 1  # jet index starts from 1
                ycand[jet_constituents, Y_FEATURES.index("jet_idx")] = jet_idx + 1

        Xs.append(X)
        ygens.append(ygen)
        ycands.append(ycand)

    return Xs, ygens, ycands


def split_sample(path, test_frac=0.8):
    files = sorted(list(path.glob("*.pkl*")))
    print("Found {} files in {}".format(files, path))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0
    return {
        "train": generate_examples(files_train),
        "test": generate_examples(files_test),
    }


def generate_examples(files):
    """Yields examples."""

    for fi in tqdm.tqdm(files):
        Xs, ygens, ycands = prepare_data_cms(str(fi))
        for ii in range(len(Xs)):
            x = Xs[ii]
            yg = ygens[ii]
            yc = ycands[ii]

            uniqs, counts = np.unique(yg[:, 0], return_counts=True)
            yield str(fi) + "_" + str(ii), {
                "X": x,
                "ygen": yg,
                "ycand": yc,
            }
