import fastjet
import numpy as np
import pickle
import bz2
import vector
import awkward as ak

DELPHES_CLASS_NAMES = [
    "none",
    "charged hadron",
    "neutral hadron",
    "hfem",
    "hfhad",
    "photon",
    "electron",
    "muon",
]


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

Y_FEATURES = [
    "type",
    "charge",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "jet_idx",
]


def prepare_data_delphes(fname, with_jet_idx=True):

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    min_jet_pt = 5.0  # GeV

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

        # add jet_idx column
        if with_jet_idx:
            ygen = np.concatenate(
                [
                    ygen.astype(np.float32),
                    np.zeros((len(ygen), 1), dtype=np.float32),
                ],
                axis=-1,
            )
            ycand = np.concatenate(
                [
                    ycand.astype(np.float32),
                    np.zeros((len(ycand), 1), dtype=np.float32),
                ],
                axis=-1,
            )

        # in the delphes sample, neutral PF candidates have only E defined, and charged PF candidates have only pT defined
        # fix this up here for the delphes PF candidates
        pz = ycand[:, Y_FEATURES.index("energy")] * np.cos(2 * np.arctan(np.exp(-ycand[:, Y_FEATURES.index("eta")])))
        pt = np.sqrt(ycand[:, Y_FEATURES.index("energy")] ** 2 - pz**2)

        # eta=atanh(pz/p) => E=pt/sqrt(1-tanh(eta))
        e = ycand[:, Y_FEATURES.index("pt")] / np.sqrt(1.0 - np.tanh(ycand[:, Y_FEATURES.index("eta")]))

        # use these computed values where they are missing
        msk_neutral = np.abs(ycand[:, Y_FEATURES.index("charge")]) == 0
        msk_charged = ~msk_neutral
        ycand[:, Y_FEATURES.index("pt")] = msk_charged * ycand[:, Y_FEATURES.index("pt")] + msk_neutral * pt
        ycand[:, Y_FEATURES.index("energy")] = msk_neutral * ycand[:, Y_FEATURES.index("energy")] + msk_charged * e

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
            e = ygen[valid, Y_FEATURES.index("energy")]
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
    files = sorted(list(path.glob("*.pkl.bz2")))
    print("Found {} files in {}".format(len(files), path))
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
    for fi in files:
        Xs, ygens, ycands = prepare_data_delphes(str(fi))
        assert len(Xs) > 0
        for iev in range(len(Xs)):
            yield str(fi) + "_" + str(iev), {
                "X": Xs[iev],
                "ygen": ygens[iev],
                "ycand": ycands[iev],
            }
