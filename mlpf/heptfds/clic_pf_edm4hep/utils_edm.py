import awkward as ak
import fastjet
import numpy as np
import vector
import random

jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
min_jet_pt = 5.0  # GeV

# from fcc/postprocessing.py
X_FEATURES_TRK = [
    "elemtype",
    "pt",
    "eta",
    "sin_phi",
    "cos_phi",
    "p",
    "chi2",
    "ndf",
    "dEdx",
    "dEdxError",
    "radiusOfInnermostHit",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
]
X_FEATURES_CL = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "iTheta",
    "energy_ecal",
    "energy_hcal",
    "energy_other",
    "num_hits",
    "sigma_x",
    "sigma_y",
    "sigma_z",
]

Y_FEATURES = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"]
labels = [0, 211, 130, 22, 11, 13]

N_X_FEATURES = max(len(X_FEATURES_CL), len(X_FEATURES_TRK))
N_Y_FEATURES = len(Y_FEATURES)


def split_sample(path, test_frac=0.8):
    files = sorted(list(path.glob("*.parquet")))
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


def split_sample_several(paths, test_frac=0.8):
    files = sum([list(path.glob("*.parquet")) for path in paths], [])
    random.shuffle(files)
    print("Found {} files".format(len(files)))
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


def prepare_data_clic(fn, with_jet_idx=True):
    ret = ak.from_parquet(fn)
    X_track = ret["X_track"]
    X_cluster = ret["X_cluster"]

    assert len(X_track) == len(X_cluster)
    nev = len(X_track)

    Xs = []
    ygens = []
    ycands = []
    for iev in range(nev):

        X1 = ak.to_numpy(X_track[iev])
        X2 = ak.to_numpy(X_cluster[iev])

        if len(X1) == 0 and len(X2) == 0:
            continue

        if len(X1) == 0:
            X1 = np.zeros((0, N_X_FEATURES))
        if len(X2) == 0:
            X2 = np.zeros((0, N_X_FEATURES))

        ygen_track = ak.to_numpy(ret["ygen_track"][iev])
        ygen_cluster = ak.to_numpy(ret["ygen_cluster"][iev])
        ycand_track = ak.to_numpy(ret["ycand_track"][iev])
        ycand_cluster = ak.to_numpy(ret["ycand_cluster"][iev])

        if len(ygen_track) == 0 and len(ygen_cluster) == 0:
            continue

        if len(ygen_track) == 0:
            ygen_track = np.zeros((0, N_Y_FEATURES - 1))
        if len(ygen_cluster) == 0:
            ygen_cluster = np.zeros((0, N_Y_FEATURES - 1))
        if len(ycand_track) == 0:
            ycand_track = np.zeros((0, N_Y_FEATURES - 1))
        if len(ycand_cluster) == 0:
            ycand_cluster = np.zeros((0, N_Y_FEATURES - 1))

        # pad feature dim between tracks and clusters to the same size
        if X1.shape[1] < N_X_FEATURES:
            X1 = np.pad(X1, [[0, 0], [0, N_X_FEATURES - X1.shape[1]]])
        if X2.shape[1] < N_X_FEATURES:
            X2 = np.pad(X2, [[0, 0], [0, N_X_FEATURES - X2.shape[1]]])

        # concatenate tracks and clusters in features and targets
        X = np.concatenate([X1, X2])
        ygen = np.concatenate([ygen_track, ygen_cluster])
        ycand = np.concatenate([ycand_track, ycand_cluster])

        if (ygen.shape[0] != X.shape[0]) or (ycand.shape[0] != X.shape[0]):
            print(X.shape, ygen.shape, ycand.shape)
            continue

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

        # replace PID with index in labels array
        arr = np.array([labels.index(p) for p in ygen[:, 0]])
        ygen[:, 0][:] = arr[:]
        arr = np.array([labels.index(p) for p in ycand[:, 0]])
        ycand[:, 0][:] = arr[:]

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
            sin_phi = ygen[valid, Y_FEATURES.index("sin_phi")]
            cos_phi = ygen[valid, Y_FEATURES.index("cos_phi")]
            phi = np.arctan2(sin_phi, cos_phi)
            energy = ygen[valid, Y_FEATURES.index("energy")]
            vec = vector.awk(ak.zip({"pt": pt, "eta": eta, "phi": phi, "energy": energy}))

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


def generate_examples(files, with_jet_idx=True):
    for fi in files:
        print(fi)
        Xs, ygens, ycands = prepare_data_clic(fi, with_jet_idx=with_jet_idx)
        for iev in range(len(Xs)):
            yield str(fi) + "_" + str(iev), {
                "X": Xs[iev].astype(np.float32),
                "ygen": ygens[iev],
                "ycand": ycands[iev],
            }


if __name__ == "__main__":
    for ex in generate_examples(
        [
            "/local/joosep/mlpf/clic_edm4hep/pi+/reco_pi+_98.parquet",
            "/local/joosep/mlpf/clic_edm4hep/pi-/reco_pi-_11.parquet",
        ]
    ):
        print(ex[0], ex[1]["X"].shape, ex[1]["ygen"].shape, ex[1]["ycand"].shape)
