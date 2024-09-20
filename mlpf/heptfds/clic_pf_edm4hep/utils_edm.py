import random

import awkward as ak
import numpy as np

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
    # added by farouk
    "energyError",
    "sigma_energy",
    "sigma_x_weighted",
    "sigma_y_weighted",
    "sigma_z_weighted",
    "energy_weighted_width",
    "pos_shower_max",
    "width_shower_max",
    "energy_shower_max",
]

Y_FEATURES = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "ispu"]
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


def prepare_data_clic(fn):
    ret = ak.from_parquet(fn)
    X_track = ret["X_track"]
    X_cluster = ret["X_cluster"]

    assert len(X_track) == len(X_cluster)
    nev = len(X_track)

    Xs = []
    ygens = []
    ycands = []
    genmets = []
    genjets = []
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
        genmet = ak.to_numpy(ret["genmet"][iev])
        genjet = ak.to_numpy(ret["genjet"][iev])

        if len(ygen_track) == 0 and len(ygen_cluster) == 0:
            continue

        # in case the event had no track or cluster, create the right shapes
        if len(ygen_track) == 0:
            ygen_track = np.zeros((0, N_Y_FEATURES))
        if len(ygen_cluster) == 0:
            ygen_cluster = np.zeros((0, N_Y_FEATURES))
        if len(ycand_track) == 0:
            ycand_track = np.zeros((0, N_Y_FEATURES))
        if len(ycand_cluster) == 0:
            ycand_cluster = np.zeros((0, N_Y_FEATURES))

        # pad feature dim between tracks and clusters to the same size
        if X1.shape[1] < N_X_FEATURES:
            X1 = np.pad(X1, [[0, 0], [0, N_X_FEATURES - X1.shape[1]]])
        if X2.shape[1] < N_X_FEATURES:
            X2 = np.pad(X2, [[0, 0], [0, N_X_FEATURES - X2.shape[1]]])

        # concatenate tracks and clusters in features and targets
        X = np.concatenate([X1, X2])
        ygen = np.concatenate([ygen_track, ygen_cluster])
        ycand = np.concatenate([ycand_track, ycand_cluster])

        # this should not happen
        if (ygen.shape[0] != X.shape[0]) or (ycand.shape[0] != X.shape[0]):
            print("Shape mismatch:", X.shape, ygen.shape, ycand.shape)
            continue
            # raise Exception("Shape mismatch")

        # replace PID with index in labels array
        arr = np.array([labels.index(p) for p in ygen[:, 0]])
        ygen[:, 0][:] = arr[:]
        arr = np.array([labels.index(p) for p in ycand[:, 0]])
        ycand[:, 0][:] = arr[:]

        Xs.append(X)
        ygens.append(ygen)
        ycands.append(ycand)
        genmets.append(genmet)
        genjets.append(genjet)
    return Xs, ygens, ycands, genmets, genjets


def generate_examples(files):
    for fi in files:
        Xs, ygens, ycands, genmets, genjets = prepare_data_clic(fi)
        for iev in range(len(Xs)):
            gm = genmets[iev][0]
            gj = genjets[iev]
            yield str(fi) + "_" + str(iev), {
                "X": Xs[iev].astype(np.float32),
                "ygen": ygens[iev].astype(np.float32),
                "ycand": ycands[iev].astype(np.float32),
                "genmet": gm,
                "genjets": gj.astype(np.float32),
            }


if __name__ == "__main__":
    for ex in generate_examples(
        [
            "/local/joosep/mlpf/clic_edm4hep/pi+/reco_pi+_98.parquet",
            "/local/joosep/mlpf/clic_edm4hep/pi-/reco_pi-_11.parquet",
        ]
    ):
        print(ex[0], ex[1]["X"].shape, ex[1]["ygen"].shape, ex[1]["ycand"].shape)
