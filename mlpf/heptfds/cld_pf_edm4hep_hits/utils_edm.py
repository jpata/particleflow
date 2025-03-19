import awkward as ak
import numpy as np
import tqdm
import random

NUM_SPLITS = 10

# from fcc/postprocessing_hits.py
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
X_FEATURES_CH = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
]
X_FEAT_NUM = max(len(X_FEATURES_TRK), len(X_FEATURES_CH))

Y_FEATURES = ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"]
labels = [0, 211, 130, 22, 11, 13]


def split_list(lst, x):
    # Calculate the size of each sublist (except potentially the last)
    sublist_size = len(lst) // x

    # Create x-1 sublists of equal size
    result = [lst[i * sublist_size : (i + 1) * sublist_size] for i in range(x - 1)]

    # Add the remaining elements to the last sublist
    result.append(lst[(x - 1) * sublist_size :])

    return result


def split_sample(path, builder_config, num_splits=NUM_SPLITS, test_frac=0.9):
    files = sorted(list(path.glob("*.parquet")))
    print("Found {} files in {}".format(len(files), path))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split = split_list(files_test, num_splits)

    return {
        "train": generate_examples(files_train_split[split_index]),
        "test": generate_examples(files_test_split[split_index]),
    }


# merge and shuffle several samples (e.g. e+, e-), split into test/train
def split_sample_several(paths, builder_config, num_splits=NUM_SPLITS, test_frac=0.9):
    files = sum([list(path.glob("*.parquet")) for path in paths], [])
    random.shuffle(files)
    print("Found {} files".format(len(files)))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0

    split_index = int(builder_config.name) - 1
    files_train_split = split_list(files_train, num_splits)
    files_test_split = split_list(files_test, num_splits)

    return {
        "train": generate_examples(files_train_split[split_index]),
        "test": generate_examples(files_test_split[split_index]),
    }


def prepare_data_cld_hits(fn):
    ret = ak.from_parquet(fn)

    X_track = ret["X_track"]
    X_hit = ret["X_hit"]
    tracks_assoc_mats = ret["gp_to_track"]

    assert len(X_track) == len(X_hit)
    nev = len(X_track)

    Xs = []
    ytargets = []
    ycands = []
    gp_to_tracks = []
    gp_to_hits = []
    genmets = []
    genjets = []
    targetjets = []
    for iev in range(nev):

        X1 = ak.to_numpy(X_track[iev])
        X2 = ak.to_numpy(X_hit[iev])

        if len(X1) == 0 and len(X2) == 0:
            continue

        ytarget_track = ak.to_numpy(ret["ytarget_track"][iev])
        ytarget_hit = ak.to_numpy(ret["ytarget_hit"][iev])
        ycand_track = ak.to_numpy(ret["ycand_track"][iev])
        ycand_hit = ak.to_numpy(ret["ycand_hit"][iev])

        genmet = ak.to_numpy(ret["genmet"][iev])
        genjet = ak.to_numpy(ret["genjet"][iev])
        targetjet = ak.to_numpy(ret["targetjet"][iev])

        if tracks_assoc_mats is not None:
            gp_to_track = ak.to_numpy(ret["gp_to_track"][iev])
            gp_to_calohit = ak.to_numpy(ret["gp_to_calohit"][iev])

            gp_to_tracks.append(gp_to_track)
            gp_to_hits.append(gp_to_calohit)

        if ytarget_track.shape[0] == 0:
            ytarget_track = np.zeros((0, 7), dtype=np.float32)
        if ycand_track.shape[0] == 0:
            ycand_track = np.zeros((0, 7), dtype=np.float32)
        if ytarget_hit.shape[0] == 0:
            ytarget_hit = np.zeros((0, 7), dtype=np.float32)
        if ycand_hit.shape[0] == 0:
            ycand_hit = np.zeros((0, 7), dtype=np.float32)

        if len(ytarget_track) == 0 and len(ytarget_hit) == 0:
            continue
        if len(ycand_track) == 0 and len(ycand_hit) == 0:
            continue

        # pad feature dim between tracks and hits to the same size
        X1 = np.pad(X1, [[0, 0], [0, X_FEAT_NUM - X1.shape[1]]])
        X2 = np.pad(X2, [[0, 0], [0, X_FEAT_NUM - X2.shape[1]]])

        # concatenate tracks and hits in features and targets
        X = np.concatenate([X1, X2])
        ytarget = np.concatenate([ytarget_track, ytarget_hit])
        ycand = np.concatenate([ycand_track, ycand_hit])
        assert ytarget.shape[0] == X.shape[0]
        assert ycand.shape[0] == X.shape[0]

        # replace PID with index in labels array
        arr = np.array([labels.index(p) for p in ytarget[:, 0]])
        ytarget[:, 0][:] = arr[:]
        arr = np.array([labels.index(p) for p in ycand[:, 0]])
        ycand[:, 0][:] = arr[:]
        Xs.append(X)
        ytargets.append(ytarget)
        ycands.append(ycand)
        genmets.append(genmet)
        genjets.append(genjet)
        targetjets.append(targetjet)

    return Xs, ytargets, ycands, genmets, genjets, targetjets, gp_to_tracks, gp_to_hits


def generate_examples(files):
    for fi in tqdm.tqdm(files):
        Xs, ytargets, ycands, genmets, genjets, targetjets, gp_to_tracks, gp_to_hits = prepare_data_cld_hits(fi)
        for iev in range(len(Xs)):
            gm = genmets[iev][0]
            gj = genjets[iev]
            tj = targetjets[iev]
            if gp_to_tracks == []:
                yield str(fi) + "_" + str(iev), {
                    "X": Xs[iev].astype(np.float32),
                    "ytarget": ytargets[iev].astype(np.float32),
                    "ycand": ycands[iev].astype(np.float32),
                    "genmet": gm,
                    "genjets": gj.astype(np.float32),
                    "targetjets": tj.astype(np.float32),
                }
            else:
                yield str(fi) + "_" + str(iev), {
                    "X": Xs[iev].astype(np.float32),
                    "ytarget": ytargets[iev].astype(np.float32),
                    "ycand": ycands[iev].astype(np.float32),
                    "genmet": gm,
                    "genjets": gj.astype(np.float32),
                    "targetjets": tj.astype(np.float32),
                    "gp_to_tracks": gp_to_tracks[iev].astype(np.float32),
                    "gp_to_hits": gp_to_hits[iev].astype(np.float32),
                }


if __name__ == "__main__":
    fn = "/local/joosep/mlpf_hits/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_111398.parquet"
    ret = prepare_data_cld_hits(fn)
