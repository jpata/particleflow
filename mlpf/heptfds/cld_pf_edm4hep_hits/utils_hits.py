import awkward as ak
import numpy as np

# workaround for 'ModuleNotFoundError: No module named importlib_resources'
try:
    import importlib_resources  # noqa
except Exception:
    import sys
    import importlib.resources

    sys.modules["importlib_resources"] = importlib.resources

from mlpf.conf import CLASS_LABELS, Dataset, EDM4HEP, ParticleFeatures

NUM_SPLITS = 10

X_FEATURES = EDM4HEP.HitFeatures.get_names()
Y_FEATURES = ParticleFeatures.get_names()

labels = CLASS_LABELS[Dataset.CLD_HITS.value]

N_X_FEATURES = len(X_FEATURES)
N_Y_FEATURES = len(Y_FEATURES)


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
    assert len(files_train_split[split_index]) > 0
    assert len(files_test_split[split_index]) > 0

    return {
        "train": generate_examples(files_train_split[split_index]),
        "test": generate_examples(files_test_split[split_index]),
    }


def prepare_data_hits(fn):
    ret = ak.from_parquet(fn)
    # X_hit_tracker is named X_track to match the structure of the track/cluster-based dataset
    X_track = ret["X_hit_tracker"]
    # X_hit_calo is named X_cluster to match the structure of the track/cluster-based dataset
    X_cluster = ret["X_hit_calo"]

    nev = len(X_track)

    Xs = []
    ytargets = []
    ycands = []
    genmets = []
    genjets = []
    targetjets = []
    for iev in range(nev):

        X1 = ak.to_numpy(X_track[iev])
        X2 = ak.to_numpy(X_cluster[iev])

        if len(X1) == 0 and len(X2) == 0:
            continue

        if len(X1) == 0:
            X1 = np.zeros((0, N_X_FEATURES))
        if len(X2) == 0:
            X2 = np.zeros((0, N_X_FEATURES))

        # concatenate tracker hits and calorimeter hits
        X = np.concatenate([X1, X2])

        # ytarget_hit_tracker is named ytarget_track to match the structure of the track/cluster-based dataset
        ytarget_track = ak.to_numpy(ret["ytarget_hit_tracker"][iev])
        # ytarget_hit_calo is named ytarget_cluster to match the structure of the track/cluster-based dataset
        ytarget_cluster = ak.to_numpy(ret["ytarget_hit_calo"][iev])

        if len(ytarget_track) == 0:
            ytarget_track = np.zeros((0, N_Y_FEATURES))
        if len(ytarget_cluster) == 0:
            ytarget_cluster = np.zeros((0, N_Y_FEATURES))

        # concatenate tracker hit targets and calorimeter hit targets
        ytarget = np.concatenate([ytarget_track, ytarget_cluster])

        # Hits don't have a baseline ycand (reco particle), so we just use zeros
        ycand = np.zeros_like(ytarget)

        genmet = ak.to_numpy(ret["genmet"][iev])
        genjet = ak.to_numpy(ret["genjet"][iev])
        targetjet = ak.to_numpy(ret["targetjet"][iev])

        if len(ytarget) == 0:
            continue

        if len(genjet) == 0:
            genjet = np.zeros((0, 4), dtype=np.float32)
        if len(targetjet) == 0:
            targetjet = np.zeros((0, 4), dtype=np.float32)

        # replace PID with index in labels array
        arr = np.array([labels.index(p) for p in ytarget[:, 0]])
        ytarget[:, 0][:] = arr[:]

        Xs.append(X)
        ytargets.append(ytarget)
        ycands.append(ycand)
        genmets.append(genmet)
        genjets.append(genjet)
        targetjets.append(targetjet)
    return Xs, ytargets, ycands, genmets, genjets, targetjets


def generate_examples(files):
    for fi in files:
        Xs, ytargets, ycands, genmets, genjets, targetjets = prepare_data_hits(fi)
        print(fi, [len(x) for x in Xs])
        for iev in range(len(Xs)):
            gm = genmets[iev][0]
            gj = genjets[iev]
            tj = targetjets[iev]
            yield str(fi) + "_" + str(iev), {
                "X": Xs[iev].astype(np.float32),
                "ytarget": ytargets[iev].astype(np.float32),
                "ycand": ycands[iev].astype(np.float32),
                "genmet": gm,
                "genjets": gj.astype(np.float32),
                "targetjets": tj.astype(np.float32),
            }
