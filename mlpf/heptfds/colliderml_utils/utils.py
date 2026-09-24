# Shared ColliderML -> TFDS utilities. Mirrors mlpf/heptfds/edm4hep_utils/utils_pf.py but reads
# the MLPF-format parquet produced by mlpf/data/colliderml/postprocessing.py (a dict-like record
# of X_track / X_cluster / ytarget_track / ytarget_cluster / genmet / genjet / targetjet /
# event_id).
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np

from mlpf.conf import ParticleFeatures

NUM_SPLITS = 10
Y_FEATURES = ParticleFeatures.get_names()
labels = [0, 211, 130, 22, 11, 13]

# X feature widths: tracks and clusters are written as the 16-wide layout by postprocessing;
# we pad/truncate to the registered X_FEATURES[Dataset.COLLIDERML] width (currently 17),
# which matches the EDM4hep joint-layout convention (union of track/cluster feature orders).
N_X_FEATURES = 17
N_Y_FEATURES = len(Y_FEATURES)


def split_list(lst, x):
    sublist_size = len(lst) // x
    result = [lst[i * sublist_size : (i + 1) * sublist_size] for i in range(x - 1)]
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


def _empty_y(n, w):
    return np.zeros((0, w), dtype=np.float32) if n == 0 else None


def _normalize_matrix(mat, width):
    """Convert an ak/numpy variable-length 2D matrix representation to a padded float32
    numpy array of shape (n, width). Accepts (n, m) ndarrays and slices the columns to the
    desired width (or pads with zeros to reach it)."""
    # inputs are already numpy/awkward-converted; this helper just normalizes the second axis
    m = np.asarray(mat)
    if m.ndim == 1 and m.size == 0:
        return np.zeros((0, width), dtype=np.float32)
    if m.shape[1] < width:
        return np.pad(m, ((0, 0), (0, width - m.shape[1])), mode="constant")
    return m[:, :width]


def prepare_events(fn: Path):
    ret = ak.from_parquet(fn)
    nev = len(ret["event_id"])
    Xs = []
    ytargets = []
    ycands = []
    genmets = []
    genjets = []
    targetjets = []
    for iev in range(nev):
        X_track = ak.to_numpy(ret["X_track"][iev])
        X_cluster = ak.to_numpy(ret["X_cluster"][iev])
        y_track = ak.to_numpy(ret["ytarget_track"][iev])
        y_cluster = ak.to_numpy(ret["ytarget_cluster"][iev])

        if (X_track.shape[0] == 0 and X_cluster.shape[0] == 0) or (y_track.shape[0] == 0 and y_cluster.shape[0] == 0):
            continue

        X_track = _normalize_matrix(X_track, N_X_FEATURES)
        X_cluster = _normalize_matrix(X_cluster, N_X_FEATURES)
        y_track = _normalize_matrix(y_track, N_Y_FEATURES)
        y_cluster = _normalize_matrix(y_cluster, N_Y_FEATURES)
        if y_track.shape[0] == 0:
            y_track = _empty_y(0, N_Y_FEATURES)
        if y_cluster.shape[0] == 0:
            y_cluster = _empty_y(0, N_Y_FEATURES)

        # option-typed parquet scalars come back from ak.to_numpy as shape-(1,) arrays; reshape
        # keeps this correct for both 0-d and option-typed readback
        genmet = float(np.asarray(ak.to_numpy(ret["genmet"][iev])).reshape(-1)[0])
        genjet = ak.to_numpy(ret["genjet"][iev])
        targetjet = ak.to_numpy(ret["targetjet"][iev])
        if genjet.size == 0:
            genjet = np.zeros((0, 4), dtype=np.float32)
        if targetjet.size == 0:
            targetjet = np.zeros((0, 4), dtype=np.float32)

        X = np.concatenate([X_track, X_cluster], axis=0)
        ytarget = np.concatenate([y_track, y_cluster], axis=0)
        ycand = np.zeros(ytarget.shape, dtype=np.float32)

        # class-label index remap; matches existing heptfds convention
        arr = np.array([labels.index(int(p)) if int(p) in labels else 0 for p in ytarget[:, 0]], dtype=np.float32)
        ytarget[:, 0] = arr

        Xs.append(X)
        ytargets.append(ytarget)
        ycands.append(ycand)
        genmets.append(genmet)
        genjets.append(genjet)
        targetjets.append(targetjet)
    return Xs, ytargets, ycands, genmets, genjets, targetjets


def generate_examples(files: List[Path]):
    for fi in files:
        Xs, ytargets, ycands, genmets, genjets, targetjets = prepare_events(fi)
        for iev in range(len(Xs)):
            yield (
                str(fi) + "_" + str(iev),
                {
                    "X": Xs[iev].astype(np.float32),
                    "ytarget": ytargets[iev].astype(np.float32),
                    "ycand": ycands[iev].astype(np.float32),
                    "genmet": genmets[iev],
                    "genjets": genjets[iev].astype(np.float32),
                    "targetjets": targetjets[iev].astype(np.float32),
                },
            )
