import numpy as np
from data_clic.postprocessing import prepare_data_clic

# these labels are for tracks from track_as_array
X_FEATURES_TRK = [
    "type",
    "px",
    "py",
    "pz",
    "nhits",
    "d0",
    "z0",
    "dedx",
    "radius_innermost_hit",
    "tan_lambda",
    "nhits",
    "chi2",
]

# these labels are for clusters from cluster_as_array
X_FEATURES_CL = ["type", "x", "y", "z", "nhits_ecal", "nhits_hcal", "energy"]

Y_FEATURES = ["type", "charge", "px", "py", "pz"]


def split_sample(path, test_frac=0.8):
    files = sorted(list(path.glob("*.parquet")))
    print("Found {} files in {}".format(files, path))
    assert len(files) > 0
    idx_split = int(test_frac * len(files))
    files_train = files[:idx_split]
    files_test = files[idx_split:]
    assert len(files_train) > 0
    assert len(files_test) > 0
    return {"train": generate_examples(files_train), "test": generate_examples(files_test)}


def generate_examples(files):
    for fi in files:
        ret = prepare_data_clic(fi)
        for iev, (X, ycand, ygen) in enumerate(ret):
            # print(X.shape, ycand.shape, ygen.shape)
            yield str(fi) + "_" + str(iev), {
                "X": X.astype(np.float32),
                "ygen": ygen.astype(np.float32),
                "ycand": ycand.astype(np.float32),
            }
