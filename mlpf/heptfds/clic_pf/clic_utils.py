import numpy as np
import vector
import awkward as ak
import fastjet
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

Y_FEATURES = ["type", "charge", "px", "py", "pz", "energy", "jet_idx"]


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
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    min_jet_pt = 5.0  # GeV
    for fi in files:
        ret = prepare_data_clic(fi)

        for iev, (X, ycand, ygen) in enumerate(ret):

            # add jet_idx column
            ygen = np.concatenate([ygen.astype(np.float32), np.zeros((len(ygen), 1), dtype=np.float32)], axis=-1)
            ycand = np.concatenate([ycand.astype(np.float32), np.zeros((len(ycand), 1), dtype=np.float32)], axis=-1)

            # prepare gen candidates for clustering
            cls_id = ygen[..., 0]
            valid = cls_id != 0
            # save mapping of index after masking -> index before masking as numpy array
            # inspired from:
            # https://stackoverflow.com/questions/432112/1044443#comment54747416_1044443
            cumsum = np.cumsum(valid) - 1
            _, index_mapping = np.unique(cumsum, return_index=True)

            px = ygen[valid, Y_FEATURES.index("px")]
            py = ygen[valid, Y_FEATURES.index("py")]
            pz = ygen[valid, Y_FEATURES.index("pz")]
            e = ygen[valid, Y_FEATURES.index("energy")]
            vec = vector.arr(ak.zip({"px": px, "py": py, "pz": pz, "energy": e}))

            # cluster jets, sort jet indices in descending order by pt
            cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
            jets = vector.arr(cluster.inclusive_jets(min_pt=min_jet_pt))
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

            yield str(fi) + "_" + str(iev), {
                "X": X.astype(np.float32),
                "ygen": ygen,
                "ycand": ycand,
            }
