# import bz2
import multiprocessing
import os.path as osp
import pickle
import sys
from glob import glob

import awkward as ak
import numpy as np
import torch
import tqdm

# from numpy.lib.recfunctions import append_fields
from torch_geometric.data import Data, Dataset

sys.path
sys.path.append("../")
from heptfds.cms_pf.cms_utils import prepare_data_cms

# def prepare_data_cms(fn):
#     """
#     Takes as input a bz2 file that contains the cms raw information, and returns a list of PyG Data() objects.
#     Each element of the list looks like this ~ Data(x=[#, 41], ygen=[#, 6], ygen_id=[#, 9], ycand=[#, 6], ycand_id=[#, 9])

#     Args
#         raw_file_name: raw parquet data file.
#     Returns
#         list of Data() objects.
#     """
#     from utils import CLASS_LABELS, X_FEATURES

#     # ELEM_NAMES = ["NONE", "TRACK", "PS1", "PS2", "ECAL", "HCAL", "GSF", "BREM", "HFEM", "HFHAD", "SC", "HO"]
#     ELEM_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

#     batched_data = []

#     data = pickle.load(bz2.BZ2File(fn, "rb"))
#     for event in data:
#         Xelem = event["Xelem"]
#         ygen = event["ygen"]
#         ycand = event["ycand"]

#         # remove PS and BREM from inputs
#         msk_ps = (Xelem["typ"] == 2) | (Xelem["typ"] == 3) | (Xelem["typ"] == 7)

#         Xelem = Xelem[~msk_ps]
#         ygen = ygen[~msk_ps]
#         ycand = ycand[~msk_ps]

#         Xelem = append_fields(
#             Xelem,
#             "typ_idx",
#             np.array(
#                 [ELEM_LABELS.index(int(i)) for i in Xelem["typ"]],
#                 dtype=np.float32,
#             ),
#         )
#         ygen = append_fields(
#             ygen,
#             "typ_idx",
#             np.array(
#                 [CLASS_LABELS["CMS"].index(abs(int(i))) for i in ygen["typ"]],
#                 dtype=np.float32,
#             ),
#         )
#         ycand = append_fields(
#             ycand,
#             "typ_idx",
#             np.array(
#                 [CLASS_LABELS["CMS"].index(abs(int(i))) for i in ycand["typ"]],
#                 dtype=np.float32,
#             ),
#         )

#         Xelem_flat = np.stack(
#             [Xelem[k].view(np.float32).data for k in X_FEATURES["CMS"]],
#             axis=-1,
#         )
#         ygen_flat = np.stack(
#             [ygen[k].view(np.float32).data for k in ["typ", "charge", "pt", "eta", "sin_phi", "cos_phi", "e"]],
#             axis=-1,
#         )
#         ycand_flat = np.stack(
#             [ycand[k].view(np.float32).data for k in ["typ", "charge", "pt", "eta", "sin_phi", "cos_phi", "e"]],
#             axis=-1,
#         )

#         # take care of outliers
#         Xelem_flat[np.isnan(Xelem_flat)] = 0
#         Xelem_flat[np.abs(Xelem_flat) > 1e4] = 0

#         ygen_flat[np.isnan(ygen_flat)] = 0
#         ygen_flat[np.abs(ygen_flat) > 1e4] = 0

#         ycand_flat[np.isnan(ycand_flat)] = 0
#         ycand_flat[np.abs(ycand_flat) > 1e4] = 0

#         d = Data(
#             x=torch.tensor(Xelem_flat),
#             ygen=torch.tensor(ygen_flat[:, 1:]),
#             ygen_id=torch.tensor(ygen_flat[:, 0]).long(),
#             ycand=torch.tensor(ycand_flat[:, 1:]),
#             ycand_id=torch.tensor(ycand_flat[:, 0]).long(),
#         )
#         batched_data.append(d)
#     return batched_data


def prepare_data_delphes(fn):
    """
    Takes as input a pkl file that contains the delphes raw information, and returns a list of PyG Data() objects.
    Each element of the list looks like this ~ Data(x=[#, 12], ygen=[#, 6], ygen_id=[#, 6], ycand=[#, 6], ycand_id=[#, 6])

    Args
        raw_file_name: raw parquet data file.
    Returns
        list of Data() objects.
    """

    with open(fn, "rb") as fi:
        data = pickle.load(fi, encoding="iso-8859-1")

    batched_data = []
    for i in range(len(data["X"])):
        # remove from ygen & ycand the first element (PID) so that they only contain the regression variables
        d = Data(
            x=torch.tensor(data["X"][i], dtype=torch.float),
            ygen=torch.tensor(data["ygen"][i], dtype=torch.float)[:, 1:],
            ygen_id=torch.tensor(data["ygen"][i], dtype=torch.float)[:, 0].long(),
            ycand=torch.tensor(data["ycand"][i], dtype=torch.float)[:, 1:],
            ycand_id=torch.tensor(data["ycand"][i], dtype=torch.float)[:, 0].long(),
        )

        batched_data.append(d)
    return batched_data


def prepare_data_clic(fn):
    def generate_examples(files):
        """
        Function that reads the CLIC data information from the parquet files.

        Args
            list of files

        Returns
            a generator which yields [{filename}_{index}, X, ygen, ycand]
        """

        labels = [0, 211, 130, 22, 11, 13]
        for fi in files:
            ret = ak.from_parquet(fi)
            X_track = ret["X_track"]
            X_cluster = ret["X_cluster"]

            assert len(X_track) == len(X_cluster)
            nev = len(X_track)

            for iev in range(nev):

                X1 = ak.to_numpy(X_track[iev])
                X2 = ak.to_numpy(X_cluster[iev])

                X1[np.isnan(X1)] = 0.0
                X1[np.isinf(X1)] = 0.0
                X2[np.isnan(X2)] = 0.0
                X2[np.isinf(X2)] = 0.0

                if len(X1) == 0 or len(X2) == 0:
                    continue

                ygen_track = ak.to_numpy(ret["ygen_track"][iev])
                ygen_cluster = ak.to_numpy(ret["ygen_cluster"][iev])
                ycand_track = ak.to_numpy(ret["ycand_track"][iev])
                ycand_cluster = ak.to_numpy(ret["ycand_cluster"][iev])

                if len(ygen_track) == 0 or len(ygen_cluster) == 0:
                    continue

                # pad feature dim between tracks and clusters to the same size
                if X1.shape[1] < X2.shape[1]:
                    X1 = np.pad(X1, [[0, 0], [0, X2.shape[1] - X1.shape[1]]])
                if X2.shape[1] < X1.shape[1]:
                    X2 = np.pad(X2, [[0, 0], [0, X1.shape[1] - X2.shape[1]]])

                # concatenate tracks and clusters in features and targets
                X = np.concatenate([X1, X2])
                ygen = np.concatenate([ygen_track, ygen_cluster])
                ycand = np.concatenate([ycand_track, ycand_cluster])

                assert ygen.shape[0] == X.shape[0]
                assert ycand.shape[0] == X.shape[0]

                # replace PID with index in labels array
                arr = np.array([labels.index(p) for p in ygen[:, 0]])
                ygen[:, 0][:] = arr[:]
                arr = np.array([labels.index(p) for p in ycand[:, 0]])
                ycand[:, 0][:] = arr[:]

                yield str(fi) + "_" + str(iev), {
                    "X": X.astype(np.float32),
                    "ygen": ygen,
                    "ycand": ycand,
                }

    events = generate_examples([fn])
    batched_data = []
    for event in events:
        Xs, ys_gen, ys_cand = event[1]["X"], event[1]["ygen"], event[1]["ycand"]
        d = Data(
            x=torch.tensor(Xs),
            ygen=torch.tensor(ys_gen[:, 1:]),
            ygen_id=torch.tensor(ys_gen[:, 0]).long(),
            ycand=torch.tensor(ys_cand[:, 1:]),
            ycand_id=torch.tensor(ys_cand[:, 0]).long(),
        )
        batched_data.append(d)
    return batched_data


def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class PFGraphDataset(Dataset):
    """
    Initialize parameters of graph dataset
    Args:
        root (str): path
    """

    def __init__(self, root, data, transform=None, pre_transform=None):
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)
        self._processed_dir = Dataset.processed_dir.fget(self)
        self.data = data

    @property
    def raw_file_names(self):
        raw_list = glob(osp.join(self.raw_dir, "*"))
        print("PFGraphDataset nfiles={}".format(len(raw_list)))
        return sorted([raw_path.replace(self.raw_dir, ".") for raw_path in raw_list])

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list])

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process_single_file(self, raw_file_name):
        """
        Loads raw datafile information and generates PyG Data() objects and stores them in .pt format.
        Args
            raw_file_name: raw data file name.
        Returns
            batched_data: a list of Data() objects of the form
             cms ~ Data(x=[#, 41], ygen=[#, 6], ygen_id=[#, 9], ycand=[#, 6], ycand_id=[#, 9])
             delphes ~ Data(x=[#, 12], ygen=[#elem, 6], ygen_id=[#, 6], ycand=[#, 6], ycand_id=[#, 6])
        """

        if self.data == "CMS":
            return prepare_data_cms(osp.join(self.raw_dir, raw_file_name))

        elif self.data == "DELPHES":
            return prepare_data_delphes(osp.join(self.raw_dir, raw_file_name))

        elif self.data == "CLIC":
            return prepare_data_clic([osp.join(self.raw_dir, raw_file_name)])

    def process_multiple_files(self, filenames, idx_file):
        datas = []
        for fn in tqdm.tqdm(filenames):
            x = self.process_single_file(fn)
            if x is None:
                continue
            datas.append(x)

        datas = sum(datas, [])
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx_file))
        torch.save(datas, p)
        print(f"saved file {p}")

    def process(self, num_files_to_batch):
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            self.process_multiple_files(fns, idx_file)
            idx_file += 1

    def process_parallel(self, num_files_to_batch, num_proc):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(num_proc)
        pool.map(process_func, pars)

    def get(self, idx):
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx))
        data = torch.load(p, map_location="cpu")
        return data

    def __getitem__(self, idx):
        return self.get(idx)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="'cms' or 'delphes'?")
    parser.add_argument("--dataset", type=str, required=True, help="Input data path")
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    """
    e.g. to run for cms
    python PFGraphDataset.py --data cms --dataset ../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi --processed_dir \
        ../../data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/processed --num-files-merge 1 --num-proc 1
    e.g. to run for delphes
    python3 PFGraphDataset.py --data delphes --dataset $sample --processed_dir $sample/processed \
        --num-files-merge 1 --num-proc 1
    """

    args = parse_args()

    pfgraphdataset = PFGraphDataset(root=args.dataset, data=args.data)

    if args.processed_dir:
        pfgraphdataset._processed_dir = args.processed_dir

    pfgraphdataset.process_parallel(args.num_files_merge, args.num_proc)
