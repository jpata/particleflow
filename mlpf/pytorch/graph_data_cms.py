import numpy as np
import os
import os.path as osp
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data, Batch
import itertools
from glob import glob
import numba
from numpy.lib.recfunctions import append_fields

import pickle
import scipy
import scipy.sparse
import math
import multiprocessing

elem_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
class_labels = [0, 1, 2, 11, 13, 22, 130, 211]

#map these to ids 0...Nclass
class_to_id = {r: class_labels[r] for r in range(len(class_labels))}

# map these to ids 0...Nclass
elem_to_id = {r: elem_labels[r] for r in range(len(elem_labels))}

# Data normalization constants for faster convergence.
# These are just estimated with a printout and rounding, don't need to be super accurate
# x_means = torch.tensor([ 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
# x_stds = torch.tensor([ 1.0, 22.0,  2.6,  1.8,  1.3,  1.9,  1.3,  1.0]).to(device)
# y_candidates_means = torch.tensor([0.0, 0.0, 0.0]).to(device)
# y_candidates_stds = torch.tensor([1.8, 2.0, 1.5]).to(device)
def process_func(args):
    self, fns, idx_file = args
    return self.process_multiple_files(fns, idx_file)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Do any in-memory transformations to data
def data_prep(data, device=torch.device('cpu')):
    #Create a one-hot encoded vector of the class labels
    data.y_candidates_id = data.ycand[:, 0].to(dtype=torch.long)
    data.y_gen_id = data.ygen[:, 0].to(dtype=torch.long)

    #one-hot encode the input categorical of the input
    elem_id_onehot = torch.nn.functional.one_hot(data.x[:, 0].to(dtype=torch.long), num_classes=len(elem_to_id))
    data.x = torch.cat([elem_id_onehot.to(dtype=torch.float), data.x[:, 1:]], axis=-1)

    data.y_candidates_weights = torch.ones(len(class_to_id)).to(device=device, dtype=torch.float)
    data.y_gen_weights = torch.ones(len(class_to_id)).to(device=device, dtype=torch.float)

    data.ycand = data.ycand[:, 1:]
    data.ygen = data.ygen[:, 1:]

    data.x[torch.isnan(data.x)] = 0.0
    data.ycand[torch.isnan(data.ycand)] = 0.0
    data.ygen[torch.isnan(data.ygen)] = 0.0
    data.ygen[data.ygen.abs()>1e4] = 0
    #print("x=", data.x)
    #print("y_candidates_id=", data.y_candidates_id)
    #print("y_gen_id=", data.y_gen_id)
    #print("ycand=", data.ycand)
    #print("ygen=", data.ygen)

class PFGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)
        self._processed_dir = Dataset.processed_dir.fget(self)

    @property
    def raw_file_names(self):
        raw_list = glob(osp.join(self.raw_dir, '*.pkl'))
        print("PFGraphDataset nfiles={}".format(len(raw_list)))
        return sorted([l.replace(self.raw_dir, '.') for l in raw_list])

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def processed_dir(self):
        return self._processed_dir

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, '*.pt'))
        return sorted([l.replace(self.processed_dir, '.') for l in proc_list])

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process_single_file(self, raw_file_name):
        with open(osp.join(self.raw_dir, raw_file_name), "rb") as fi:
            all_data = pickle.load(fi, encoding='iso-8859-1')

        batch_data = []

        # all_data is a list of only one element.. this element is a dictionary with keys: ["Xelem", "ycan", "ygen", 'dm', 'dm_elem_cand', 'dm_elem_gen']
        data = all_data[0]
        mat = data["dm_elem_cand"].copy()

        # Xelem contains all elements in 1 event
        # Xelem[i] contains the element #i in the event
        Xelem = data["Xelem"]
        ygen = data["ygen"]
        ycand = data["ycand"]

        # attach to every Xelem[i] (which is one element in the event) an extra elem_label
        Xelem = append_fields(Xelem, "typ_idx", np.array([elem_labels.index(int(i)) for i in Xelem["typ"]], dtype=np.float32))
        ygen = append_fields(ygen, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ygen["typ"]], dtype=np.float32))
        ycand = append_fields(ycand, "typ_idx", np.array([class_labels.index(abs(int(i))) for i in ycand["typ"]], dtype=np.float32))

        Xelem_flat = np.stack([Xelem[k].view(np.float32).data for k in [
            'typ_idx',
            'pt', 'eta', 'phi', 'e',
            'layer', 'depth', 'charge', 'trajpoint',
            'eta_ecal', 'phi_ecal', 'eta_hcal', 'phi_hcal',
            'muon_dt_hits', 'muon_csc_hits']], axis=-1
        )
        ygen_flat = np.stack([ygen[k].view(np.float32).data for k in [
            'typ_idx',
            'eta', 'phi', 'e', 'charge',
            ]], axis=-1
        )
        ycand_flat = np.stack([ycand[k].view(np.float32).data for k in [
            'typ_idx',
            'eta', 'phi', 'e', 'charge',
            ]], axis=-1
        )
        r = torch_geometric.utils.from_scipy_sparse_matrix(mat)

        x = torch.tensor(Xelem_flat, dtype=torch.float)
        ygen = torch.tensor(ygen_flat, dtype=torch.float)
        ycand = torch.tensor(ycand_flat, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=r[0].to(dtype=torch.long),
            #edge_attr=r[1].to(dtype=torch.float),
            ygen=ygen, ycand=ycand,
        )
        data_prep(data)
        batch_data += [data]

        return batch_data

    def process_multiple_files(self, filenames, idx_file):
        datas = [self.process_single_file(fn) for fn in filenames]
        datas = sum(datas, [])
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx_file))
        print(p)
        torch.save(datas, p)

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
        p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
        data = torch.load(p)
        return data

    def __getitem__(self, idx):
        return self.get(idx)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset path", required=True)
    parser.add_argument("--processed_dir", type=str, help="processed", required=False, default=None)
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    parser.add_argument("--num-proc", type=int, default=24, help="number of processes")
    args = parser.parse_args()

    pfgraphdataset = PFGraphDataset(root=args.dataset)

    if args.processed_dir:
        pfgraphdataset._processed_dir = args.processed_dir

    pfgraphdataset.process_parallel(args.num_files_merge,args.num_proc)
    #pfgraphdataset.process(args.num_files_merge)
