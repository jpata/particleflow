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

import pickle
import scipy
import scipy.sparse
import math
import multiprocessing

#all candidate pdg-ids (multiclass labels)
class_labels = [0., -211., -13., -11., 1., 2., 11.0, 13., 22., 130., 211.]

#detector element labels
elem_labels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

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
    new_ids = torch.zeros_like(data.x[:, 0])
    for k, v in elem_to_id.items():
        m = data.x[:, 0] == v
        new_ids[m] = k
    data.x[:, 0] = new_ids

    #Convert pdg-ids to consecutive class labels
    new_ids = torch.zeros_like(data.ycand[:, 0])
    for k, v in class_to_id.items():
        m = data.ycand[:, 0] == v
        new_ids[m] = k
    data.ycand[:, 0] = new_ids
    
    new_ids = torch.zeros_like(data.ygen[:, 0])
    for k, v in class_to_id.items():
        m = data.ygen[:, 0] == v
        new_ids[m] = k
    data.ygen[:, 0] = new_ids

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

class PFGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)

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
        for idata, data in enumerate(all_data):
            mat = data["dm"].copy()
            mat_reco_cand = data["dm_elem_cand"].copy()
            mat_reco_gen = data["dm_elem_gen"].copy()

            mul1 = mat.multiply(mat_reco_cand)
            mul2 = mat.multiply(mat_reco_gen)
            mul1 = mul1>0
            mul2 = mul2>0
            if len(mat.row) > 0:
                mat_reco_cand = scipy.sparse.coo_matrix((np.array(mul1[mat.row, mat.col]).squeeze(), (mat.row, mat.col)), shape=(mat.shape[0], mat.shape[1]))
                mat_reco_gen = scipy.sparse.coo_matrix((np.array(mul2[mat.row, mat.col]).squeeze(), (mat.row, mat.col)), shape=(mat.shape[0], mat.shape[1]))
            else:
                mat_reco_cand = scipy.sparse.coo_matrix(np.zeros((mat.shape[0], mat.shape[1])))
                mat_reco_gen = scipy.sparse.coo_matrix(np.zeros((mat.shape[0], mat.shape[1])))

            X = data["Xelem"]
            ygen = data['ygen']
            ycand = data['ycand']
            #node_sel = X[:, 4] > 0.2
            #row_index, col_index, dm_data = mat.row, mat.col, mat.data

            #num_elements = X.shape[0]
            #num_edges = row_index.shape[0]

            #edge_index = np.zeros((2, 2*num_edges))
            #edge_index[0, :num_edges] = row_index
            #edge_index[1, :num_edges] = col_index
            #edge_index[0, num_edges:] = col_index
            #edge_index[1, num_edges:] = row_index
            #edge_index = torch.tensor(edge_index, dtype=torch.long)

            #edge_data = dm_data
            #edge_attr = np.zeros((2*num_edges, 1))
            #edge_attr[:num_edges,0] = edge_data
            #edge_attr[num_edges:,0] = edge_data
            #edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            r = torch_geometric.utils.from_scipy_sparse_matrix(mat)
            rc = torch_geometric.utils.from_scipy_sparse_matrix(mat_reco_cand)
            rg = torch_geometric.utils.from_scipy_sparse_matrix(mat_reco_gen)

            #edge_index, edge_attr = torch_geometric.utils.subgraph(torch.tensor(node_sel, dtype=torch.bool),
            #    edge_index, edge_attr, relabel_nodes=True, num_nodes=len(X))

            x = torch.tensor(X, dtype=torch.float)
            ygen = torch.tensor(ygen, dtype=torch.float)
            ycand = torch.tensor(ycand, dtype=torch.float)

            data = Data(
                x=x,
                edge_index=r[0].to(dtype=torch.long),
                edge_attr=r[1].to(dtype=torch.float),
                ygen=ygen, ycand=ycand,
                target_edge_attr_cand = rc[1].to(dtype=torch.float),
                target_edge_attr_gen = rg[1].to(dtype=torch.float),
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

    def process_parallel(self, num_files_to_batch):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(24)
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
    parser.add_argument("--num-files-merge", type=int, default=10, help="number of files to merge")
    args = parser.parse_args()
 
    pfgraphdataset = PFGraphDataset(root=args.dataset)
    pfgraphdataset.process_parallel(args.num_files_merge)
