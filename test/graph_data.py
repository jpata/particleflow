import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
from glob import glob
import numba

import scipy
import scipy.sparse

@numba.njit
def regularize_X_y(X_elements, y_candidates, X_element_block_id, y_candidate_block_id, edge_rows, edge_cols):
    ret_x = np.zeros_like(X_elements)
    ret_x_id = np.zeros_like(X_element_block_id)
    ret_y = np.zeros((X_elements.shape[0], y_candidates.shape[1]))
    ret_y_id = np.zeros((X_elements.shape[0]), dtype=np.int64)
    
    idx = 0

    #reorder the input elements so blocks are contiguous
    inds = np.argsort(X_element_block_id)
    X_elements = X_elements[inds]
    X_element_block_id = X_element_block_id[inds]

    #reorder the sparse distance matrix according to the new element ordering 
    edge_rows_new = np.zeros(len(edge_rows), dtype=edge_cols.dtype)
    edge_cols_new = np.zeros(len(edge_cols), dtype=edge_cols.dtype)
    n_edges = len(edge_rows)
    for i in range(len(inds)):
        for j in range(n_edges):
            if edge_rows[j] == inds[i]:
                edge_rows_new[j] = i 
            if edge_cols[j] == inds[i]:
                edge_cols_new[j] = i

    for cl in np.unique(X_element_block_id):
        m1 = X_element_block_id == cl
        m2 = y_candidate_block_id == cl

        x = X_elements[m1]
        y = y_candidates[m2]

        n = x.shape[0]
        ret_x[idx:idx+n] = x[:]
        ret_x_id[idx:idx+n] = cl
        ret_y[idx:idx+y.shape[0]] = y[:]
        ret_y_id[idx:idx+n] = cl
        
        idx += n
    assert(np.all(X_elements == ret_x))
 
    return inds, ret_x, ret_y, ret_x_id, ret_y_id, edge_rows_new, edge_cols_new

@numba.njit
def compute_distances(elements, dm):
    nel = len(elements)
    ndist = 0
    for i in range(nel):
        tp1 = elements[i, 0]
        for j in range(i+1, nel):
            tp2 = elements[j, 0]
            d = dist(elements[i, :], elements[j, :])
            if d < 0.02 and dm[i,j]==0:
                d = np.sqrt(d)
                dm[i,j] = d
                dm[j,i] = d
                ndist += 1
    #print("computed", ndist)

@numba.njit
def dist(el1, el2):
    if el1[0] > el2[0]:
        el = el1
        el1 = el2
        el2 = el
    deta2 = (el1[2] - el2[2])**2
    dphi2 = (el1[3] - el2[3])**2
    return deta2 + dphi2
 
class PFGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, connect_all=False, max_elements=None, max_candidates=None):
        self._connect_all = connect_all
        self._max_elements = max_elements
        self._max_candidates = max_candidates
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        raw_list = glob(self.raw_dir+'/*ev*.npz')
        return sorted([l.replace(self.raw_dir,'.') for l in raw_list])

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self.raw_file_names))]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        feature_scale = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        i = 0
        for raw_file_name in self.raw_file_names:
            dist_file_name = raw_file_name.replace('ev','dist')
            print("loading data from files: {0}, {1}".format(osp.join(self.raw_dir, raw_file_name), osp.join(self.raw_dir, dist_file_name)))
            try:
                fi = np.load(osp.join(self.raw_dir, raw_file_name))
                fi_dist = np.load(osp.join(self.raw_dir, dist_file_name))
            except Exception as e:
                print("Could not open files: {0}, {1}".format(osp.join(self.raw_dir, raw_file_name), osp.join(self.raw_dir, dist_file_name)))
                continue
            X_elements = fi['elements'][:self._max_elements]
            X_element_block_id = fi['element_block_id'][:self._max_elements]
            y_candidates = fi['candidates'][:self._max_candidates]
            y_candidate_block_id = fi['candidate_block_id'][:self._max_candidates]
            row_index = fi_dist['row']
            col_index = fi_dist['col']
            mat = scipy.sparse.coo_matrix((fi_dist["data"], (row_index, col_index)), shape=(len(X_elements), len(X_elements))).todense()
            #Add additional edges to create more initial connectivity
            #compute_distances(X_elements, mat)

            mat = scipy.sparse.coo_matrix(mat)
            row_index, col_index, dm_data = mat.row, mat.col, mat.data

            #Sort elements such that blocks are contiguous and Ncand == Nelem
            inds, X_elements, y_candidates, X_element_block_id, y_candidate_block_id, row_index, col_index = regularize_X_y(
                X_elements, y_candidates, X_element_block_id, y_candidate_block_id, row_index, col_index)

            num_elements = X_elements.shape[0]
            num_edges = row_index.shape[0]

            edge_index = np.zeros((2, 2*num_edges))
            edge_index[0,:num_edges] = row_index
            edge_index[1,:num_edges] = col_index
            edge_index[0,num_edges:] = col_index
            edge_index[1,num_edges:] = row_index
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            edge_data = dm_data
            edge_attr = np.zeros((2*num_edges,1))
            edge_attr[:num_edges,0] = edge_data
            edge_attr[num_edges:,0] = edge_data
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            x = torch.tensor(X_elements/feature_scale, dtype=torch.float)

            y = [X_element_block_id[i]==X_element_block_id[j] for (i,j) in edge_index.t().contiguous()]
            y = torch.tensor(y, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr,
                y_candidates=torch.tensor(y_candidates, dtype=torch.float),
                block_ids = torch.tensor(y_candidate_block_id, dtype=torch.float)
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            p = osp.join(self.processed_dir, 'data_{}.pt'.format(i))
            print(p)
            torch.save(data, p)
            i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


if __name__ == "__main__":

    pfgraphdataset = PFGraphDataset(root='data/TTbar_run3/')
    pfgraphdataset.raw_dir = "data/TTbar_run3"
    pfgraphdataset.processed_dir = "data/TTbar_run3/processed_jd2"
    pfgraphdataset.process()
