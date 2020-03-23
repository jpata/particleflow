import numpy as np
import os
import os.path as osp
import torch
import torch_geometric
import torch_geometric.utils
from torch_geometric.data import Dataset, Data
import itertools
from glob import glob
import numba

import scipy
import scipy.sparse
import math

@numba.njit
def deltaphi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

@numba.njit
def regularize_X_y(X_elements, y_candidates, X_element_block_id, y_candidate_block_id, edge_rows, edge_cols):
    ret_x = np.zeros_like(X_elements)
    ret_id = np.zeros_like(X_element_block_id)
    ret_y = np.zeros((X_elements.shape[0], y_candidates.shape[1]))
    
    idx = 0

    #reorder the input elements so blocks are contiguous
    inds = np.argsort(X_element_block_id)
    X_elements = X_elements[inds]
    X_element_block_id = X_element_block_id[inds]

    num_dropped = 0

    for cl in np.unique(X_element_block_id):
        m1 = X_element_block_id == cl
        m2 = y_candidate_block_id == cl

        #source element array
        x = X_elements[m1]
        #target candidate array
        y = y_candidates[m2]

        #we should always have more elements than candidates, this is a temporary workaround
        if len(y) > len(x):
            print("ERROR: dropping candidates: ")
            print(y[len(x):])
            num_dropped += (len(y) - len(x))
            #More candidates than elements, drop the remaining candidates
            y = y[:len(x)]

        #this array contains for each element in X an index to the index of the matched candidate in the target array
        inds_new = -1*np.ones(len(x), dtype=np.int32)

        #keep track of which candidates were already used by another element
        inds_used = np.zeros(len(y), dtype=np.dtype('?'))
        
        #For each element, find the first unmatched PF candidate with dR<0.01
        for ix in range(len(x)):
            for iy in range(len(y)):
                deta2 = (x[ix, 2] - y[iy, 2])**2
                dphi2 = deltaphi(x[ix, 3], y[iy, 3])**2
                dr = math.sqrt(deta2 + dphi2)
                #If this candidate was not yet matched to another element, assign the candidate iy to the element ix
                if dr < 0.01 and inds_used[iy]==False:
                    inds_new[ix] = iy
                    inds_used[iy] = True
                    break

        #Now find all the target PF candidates that were not assigned to any element and assign them
        #to the first unmatched element - this is again a workaround, because we don't know which individual element
        #produced those candidates
        inds_unmatched_y = np.where(~inds_used)[0]
        inds_unmatched_x = np.where(inds_new==-1)[0]
        if len(inds_unmatched_y) > 0:
            i = 0
            for ind in inds_unmatched_y:
                assert(i < len(inds_unmatched_x))
                inds_new[inds_unmatched_x[i]] = ind
                inds_used[ind] = True
                i += 1 
        #make sure all candidates were matched to an element
        assert(np.all(inds_used))

        #create the input array
        n = x.shape[0]
        ret_x[idx:idx+n] = x[:]
        ret_id[idx:idx+n] = cl
        
        #create the target array in the appropriate order
        i = 0
        for ind in inds_new:
            if ind != -1:
                ret_y[idx+i] = y[ind]
            i += 1
        
        idx += n
    assert(np.all(X_elements == ret_x))
    #assert(np.sum(y_candidates[:, 1]) == np.sum(ret_y[:, 1]))

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

    if num_dropped > 0:
        print("ERROR!: Dropped candidates", num_dropped, len(y_candidates))
    return inds, ret_x, ret_y, ret_id, edge_rows_new, edge_cols_new

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
        idx_file = 0
        for raw_file_name in self.raw_file_names:

            dist_file_name = raw_file_name.replace('ev','dist')
            reco_cand_adj_file_name = raw_file_name.replace('ev','cand')
            reco_gen_adj_file_name = raw_file_name.replace('ev','gen')
            print("loading data from files: {0}, {1}".format(osp.join(self.raw_dir, raw_file_name), osp.join(self.raw_dir, dist_file_name)))
            fi = np.load(osp.join(self.raw_dir, raw_file_name))
            mat = scipy.sparse.load_npz(osp.join(self.raw_dir, dist_file_name))
            mat_reco_cand = scipy.sparse.load_npz(osp.join(self.raw_dir, reco_cand_adj_file_name))
            mat_reco_gen = scipy.sparse.load_npz(osp.join(self.raw_dir, reco_gen_adj_file_name))
            mul1 = mat.multiply(mat_reco_cand)
            mul2 = mat.multiply(mat_reco_gen)
            mul1 = mul1>0
            mul2 = mul2>0
            mat_reco_cand = scipy.sparse.coo_matrix((np.array(mul1[mat.row, mat.col]).squeeze(), (mat.row, mat.col)), shape=(mat.shape[0], mat.shape[1]))
            mat_reco_gen = scipy.sparse.coo_matrix((np.array(mul2[mat.row, mat.col]).squeeze(), (mat.row, mat.col)), shape=(mat.shape[0], mat.shape[1]))

            X = fi['X']
            ygen = fi['ygen']
            ycand = fi['ycand']
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
            print("x={} ygen={} ycand={} edge_attr={}".format(x.shape, ygen.shape, ycand.shape, r[0].shape))
            p = osp.join(self.processed_dir, 'data_{}.pt'.format(idx_file))
            torch.save(data, p)
            idx_file += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


if __name__ == "__main__":

    dataset = "TTbar_gen_phase1"
    pfgraphdataset = PFGraphDataset(root='data/{}/'.format(dataset))
    pfgraphdataset.raw_dir = "data/{}".format(dataset)
    pfgraphdataset.processed_dir = "data/{}/processed".format(dataset)
    pfgraphdataset.process()
