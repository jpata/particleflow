import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools
from glob import glob
import numba

class PFGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, connect_all=False, max_elements=None, max_candidates=None):
        self._connect_all = connect_all
        self._max_elements = max_elements
        self._max_candidates = max_candidates
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)
        self.raw_dir = "data/TTbar/191009_155100"

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

        @numba.njit(parallel=True,fastmath=True)
        def withinDeltaR(first, second, dr=0.4):
            eta1 = first[:,2]
            eta2 = second[:,2]
            phi1 = first[:,3]
            phi2 = second[:,3]
            deta = np.abs(eta1 - eta2)
            dphi = np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi
            dr2 = dr*dr
            return ((deta**2 + dphi**2) < dr2)
            
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
            num_elements = X_elements.shape[0]

            row_index = fi_dist['row']
            col_index = fi_dist['col']
            num_edges = row_index.shape[0]

            edge_attr = fi_dist['data']
            edge_attr = edge_attr.reshape((num_edges,1))
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            edge_index = np.zeros((2, 2*num_edges))
            edge_index[0,:num_edges] = row_index
            edge_index[1,:num_edges] = col_index
            edge_index[0,num_edges:] = col_index
            edge_index[1,num_edges:] = row_index

            edge_index = torch.tensor(edge_index, dtype=torch.long)

            x = torch.tensor(X_elements/feature_scale, dtype=torch.float)

            y = [X_element_block_id[i]==X_element_block_id[j] for (i,j) in edge_index.t().contiguous()]
            y = torch.tensor(y, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


if __name__ == "__main__":

    pfgraphdataset = PFGraphDataset(root='graph_data/',connect_all=False,max_elements=None,max_candidates=None)
    pfgraphdataset.process()
