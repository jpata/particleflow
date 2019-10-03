
import uproot
import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
import itertools

class PFGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, connect_all=True, maxclusters=100, maxtracks=100, maxcands=100):
        self._connect_all = connect_all
        self._maxclusters = maxclusters
        self._maxtracks = maxtracks
        self._maxcands = maxcands
        super(PFGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['step3_AOD_1.root']

    @property
    def processed_file_names(self):
        nevents = 0
        for raw_file_name in self.raw_file_names:
            fi = uproot.open(osp.join(self.raw_dir, raw_file_name))
            tree = fi.get("pftree")
            nevents += len(tree)
        return ['data_{}.pt'.format(i) for i in range(nevents)]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def load_data(self):
        for raw_file_name in self.raw_file_names:
            print("loading data from ROOT files: {0}".format(osp.join(self.raw_dir, raw_file_name)))
        Xs_cluster = []
        Xs_track = []
        ys_cand = []
        
        for fn in self.raw_file_names:
            try:
                fi = uproot.open(osp.join(self.raw_dir, raw_file_name))
                tree = fi.get("pftree")
            except Exception as e:
                print("Could not open file {0}".format(fn))
                continue
            data = tree.arrays(tree.keys())
            data = {str(k, 'ascii'): v for k, v in data.items()}
            for iev in range(len(tree)):
                pt = data["pfcands_pt"][iev]
                eta = data["pfcands_eta"][iev]
                phi = data["pfcands_phi"][iev]
                charge = data["pfcands_charge"][iev]
    
                Xs_cluster += [np.stack([
                    data["clusters_energy"][iev][:self._maxclusters],
                    data["clusters_eta"][iev][:self._maxclusters],
                    data["clusters_phi"][iev][:self._maxclusters],
                ], axis=1)
                           ]
                Xs_track += [np.stack([
                    np.abs(1.0/data["tracks_qoverp"][iev][:self._maxtracks]),
                    data["tracks_inner_eta"][iev][:self._maxtracks],
                    data["tracks_inner_phi"][iev][:self._maxtracks],
                    data["tracks_outer_eta"][iev][:self._maxtracks],
                    data["tracks_outer_phi"][iev][:self._maxtracks],
                ], axis=1)
                         ]
                ys_cand += [np.stack([
                    pt[:self._maxcands],
                    eta[:self._maxcands],
                    phi[:self._maxcands],
                    charge[:self._maxcands]
                ], axis=1)
                        ]
        print("Loaded {0} events".format(len(Xs_cluster)))

        #zero pad 
        real_maxclusters = np.array([Xs_cluster[i].shape[0] for i in range(len(Xs_cluster))])
        for i in range(len(Xs_cluster)):
            Xs_cluster[i] = np.pad(Xs_cluster[i], [(0, self._maxclusters - Xs_cluster[i].shape[0]), (0,0)], mode='constant')
        
        real_maxtracks = np.array([Xs_track[i].shape[0] for i in range(len(Xs_cluster))])
        for i in range(len(Xs_track)):
            Xs_track[i] = np.pad(Xs_track[i], [(0, self._maxtracks - Xs_track[i].shape[0]), (0,0)], mode='constant')
    
        real_maxcands = np.array([ys_cand[i].shape[0] for i in range(len(ys_cand))])
        for i in range(len(ys_cand)):
            ys_cand[i] = np.pad(ys_cand[i], [(0, self._maxcands - ys_cand[i].shape[0]), (0,0)], mode='constant')
        
        Xs_cluster = np.stack(Xs_cluster, axis=0)
        Xs_track = np.stack(Xs_track, axis=0)
        ys_cand = np.stack(ys_cand, axis=0)

        Xs_cluster = Xs_cluster.reshape(Xs_cluster.shape[0], self._maxclusters, 3)
        Xs_track = Xs_track.reshape(Xs_track.shape[0], self._maxtracks, 5)
        ys_cand = ys_cand.reshape(ys_cand.shape[0], self._maxcands, 4)

        return Xs_cluster, Xs_track, ys_cand, real_maxclusters, real_maxtracks, real_maxcands

    def process(self):

        def withinDeltaR(first, second, dr=0.2):
            eta1 = first[:,1]
            eta2 = second[:,1]
            phi1 = first[:,2]
            phi2 = second[:,2]
            deta = np.abs(eta1 - eta2)
            dphi = np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi
            dr2 = dr*dr
            return ((deta**2 + dphi**2) < dr2)*1.
            
        feature_scale = np.array([1., 1., 1.])
        feature_scale_track = np.array([1., 1., 1., 1., 1.])

        Xs_cluster, Xs_track, ys_cand, real_maxclusters, real_maxtracks, real_maxcands = self.load_data()
        i = 0
        for event in range(Xs_cluster.shape[0]):
            if self._connect_all:
                pairs = [[i, j] for (i, j) in itertools.product(range(int(real_maxclusters[event])),range(int(real_maxclusters[event]))) if i!=j]
                pairs_track = [[i, j] for (i, j) in itertools.product(range(int(real_maxtracks[event])),range(int(real_maxtracks[event]))) if i!=j]
                pairs_cluster_track = [[i, j] for (i, j) in itertools.product(range(int(real_maxclusters[event])),range(int(real_maxtracks[event])))]
            else:
                pass
            edge_index = torch.tensor(pairs, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            edge_index_track = torch.tensor(pairs_track, dtype=torch.long)
            edge_index_track = edge_index_track.t().contiguous()
            edge_index_cluster_track = torch.tensor(pairs_cluster_track, dtype=torch.long)
            edge_index_cluster_track = edge_index_cluster_track.t().contiguous()
            x = torch.tensor(Xs_cluster[event]/feature_scale, dtype=torch.float)
            x_track = torch.tensor(Xs_track[event]/feature_scale_track, dtype=torch.float)

            #y = torch.tensor(ys_cand, dtype=torch.float)

            row, col = edge_index
            y = withinDeltaR(Xs_cluster[event][row], Xs_cluster[event][col])
            y = torch.tensor(y, dtype=torch.float)
            row, col = edge_index_track
            y_track = withinDeltaR(Xs_track[event][row], Xs_track[event][col])
            y_track = torch.tensor(y_track, dtype=torch.float)
            row, col = edge_index_cluster_track
            y_cluster_track = withinDeltaR(Xs_cluster[event][row], Xs_track[event][col])
            y_cluster_track = torch.tensor(y_cluster_track, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.x_track = x_track
            data.edge_index_track = edge_index_track
            data.edge_index_cluster_track = edge_index_cluster_track
            data.y_track = y_track
            data.y_cluster_track = y_cluster_track

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

    pfgraphdataset = PFGraphDataset(root='graph_data/',connect_all=True,maxclusters=100,maxtracks=100,maxcands=100)

