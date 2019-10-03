
import uproot
import numpy as np
import glob
import os
import datetime
import json

def load_data(filename_pattern, maxclusters, maxtracks, maxcands):
    print("loading data from ROOT files: {0}".format(filename_pattern))
    Xs_cluster = []
    Xs_track = []
    ys_cand = []
        
    for fn in glob.glob(filename_pattern):
        try:
            fi = uproot.open(fn)
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
                data["clusters_energy"][iev][:maxclusters],
                data["clusters_eta"][iev][:maxclusters],
                data["clusters_phi"][iev][:maxclusters],
                ], axis=1)
            ]
            Xs_track += [np.stack([
                np.abs(1.0/data["tracks_qoverp"][iev][:maxtracks]),
                data["tracks_inner_eta"][iev][:maxtracks],
                data["tracks_inner_phi"][iev][:maxtracks],
                data["tracks_outer_eta"][iev][:maxtracks],
                data["tracks_outer_phi"][iev][:maxtracks],
                ], axis=1)
            ]
            ys_cand += [np.stack([
                pt[:maxcands],
                eta[:maxcands],
                phi[:maxcands],
                charge[:maxcands]
                ], axis=1)
            ]
    print("Loaded {0} events".format(len(Xs_cluster)))

    #zero pad 
    for i in range(len(Xs_cluster)):
        Xs_cluster[i] = np.pad(Xs_cluster[i], [(0, maxclusters - Xs_cluster[i].shape[0]), (0,0)], mode='constant')
    
    for i in range(len(Xs_track)):
        Xs_track[i] = np.pad(Xs_track[i], [(0, maxtracks - Xs_track[i].shape[0]), (0,0)], mode='constant')
    
    for i in range(len(ys_cand)):
        ys_cand[i] = np.pad(ys_cand[i], [(0,maxcands - ys_cand[i].shape[0]), (0,0)], mode='constant')

    Xs_cluster = np.stack(Xs_cluster, axis=0)
    Xs_track = np.stack(Xs_track, axis=0)
    ys_cand = np.stack(ys_cand, axis=0)

    Xs_cluster = Xs_cluster.reshape(Xs_cluster.shape[0], maxclusters, 3)
    Xs_track = Xs_track.reshape(Xs_track.shape[0], maxtracks, 5)
    ys_cand = ys_cand.reshape(ys_cand.shape[0], maxcands, 4)

if __name__ == "__main__":

    input_rootfiles_pattern = "./data/TTbar/*.root"
    maxclusters = 100
    maxtracks = 100
    maxcands = 100

    load_data(input_rootfiles_pattern, maxclusters, maxtracks, maxcands)
