import sys
import numpy as np

from training import load_data, create_images, zeropad

if __name__ == "__main__":

    maxclusters = -1
    maxtracks = -1
    maxcands = -1
    image_bins = 256
   
    infile = sys.argv[1] 
    cache_filename = infile.replace(".root", ".npz")
    Xs_cluster, Xs_track, ys_cand = load_data(infile, image_bins, maxclusters, maxtracks, maxcands)
    print(np.mean([len(x) for x in Xs_cluster]))
    data_images_in, data_images_out = create_images(Xs_cluster, Xs_track, ys_cand, image_bins)
    with open(cache_filename, "wb") as fi:
        np.savez(fi, data_images_in=data_images_in, data_images_out=data_images_out)

    Xs_cluster, Xs_track, ys_cand = zeropad(Xs_cluster, Xs_track, ys_cand, 5000, 5000, 5000)
    with open(infile.replace(".root", "_flat.npz"), "wb") as fi:
        np.savez(fi, Xs_cluster=Xs_cluster, Xs_track=Xs_track, ys_cand=ys_cand)
