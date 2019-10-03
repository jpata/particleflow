import sys
import numpy as np

from training import load_data

if __name__ == "__main__":

    maxclusters = -1
    maxtracks = -1
    maxcands = -1
    image_bins = 256
   
    infile = sys.argv[1] 
    cache_filename = infile.replace(".root", ".npz")
    data_images_in, data_images_out = load_data(infile, image_bins, maxclusters, maxtracks, maxcands)
    with open(cache_filename, "wb") as fi:
        np.savez(fi, data_images_in=data_images_in, data_images_out=data_images_out)
