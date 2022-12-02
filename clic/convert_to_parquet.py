import bz2
import pickle
import awkward
import sys
import os

infile = sys.argv[1]
outfile = infile.replace(".pkl", ".parquet")
data = pickle.load(open(infile, "rb"))
arr = awkward.from_iter(data)
awkward.to_parquet(arr, outfile)
