#!/usr/bin/env python3
from __future__ import print_function
import sys, os, fnmatch

outdir = "/hdfs/local/joosep/mlpf/gen"
samples = [
    "SinglePiFlatPt0p7To10_cfi",
    "SingleTauFlatPt2To150_cfi",
    "SingleMuFlatPt0p7To10_cfi",
    "SingleElectronFlatPt1To100_pythia8_cfi",
    "SingleGammaFlatPt10To100_pythia8_cfi",
    "SinglePi0E10_pythia8_cfi",
]

samples_pu = [
    "TTbar_14TeV_TuneCUETP8M1_cfi",
]

if __name__ == "__main__":

    for s in samples+samples_pu:
        is_pu = s in samples_pu

        os.makedirs(outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(outdir + "/" + s + "/root", exist_ok=True)

        for iseed in range(1,2001):
            if not os.path.isfile(outdir+"/"+s+"/raw/pfntuple_{}.pkl".format(iseed)):
                if is_pu:
                    print("sbatch mlpf/tallinn/genjob_pu.sh {} {}".format(s, iseed))
                else:
                    print("sbatch mlpf/tallinn/genjob.sh {} {}".format(s, iseed))
                
