#!/usr/bin/env python3
from __future__ import print_function
import sys, os, fnmatch

outdir = "/hdfs/local/joosep/mlpf/gen/v2"

samples = [
#    "SinglePiMinusFlatPt0p7To1000_cfi",
#    "SingleGammaFlatPt1To1000_pythia8_cfi",
#    "SingleElectronFlatPt1To1000_pythia8_cfi",
#    "SingleTauFlatPt1To1000_cfi",
#    "SinglePi0Pt1To1000_pythia8_cfi",
#    "SingleProtonMinusFlatPt0p7To1000_cfi",
#    "SingleNeutronFlatPt0p7To1000_cfi",
#    "SingleMuFlatLogPt_100MeVto2TeV_cfi",
]

samples_pu = [
    "TTbar_14TeV_TuneCUETP8M1_cfi",
    "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",
    "QCDForPF_14TeV_TuneCUETP8M1_cfi",
    "QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi",
]

NUM_SAMPLES = 1000

if __name__ == "__main__":

    iseed = 1
    for s in samples_pu+samples:
        is_pu = s in samples_pu

        num = 10
        if is_pu:
            num = NUM_SAMPLES

        os.makedirs(outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(outdir + "/" + s + "/root", exist_ok=True)

        for nsamples in range(num):
            if not os.path.isfile(outdir+"/"+s+"/raw/pfntuple_{}.pkl.bz2".format(iseed)):
                if is_pu:
                    print("sbatch mlpf/tallinn/genjob_pu.sh {} {}".format(s, iseed))
                else:
                    print("sbatch mlpf/tallinn/genjob.sh {} {}".format(s, iseed))
            iseed += 1 
