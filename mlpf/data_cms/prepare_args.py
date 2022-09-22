#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/hdfs/local/joosep/mlpf/gen/v2_gun/"

samples = [
    "SingleElectronFlatPt1To1000_pythia8_cfi",
    "SingleGammaFlatPt1To1000_pythia8_cfi",
    "SingleMuFlatLogPt_100MeVto2TeV_cfi",
    "SingleNeutronFlatPt0p7To1000_cfi",
    "SinglePi0Pt1To1000_pythia8_cfi",
    "SinglePiMinusFlatPt0p7To1000_cfi",
    "SingleProtonMinusFlatPt0p7To1000_cfi",
    "SingleTauFlatPt1To1000_cfi",
]

samples_pu = [
    "TTbar_14TeV_TuneCUETP8M1_cfi",
    "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",
    "QCDForPF_14TeV_TuneCUETP8M1_cfi",
    "QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi",
    "SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi",
]

NUM_SAMPLES = 1000

if __name__ == "__main__":

    for s in samples_pu + samples:
        is_pu = s in samples_pu

        os.makedirs(outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(outdir + "/" + s + "/root", exist_ok=True)

        for nsamples in range(NUM_SAMPLES):
            if not os.path.isfile(outdir + "/" + s + "/raw/pfntuple_{}.pkl.bz2".format(nsamples)):
                if is_pu:
                    print("sbatch mlpf/tallinn/genjob_pu.sh {} {}".format(s, nsamples))
                else:
                    print("sbatch mlpf/tallinn/genjob.sh {} {}".format(s, nsamples))
