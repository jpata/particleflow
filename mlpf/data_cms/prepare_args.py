#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/local/joosep/mlpf/cms/v2/"

samples = [
    #    "SingleElectronFlatPt1To1000_pythia8_cfi",
    #    "SingleGammaFlatPt1To1000_pythia8_cfi",
    #    "SingleMuFlatPt1To1000_pythia8_cfi",
    #    "SingleNeutronFlatPt0p7To1000_cfi",
    #    "SinglePi0Pt1To1000_pythia8_cfi",
    #    "SinglePiMinusFlatPt0p7To1000_cfi",
    #    "SingleProtonMinusFlatPt0p7To1000_cfi",
    #    "SingleTauFlatPt1To1000_cfi",
    #    "MultiParticlePFGun_cfi",
    ("MultiParticlePFGun200_cfi", 100000, 102050),
    ("MultiParticlePFGun500_cfi", 200000, 202050),
]

samples_pu = [
    #    "TTbar_14TeV_TuneCUETP8M1_cfi",
    #    "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",
    #    "QCDForPF_14TeV_TuneCUETP8M1_cfi",
    #    "QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi",
    #    "SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi",
    #    "ZpTT_1500_14TeV_TuneCP5_cfi",
]

if __name__ == "__main__":

    for s, seed0, seed1 in samples_pu + samples:
        is_pu = s in samples_pu

        os.makedirs(outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(outdir + "/" + s + "/root", exist_ok=True)

        for seed in range(seed0, seed1):
            if not os.path.isfile(outdir + "/" + s + "/raw/pfntuple_{}.pkl.bz2".format(seed)):
                if is_pu:
                    print("sbatch genjob_pu.sh {} {}".format(s, seed))
                else:
                    print("sbatch genjob.sh {} {}".format(s, seed))
