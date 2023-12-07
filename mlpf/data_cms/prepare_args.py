#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/local/joosep/mlpf/cms/v3_pre1"

samples = [
    #    "SingleElectronFlatPt1To1000_pythia8_cfi",
    #    "SingleGammaFlatPt1To1000_pythia8_cfi",
    #    "SingleMuFlatPt1To1000_pythia8_cfi",
    #    "SingleNeutronFlatPt0p7To1000_cfi",
    #    "SinglePi0Pt1To1000_pythia8_cfi",
    #    "SinglePiMinusFlatPt0p7To1000_cfi",
    #    "SingleProtonMinusFlatPt0p7To1000_cfi",
    #    "SingleTauFlatPt1To1000_cfi",
    #    ("MultiParticlePFGun50_cfi", 100000, 110050),
    #    "TTbar_14TeV_TuneCUETP8M1_cfi",
    #    "ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",
    #    "QCDForPF_14TeV_TuneCUETP8M1_cfi",
    #    "QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi",
    #    ("SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi", 200000, 202050),
    #    "ZpTT_1500_14TeV_TuneCP5_cfi",
    ("TTbar_14TeV_TuneCUETP8M1_cfi", 100000, 100100, "genjob_pu55to75.sh", outdir + "_pu55to75"),
    ("TTbar_14TeV_TuneCUETP8M1_cfi", 200000, 200100, "genjob_nopu.sh", outdir + "_nopu"),
]

if __name__ == "__main__":

    for s, seed0, seed1, script, this_outdir in samples:
        os.makedirs(this_outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(this_outdir + "/" + s + "/root", exist_ok=True)

        for seed in range(seed0, seed1):
            if not os.path.isfile(this_outdir + "/" + s + "/raw/pfntuple_{}.pkl.bz2".format(seed)):
                print("sbatch {} {} {}".format(script, s, seed))
