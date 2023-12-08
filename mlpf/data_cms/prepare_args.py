#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/local/joosep/mlpf/cms/v3"

samples = [
    ("TTbar_14TeV_TuneCUETP8M1_cfi",                           100000, 100100, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    ("ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",                200000, 200100, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    ("QCDForPF_14TeV_TuneCUETP8M1_cfi",                        300000, 300100, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    ("QCD_Pt_3000_7000_14TeV_TuneCUETP8M1_cfi",                400000, 400100, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    ("SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi", 500000, 500100, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    ("ZpTT_1500_14TeV_TuneCP5_cfi",                            600000, 600100, "genjob_pu55to75.sh", outdir + "/pu55to75"),

    ("TTbar_14TeV_TuneCUETP8M1_cfi",                           700000, 700100, "genjob_nopu.sh", outdir + "/nopu"),
    ("MultiParticlePFGun50_cfi",                               800000, 800100, "genjob_nopu.sh", outdir + "/nopu"),

    ("SingleElectronFlatPt1To1000_pythia8_cfi",                900000, 900100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SingleGammaFlatPt1To1000_pythia8_cfi",                  1000000,1000100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SingleMuFlatPt1To1000_pythia8_cfi",                     1100000,1100100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SingleNeutronFlatPt0p7To1000_cfi",                      1200000,1200100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SinglePi0Pt1To1000_pythia8_cfi",                        1300000,1300100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SinglePiMinusFlatPt0p7To1000_cfi",                      1400000,1400100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SingleProtonMinusFlatPt0p7To1000_cfi",                  1500000,1500100, "genjob_nopu.sh", outdir + "/nopu"),
    ("SingleTauFlatPt1To1000_cfi",                            1600000,1600100, "genjob_nopu.sh", outdir + "/nopu"),
]

if __name__ == "__main__":

    for s, seed0, seed1, script, this_outdir in samples:
        os.makedirs(this_outdir + "/" + s + "/raw", exist_ok=True)
        os.makedirs(this_outdir + "/" + s + "/root", exist_ok=True)

        for seed in range(seed0, seed1):
            if not os.path.isfile(this_outdir + "/" + s + "/raw/pfntuple_{}.pkl.bz2".format(seed)):
                print("sbatch {} {} {}".format(script, s, seed))
