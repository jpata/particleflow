#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/local/joosep/mlpf/cms/20240823_simcluster"

samples = [
    #    ("TTbar_14TeV_TuneCUETP8M1_cfi",                           100000, 110010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",                200000, 210010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("QCDForPF_14TeV_TuneCUETP8M1_cfi",                        300000, 310010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi", 500000, 520010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("ZpTT_1500_14TeV_TuneCP5_cfi",                            600000, 620010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("VBF_TuneCP5_14TeV_pythia8_cfi",                          700000, 705010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("SinglePiMinusFlatPt0p7To1000_cfi",                       800000, 801010, "genjob_pu55to75.sh", outdir + "/pu55to75"),
    #    ("TTbar_14TeV_TuneCUETP8M1_cfi",                           700000, 720010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("VBF_TuneCP5_14TeV_pythia8_cfi",                          900000, 920010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("QCDForPF_14TeV_TuneCUETP8M1_cfi",                       1000000,1020010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi",               1100000,1120010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("MultiParticlePFGun50_cfi",                               800000, 801000, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleElectronFlatPt1To1000_pythia8_cfi",                900000, 910010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleGammaFlatPt1To1000_pythia8_cfi",                  1000000,1010010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleMuFlatPt1To1000_pythia8_cfi",                     1100000,1110010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleNeutronFlatPt0p7To1000_cfi",                      1200000,1201010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SinglePi0Pt1To1000_pythia8_cfi",                        1300000,1301010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SinglePiMinusFlatPt0p7To1000_cfi",                      1400000,1401010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleProtonMinusFlatPt0p7To1000_cfi",                  1500000,1501010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleTauFlatPt1To1000_cfi",                            1600000,1601010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("SingleK0FlatPt1To1000_pythia8_cfi",                     1700000,1710010, "genjob_nopu.sh", outdir + "/nopu"),
    #    ("TTbar_14TeV_TuneCUETP8M1_cfi",                           100000, 100050, "genjob_pu0to150.sh", outdir + "/pu0to150"),
]

if __name__ == "__main__":

    for samp, seed0, seed1, script, this_outdir in samples:
        os.makedirs(this_outdir + "/" + samp + "/raw", exist_ok=True)
        os.makedirs(this_outdir + "/" + samp + "/root", exist_ok=True)

        for seed in range(seed0, seed1):
            p = this_outdir + "/" + samp + "/root/pfntuple_{}.root".format(seed)
            if not os.path.isfile(p):
                print(
                    f"sbatch --mem-per-cpu 8G --partition main --time 20:00:00 --cpus-per-task 1 scripts/tallinn/cmssw-el8.sh mlpf/data/cms/{script} {samp} {seed}"
                )
