from __future__ import print_function
import sys, os, fnmatch

samples = [
    #"SinglePiFlatPt0p7To10_cfi",
    #"SingleTauFlatPt2To150_cfi",
    #"SingleMuFlatPt0p7To10_cfi",
    #"SingleElectronFlatPt1To100_pythia8_cfi",
    #"SingleGammaFlatPt10To100_pythia8_cfi",
    #"SinglePi0E10_pythia8_cfi",
    #"MinBias_14TeV_pythia8_TuneCUETP8M1_cfi",
    "TTbar_14TeV_TuneCUETP8M1_cfi",
]

if __name__ == "__main__":
    for s in samples:
        for iseed in range(1001, 2001):
            print("{} {}".format(s, iseed))
