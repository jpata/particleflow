#!/bin/bash

#gun samples
./test/run_postprocessing.sh test/SingleGammaFlatPt10To100_pythia8_cfi -1 100
./test/run_postprocessing.sh test/SingleElectronFlatPt1To100_pythia8_cfi -1 100
./test/run_postprocessing.sh test/SingleMuFlatPt0p7To10_cfi -1 100
./test/run_postprocessing.sh test/SinglePi0E10_pythia8_cfi -1 100
./test/run_postprocessing.sh test/SinglePiFlatPt0p7To10_cfi -1 100
./test/run_postprocessing.sh test/SingleTauFlatPt2To150_cfi -1 100

#this one is quite time-consuming, up to 24h
#./test/run_postprocessing.sh data/TTbar_14TeV_TuneCUETP8M1_cfi 5 100
