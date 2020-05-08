#!/bin/bash

./test/run_postprocessing.sh data/SingleGammaFlatPt10To100_pythia8_cfi 100 100
./test/run_postprocessing.sh data/SingleElectronFlatPt1To100_pythia8_cfi 100 100
./test/run_postprocessing.sh data/SingleMuFlatPt0p7To10_cfi 100 100
./test/run_postprocessing.sh data/SinglePi0E10_pythia8_cfi 100 100
./test/run_postprocessing.sh data/MinBias_14TeV_pythia8_TuneCUETP8M1_cfi 100 100
./test/run_postprocessing.sh data/SingleTauFlatPt2To150_cfi 100 100

#this one is quite time-consuming, up to 24h
#./test/run_postprocessing.sh data/TTbar_14TeV_TuneCUETP8M1_cfi 5 100
