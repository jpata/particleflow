#!/bin/bash

#seq 1 10 | parallel -j10 ./genjob_pu.sh TTbar_14TeV_TuneCUETP8M1_cfi {}
#seq 1 10 | parallel -j10 ./genjob_pu.sh ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SingleElectronFlatPt1To100_pythia8_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SinglePiFlatPt0p7To10_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SingleTauFlatPt2To150_cfi {}

#./genjob.sh SingleNeutronFlatPt0p7To1000_cfi 1 &
#./genjob.sh SingleProtonPlusFlatPt0p7To1000_cfi 1 &
#./genjob.sh SingleProtonMinusFlatPt0p7To1000_cfi 1 &
#./genjob.sh SinglePiPlusFlatPt0p7To1000_cfi 1 &
#./genjob.sh SinglePiMinusFlatPt0p7To1000_cfi 1 &
#./genjob.sh SingleGammaFlatPt1To1000_pythia8_cfi 1 &
#./genjob.sh SingleElectronFlatPt1To1000_pythia8_cfi 1
#./genjob.sh SingleTauFlatPt1To1000_cfi 1 &
#./genjob.sh SinglePi0Pt1To1000_pythia8_cfi 1 &

./genjob_pu.sh TTbar_14TeV_TuneCUETP8M1_cfi 1
./genjob_pu.sh ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi 1
#./genjob_pu.sh TTbar_14TeV_TuneCUETP8M1_cfi 2
