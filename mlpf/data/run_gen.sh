#!/bin/bash
seq 1 500 | parallel -j10 ./genjob_pu.sh TTbar_14TeV_TuneCUETP8M1_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SingleElectronFlatPt1To100_pythia8_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SinglePiFlatPt0p7To10_cfi {}
#seq 1 100 | parallel -j12 ./genjob.sh SingleTauFlatPt2To150_cfi {}
