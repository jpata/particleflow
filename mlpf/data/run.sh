#!/bin/bash

#mkdir -p SinglePiFlatPt0p7To10_cfi
#for i in `seq 21 28`; do
#    ./genjob.sh SinglePiFlatPt0p7To10_cfi $i &
#done
#wait

#mkdir -p TTbar_14TeV_TuneCUETP8M1_cfi
#for i in `seq 3 4`; do
#    ./genjob_pu.sh TTbar_14TeV_TuneCUETP8M1_cfi $i &
#done
#wait

#SAMPLE=SingleMuFlatLogPt_100MeVto2TeV_cfi
#mkdir -p $SAMPLE 
#for i in `seq 1 10`; do
#    ./genjob.sh $SAMPLE $i &
#done
#wait

SAMPLE=SingleElectronFlatPt1To100_pythia8_cfi
mkdir -p $SAMPLE 
for i in `seq 1 10`; do
    ./genjob.sh $SAMPLE $i &
done
wait

#SAMPLE=SingleGammaFlatPt10To100_pythia8_cfi
#mkdir -p $SAMPLE 
#for i in `seq 1 10`; do
#    ./genjob.sh $SAMPLE $i &
#done
#wait
