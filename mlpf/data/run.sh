#!/bin/bash

SAMPLE=SinglePiFlatPt0p7To10_cfi
mkdir -p $SAMPLE
for i in `seq 21 28`; do
    ./genjob.sh $SAMPLE $i &
done
wait
rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 
\ls -1 $SAMPLE/root/*.root | parallel --gnu -j16 python3 postprocessing2.py --input {} --outpath $SAMPLE/raw --save-normalized-table --events-per-file 5000

SAMPLE=TTbar_14TeV_TuneCUETP8M1_cfi
mkdir -p $SAMPLE 
for i in `seq 3 4`; do
    ./genjob_pu.sh $SAMPLE $i &
done
wait
rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 
\ls -1 $SAMPLE/root/*.root | parallel --gnu -j16 python3 postprocessing2.py --input {} --outpath $SAMPLE/raw --save-normalized-table --events-per-file 500

SAMPLE=SingleMuFlatLogPt_100MeVto2TeV_cfi
mkdir -p $SAMPLE 
for i in `seq 1 10`; do
    ./genjob.sh $SAMPLE $i &
done
wait

rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 
\ls -1 $SAMPLE/root/*.root | parallel --gnu -j12 python3 postprocessing2.py --input {} --outpath $SAMPLE/raw --save-normalized-table --events-per-file 5000

SAMPLE=SingleElectronFlatPt1To100_pythia8_cfi
mkdir -p $SAMPLE 
for i in `seq 1 10`; do
    ./genjob.sh $SAMPLE $i &
done
wait

rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 
\ls -1 $SAMPLE/root/*.root | parallel --gnu -j16 python3 postprocessing2.py --input {} --outpath $SAMPLE/raw --save-normalized-table --events-per-file 5000

SAMPLE=SingleGammaFlatPt10To100_pythia8_cfi
mkdir -p $SAMPLE 
for i in `seq 1 10`; do
    ./genjob.sh $SAMPLE $i &
done
wait

rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 

SAMPLE=SingleTauFlatPt2To150_cfi
mkdir -p $SAMPLE 
for i in `seq 1 1`; do
    ./genjob.sh $SAMPLE $i &
done
wait

rm -Rf $SAMPLE/raw
mkdir -p $SAMPLE/raw
mkdir -p $SAMPLE/root
mv $SAMPLE/*.root $SAMPLE/root/ 
\ls -1 $SAMPLE/root/*.root | parallel --gnu -j16 python3 postprocessing2.py --input {} --outpath $SAMPLE/raw --save-normalized-table --events-per-file 5000
