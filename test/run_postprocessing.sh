#!/bin/bash

IMG=~jpata/gpuservers/singularity/images/pytorch.simg

#which dataset to process from data/
DATASET=TTbar_14TeV_TuneCUETP8M1_cfi

#how many event graphs to save in one pickle file
PERFILE=1
#how many pickle files to merge to one batch
MERGE=1

#DATASET=SingleElectronFlatPt1To100_pythia8_cfi
#PERFILE=-1
#MERGE=5

#DATASET=SinglePiFlatPt0p7To10_cfi
#PERFILE=-1
#MERGE=10

cd data/$DATASET
rm -Rf raw processed
mkdir -p raw
mkdir -p processed

\ls -1 pfntuple*.root | parallel --gnu -j20 python ../../test/postprocessing2.py --input {} --events-per-file $PERFILE

cd ../..
singularity exec --nv $IMG python test/graph_data.py --dataset data/$DATASET --num-files-merge $MERGE
