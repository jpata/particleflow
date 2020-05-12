#!/bin/bash

IMG=~jpata/gpuservers/singularity/images/pytorch.simg

#which dataset to process
DATASET=$1

#how many event graphs to save in one pickle file
PERFILE=$2

#how many pickle files to merge to one pytorch/tensorflow file
MERGE=$3

#Produce pickle files
mkdir -p $DATASET/raw
\ls -1 $DATASET/*.root | parallel -j20 --gnu singularity exec $IMG \
  python test/postprocessing2.py --input {} \
    --events-per-file $PERFILE --outpath $DATASET/raw --save-normalized-table

#Produce TFRecords
mkdir -p $DATASET/tfr/cand
singularity exec -B /storage $IMG \
  python3 test/tf_data.py --target cand --datapath $DATASET --num-files-per-tfr $MERGE
