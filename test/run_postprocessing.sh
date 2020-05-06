#!/bin/bash

IMG=~jpata/gpuservers/singularity/images/pytorch.simg

#which dataset to process
DATASET=$1

#how many event graphs to save in one pickle file
PERFILE=$2

#how many pickle files to merge to one pytorch/tensorflow file
MERGE=$3

#Produce pickle files
\ls -1 $DATASET/*.root | parallel -j20 --gnu singularity exec --nv $IMG \
  python test/postprocessing2.py --input {} \
    --events-per-file $PERFILE --outpath raw --save-normalized-table

#Produce TFRecords
singularity exec -B /storage --nv $IMG \
  python3 test/tf_data.py --target cand --datapath $DATASET --num-files-per-tfr $MERGE
