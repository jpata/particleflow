#!/bin/bash

set -e

# # make directory to hold datafiles in home directory of the repo
# mkdir -p ../../data

# # get the cms data
# rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms data/

# unzip the data
for d in cms/* ; do
  echo $d
    mv $d/root $d/processed
    cd $d/raw/
    bzip2 -d *
    cd ../../../
done

# process the cms data
for d in cms/* ; do
  echo $d
  #generate pytorch data files from pkl files
  python3 preprocess_data.py --dataset $d \
    --processed_dir $d/processed --num-files-merge 10 --num-proc 1
done

mv cms ../../data/
