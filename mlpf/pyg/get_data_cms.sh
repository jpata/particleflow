#!/bin/bash

set -e

# # make directory to hold datafiles in home directory of the repo
# mkdir -p ../../data

# # get the cms data
# rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms data/

# # unzip the data
# for sample in cms/* ; do
#   echo $sample
#     mv $sample/root $sample/processed
#     cd $sample/raw/
#     bzip2 -d *
#     cd ../../../
# done

# process the cms data
for sample in cms/* ; do
  echo $sample
  # #generate pytorch data files from pkl files
  # python3 preprocess_data.py --data cms --dataset $sample \
  #   --processed_dir $sample/processed --num-files-merge 10 --num-proc 1
done

# mv cms ../../data/
