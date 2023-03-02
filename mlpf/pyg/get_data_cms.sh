#!/bin/bash

set -e

# get the cms data
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms .

# unzip the data
for sample in cms/* ; do
  echo $sample
    mv $sample/root $sample/processed
    cd $sample/raw/
    bzip2 -d *
    cd ../../../
done

# make data/ directory to hold the cms/ directory of datafiles under particleflow/
mkdir -p ../../../data

# move the cms/ directory of datafiles there
mv cms ../../../data/
cd ..

# process the raw datafiles
echo -----------------------
for sample in ../../data/cms/* ; do
  echo $sample
  #generate pytorch data files from pkl files
  python3 PFGraphDataset.py --data cms --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 1 --num-proc 1
done
echo -----------------------
