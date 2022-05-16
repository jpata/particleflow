#!/bin/bash

set -e

# make directory to hold datafiles in home directory of the repo
mkdir -p ../../data

# make delphes directories
mkdir -p delphes/pythia8_ttbar/raw
mkdir -p delphes/pythia8_ttbar/processed

mkdir -p delphes/pythia8_qcd/raw
mkdir -p delphes/pythia8_qcd/processed

# get the ttbar data for training
cd data/delphes/pythia8_ttbar/raw/
for j in {0..9}
do
  for i in {0..49}
  do
    wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_"$j"_"$i".pkl.bz2
  done
done
bzip2 -d *

# get the qcd data for extra validation
cd ../../pythia8_qcd/raw/
for i in {0..49}
do
    wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_"$i".pkl.bz2
done
bzip2 -d *

# get back in the pytorch directory
cd ../../../../

#generate pytorch data files from pkl files
python3 preprocess_data.py --data delphes --dataset delphes/pythia8_ttbar \
  --processed_dir delphes/pythia8_ttbar/processed --num-files-merge 1 --num-proc 1

#generate pytorch data files from pkl files
python3 preprocess_data.py --data delphes --dataset delphes/pythia8_qcd \
  --processed_dir delphes/pythia8_qcd/processed --num-files-merge 1 --num-proc 1

mv delphes ../../data/
