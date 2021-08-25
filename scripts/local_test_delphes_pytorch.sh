#!/bin/bash
set -e

rm -Rf test_tmp_delphes
mkdir test_tmp_delphes
cd test_tmp_delphes

mkdir -p experiments
mkdir -p data/pythia8_ttbar
mkdir -p data/pythia8_ttbar/raw
mkdir -p data/pythia8_ttbar/processed

mkdir -p data/pythia8_qcd
mkdir -p data/pythia8_qcd/raw
mkdir -p data/pythia8_qcd/processed

# download 2 files for training/validation
cd data/pythia8_ttbar/raw
echo Downloading the training/validation data files..
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
bzip2 -d *
cd ../../..

# download 1 file for testing
cd data/pythia8_qcd/raw
echo Downloading the testing data files..
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
bzip2 -d *
cd ../../..

#generate pytorch data files from pkl files
echo Processing the training/validation data files..
python3 ../mlpf/pytorch_delphes/graph_data_delphes.py --dataset data/pythia8_ttbar \
  --processed_dir data/pythia8_ttbar/processed --num-files-merge 1 --num-proc 1

#generate pytorch data files from pkl files
echo Processing the testing data files..
python3 ../mlpf/pytorch_delphes/graph_data_delphes.py --dataset data/pythia8_qcd/ \
  --processed_dir data/pythia8_qcd/processed --num-files-merge 1 --num-proc 1

# before training a model, first get rid of any previous models stored
rm -Rf experiments/*

cd ../mlpf/

#run the pytorch training
echo Beginning the training..
python3 pytorch_pipeline.py \
  --n_epochs=10 --n_train=1 --n_valid=1 --n_test=1 --batch_size=4 \
  --dataset='../test_tmp_delphes/data/pythia8_ttbar' \
  --dataset_qcd='../test_tmp_delphes/data/pythia8_qcd' \
  --outpath='../test_tmp_delphes/experiments'
