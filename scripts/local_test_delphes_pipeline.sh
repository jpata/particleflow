#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets

#Download test input files (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
mkdir -p data/delphes_pf/pythia8_ttbar/raw
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
mv *.pkl.bz2 data/delphes_pf/pythia8_ttbar/raw

mkdir -p data/delphes_pf/pythia8_qcd/raw
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_1.pkl.bz2
mv *.pkl.bz2 data/delphes_pf/pythia8_qcd/raw

#Generate tensorflow datasets
tfds build mlpf/heptfds/delphes_pf/delphes_ttbar_pf --download_dir data/ --manual_dir data/delphes_pf
tfds build mlpf/heptfds/delphes_pf/delphes_qcd_pf --download_dir data/ --manual_dir data/delphes_pf

#Run a simple training on a few events
python mlpf/pipeline.py train --config parameters/tensorflow/delphes.yaml --nepochs 1 --ntrain 5 --ntest 5 --customize pipeline_test

#Check the weight files
ls ./experiments/delphes_*/weights/

#Generate the prediction files
python mlpf/pipeline.py evaluate --nevents 10 --customize pipeline_test --train-dir ./experiments/delphes_*

#Run plots
python mlpf/pipeline.py plots --train-dir ./experiments/delphes_*
