#!/bin/bash
set -e
export PYTHONPATH=`pwd`/hep_tfds

mkdir -p data/delphes_pf/pythia8_ttbar/raw
mkdir -p data/delphes_pf/pythia8_qcd/val

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
mv tev14_pythia8_ttbar_0_0.pkl.bz2 data/delphes_pf/pythia8_ttbar/raw/
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
mv tev14_pythia8_qcd_10_0.pkl.bz2 data/delphes_pf/pythia8_qcd/val/

tfds build hep_tfds/heptfds/delphes_pf --download_dir data/

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/delphes.yaml --nepochs 2 --ntrain 5 --ntest 5 --customize pipeline_test

ls ./experiments/delphes_*/weights/

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -t ./experiments/delphes_*
