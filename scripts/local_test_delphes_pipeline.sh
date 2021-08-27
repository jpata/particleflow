#!/bin/bash
set -e

mkdir -p data/pythia8_ttbar/raw
mkdir -p data/pythia8_qcd/val

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
mv tev14_pythia8_ttbar_0_0.pkl.bz2 data/pythia8_ttbar/raw/
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
mv tev14_pythia8_qcd_10_0.pkl.bz2 data/pythia8_qcd/val/

cd ../..

mkdir -p experiments

#Run a simple training on a few events
rm -Rf data/pythia8_ttbar/tfr
python3 mlpf/pipeline.py data -c parameters/delphes.yaml

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/delphes.yaml --nepochs 2 --ntrain 5 --ntest 5

ls ./experiments/delphes_*/weights/

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -c parameters/delphes.yaml -t ./experiments/delphes_*
