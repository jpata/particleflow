#!/bin/bash
set -e

mkdir -p data/pythia8_ttbar
mkdir -p data/pythia8_ttbar/val
cd data/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
mv tev14_pythia8_ttbar_0_1.pkl.bz2 val/

cd ../..

mkdir -p experiments
rm -Rf experiments/test-*

#Run a simple training on a few events
rm -Rf data/pythia8_ttbar/tfr
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action data

#Run a simple training on a few events
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action train

#Generate the pred.npz file of predictions
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action eval --weights ./experiments/test-*/weights-01-*.hdf5

#Generate the timing file
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action time --weights ./experiments/test-*/weights-01-*.hdf5

