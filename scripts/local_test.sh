#!/bin/bash
set -e

mkdir -p data/pythia8_ttbar
cd data/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4452283)
wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_1.pkl.bz2

cd ../..

mkdir -p experiments
rm -Rf experiments/test-*

#Run a simple training on a few events
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action data

#Run a simple training on a few events
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action train

#Generate the pred.npz file of predictions
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action eval --weights ./experiments/test-*/weights.10-*.hdf5

#Generate the timing file
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action time --weights ./experiments/test-*/weights.10-*.hdf5

