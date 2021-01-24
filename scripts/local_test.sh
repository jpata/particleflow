#!/bin/bash
set -e

mkdir -p delphes/out/pythia8_ttbar
cd delphes/out/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4452283)
wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_8_0.pkl.bz2

cd ../..

mkdir -p experiments
rm -Rf experiments/test-*

#Generate larger-than-RAM TFRecord dataset
python3 ../mlpf/tensorflow/delphes_data.py --datapath out/pythia8_ttbar

#Run a simple training on a few events
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action train

#Generate the pred.npz file of predictions
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action validate --weights ./experiments/test-*/weights.10-*.hdf5

#Generate the timing file
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action timing --weights ./experiments/test-*/weights.10-*.hdf5

