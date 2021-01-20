#!/bin/bash
set -e

mkdir -p delphes/out/pythia8_ttbar
cd delphes/out/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo)
wget --no-check-certificate https://jpata.web.cern.ch/jpata/particleflow/pythia8_ttbar/tev14_pythia8_ttbar_0_0.pkl.bz2

cd ../..

python3 ../mlpf/tensorflow/delphes_data.py --datapath out/pythia8_ttbar
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action train
python3 ../mlpf/tensorflow/delphes_model.py --model-spec parameters/test.yaml --action validate --weights ./experiments/test-*/weights.100-*.hdf5
