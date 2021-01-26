#!/bin/bash
set -e

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd data/pythia8_ttbar/root

wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_1.root

cd ../../..

mkdir -p experiments
rm -Rf experiments/test-*

#Create the ntuples
rm -Rf data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
python3 mlpf/data/postprocessing2.py --input data/pythia8_ttbar/root/pfntuple_1.root --outpath data/TTbar_14TeV_TuneCUETP8M1_cfi/raw --save-normalized-table --events-per-file 5

#Run a simple training on a few events
rm -Rf data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr
python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action data

#Run a simple training on a few events
python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action train

#Generate the pred.npz file of predictions
python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action eval --weights ./experiments/test-*/weights.02-*.hdf5

#Generate the timing file
python3 mlpf/launcher.py --model-spec parameters/cms-gnn-skipconn.yaml --action time --weights ./experiments/test-*/weights.02-*.hdf5

