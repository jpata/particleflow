#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PYTHONPATH=`pwd`/mlpf:$PYTHONPATH

rm -Rf data/p8_ee_tt_ecm365
mkdir -p data/p8_ee_tt_ecm365
cd data/p8_ee_tt_ecm365

#download some test data
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_1.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_2.root

cd ../..

python3 fcc/postprocessing.py data/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_1.root
python3 fcc/postprocessing.py data/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_2.root

# #run the clic data validation notebook
# cd notebooks
# papermill --inject-output-path --log-output -p path ../data/clic/gev380ee_pythia6_ttbar_rfull201/ clic.ipynb ./out.ipynb
# cd ..

# tfds build mlpf/heptfds/clic_pf/ttbar --manual_dir `pwd`

# #Train, evaluate and make plots
# python mlpf/pipeline.py train --config parameters/clic.yaml --nepochs 1 --customize pipeline_test
# python mlpf/pipeline.py evaluate --nevents 5 --customize pipeline_test --train-dir ./experiments/clic* --weights ./experiments/clic*/weights/weights-01-*.hdf5
# python mlpf/pipeline.py plots --train-dir ./experiments/clic*
