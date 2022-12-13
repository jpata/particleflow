#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets

rm -Rf data/clic/gev380ee_pythia6_ttbar_rfull201
mkdir -p data/clic/gev380ee_pythia6_ttbar_rfull201
cd data/clic/gev380ee_pythia6_ttbar_rfull201

#download some test data
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0001_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0002_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0003_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0004_pandora.parquet

cd ../../..

#run the clic data validation notebook
cd notebooks
papermill --inject-output-path --log-output -p path ../data/clic/gev380ee_pythia6_ttbar_rfull201/ clic.ipynb ./out.ipynb
cd ..

tfds build mlpf/heptfds/clic_pf/ttbar --manual_dir `pwd`

python mlpf/pipeline.py train --config parameters/clic.yaml --nepochs 1 --customize pipeline_test --ntrain 10 --ntest 10
python mlpf/pipeline.py evaluate --nevents 5 --customize pipeline_test --train-dir ./experiments/clic* --weights ./experiments/clic*/weights/weights-01-*.hdf5
python mlpf/pipeline.py plots --train-dir ./experiments/clic*
