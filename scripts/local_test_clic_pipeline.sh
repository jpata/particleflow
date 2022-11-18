#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets

rm -Rf local_test_data/gev380ee_pythia6_ttbar_rfull201
cd local_test_data/gev380ee_pythia6_ttbar_rfull201/

#download some test data
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0001_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0002_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0003_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0004_pandora.parquet

cd ../..

#run the clic data validation notebook
cd notebooks
papermill --inject-output-path --log-output -p path ../local_test_data/gev380ee_pythia6_ttbar_rfull201/ clic.ipynb ./out.ipynb
