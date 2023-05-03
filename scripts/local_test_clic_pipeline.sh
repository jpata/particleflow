#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PYTHONPATH=`pwd`/mlpf:$PYTHONPATH

rm -Rf data/p8_ee_tt_ecm380
mkdir -p data/p8_ee_tt_ecm380
cd data/p8_ee_tt_ecm380

#download some test data
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_2.root

cd ../..

python3 fcc/postprocessing.py data/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1.root data/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1.parquet
python3 fcc/postprocessing.py data/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_2.root data/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_2.parquet

tfds build mlpf/heptfds/clic_pf_edm4hep/ttbar --manual_dir data

# #Train, evaluate and make plots
python mlpf/pipeline.py train --config parameters/clic.yaml --nepochs 1 --customize pipeline_test --ntrain 10 --ntest 10
python mlpf/pipeline.py evaluate --nevents 10 --customize pipeline_test --train-dir ./experiments/clic* --weights ./experiments/clic*/weights/weights-01-*.hdf5
python mlpf/pipeline.py plots --train-dir ./experiments/clic*

#try to train a fp16 model
python mlpf/pipeline.py train --config parameters/clic-fp16.yaml --nepochs 1 --customize pipeline_test --ntrain 10 --ntest 10
