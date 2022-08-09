#!/bin/bash
set -e
export PYTHONPATH=`pwd`/hep_tfds

rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi

mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_2.root

cd ../../..

#Create the ntuples
rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
for file in `\ls -1 local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root/*.root`; do
	python3 mlpf/data/postprocessing2.py \
	  --input $file \
	  --outpath local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --save-normalized-table --num-events 10
done

mkdir -p experiments

tfds build hep_tfds/heptfds/cms_pf/ttbar --manual_dir ./local_test_data

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/cms.yaml --nepochs 2 --customize pipeline_test

ls ./experiments/cms_*/weights/

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate --customize pipeline_test -t ./experiments/cms_*

#Retrain from existing weights
python3 mlpf/pipeline.py train -c parameters/cms.yaml --nepochs 2 --customize pipeline_test -w ./experiments/cms_*/weights/weights-02-*.hdf5
