#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets

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
	python mlpf/data/postprocessing2.py \
	  --input $file \
	  --outpath local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --save-normalized-table --num-events 10
done

mkdir -p experiments

tfds build hep_tfds/heptfds/cms_pf/ttbar --manual_dir ./local_test_data

#Run a simple training on a few events
python mlpf/pipeline.py train -c parameters/cms-gen.yaml --nepochs 1 --customize pipeline_test

ls ./experiments/cms*/weights/

#Generate the pred.npz file of predictions
python mlpf/pipeline.py evaluate --customize pipeline_test -n 10 -t ./experiments/cms* -w ./experiments/cms*/weights/weights-01-*.hdf5

#Evaluate the notebook
papermill --inject-output-path --log-output -p path ./experiments/cms*/evaluation/epoch_1/cms_pf_ttbar/ notebooks/cms-mlpf.ipynb ./out.ipynb

#Retrain from existing weights
python mlpf/pipeline.py train -c parameters/cms-gen.yaml --nepochs 1 --customize pipeline_test -w ./experiments/cms*/weights/weights-01-*.hdf5
