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
	python mlpf/data_cms/postprocessing2.py \
	  --input $file \
	  --outpath local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --save-normalized-table --num-events 10
done

mkdir -p experiments

tfds build mlpf/heptfds/cms_pf/ttbar --manual_dir ./local_test_data

#Run a simple training on a few events
python mlpf/pipeline.py train --config parameters/cms-gen.yaml --nepochs 1 --customize pipeline_test

ls ./experiments/cms*/weights/

#Generate the pred.npz file of predictions
python mlpf/pipeline.py evaluate --customize pipeline_test --nevents 10 --train-dir ./experiments/cms* --weights ./experiments/cms*/weights/weights-01-*.hdf5

cd notebooks

#Evaluate the notebook
papermill --inject-output-path --log-output -p path ../experiments/cms*/evaluation/epoch_1/cms_pf_ttbar/ ./cms-mlpf.ipynb ./out.ipynb

cd ..

#Retrain from existing weights
python mlpf/pipeline.py train -c parameters/cms-gen.yaml --nepochs 1 --customize pipeline_test --weights ./experiments/cms*/weights/weights-01-*.hdf5
