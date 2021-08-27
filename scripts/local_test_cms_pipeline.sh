#!/bin/bash
set -e

rm -Rf data/TTbar_14TeV_TuneCUETP8M1_cfi

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd data/TTbar_14TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_2.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_3.root

cd ../../..

#Create the ntuples
rm -Rf data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
for file in `\ls -1 data/TTbar_14TeV_TuneCUETP8M1_cfi/root/*.root`; do
	python3 mlpf/data/postprocessing2.py \
	  --input $file \
	  --outpath data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --save-normalized-table --events-per-file 5
done

#Set aside some data for validation
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/val
mv data/TTbar_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_3_0.pkl data/TTbar_14TeV_TuneCUETP8M1_cfi/val/

mkdir -p experiments
rm -Rf experiments/test-*

#Run a simple training on a few events
rm -Rf data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr
python3 mlpf/pipeline.py data -c parameters/cms.yaml

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/cms.yaml --nepochs 2 --ntrain 5 --ntest 5

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -c parameters/cms.yaml -t ./experiments/cms-*

#Load the model
python3 scripts/test_load_tfmodel.py ./experiments/cms-*/model_frozen/frozen_graph.pb