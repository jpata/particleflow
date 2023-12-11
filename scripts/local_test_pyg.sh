#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets

rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi

mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/v3/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_10001.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/v3/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_10002.root

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

python mlpf/pyg_pipeline.py --config parameters/pyg-workflow-test.yaml --dataset cms --data-dir ./tensorflow_datasets/ --prefix MLPF_test_ --nvalid 1 --gpus 0 --train --test --make-plots

python mlpf/pyg_pipeline.py --config parameters/pyg-workflow-test.yaml --dataset cms --data-dir ./tensorflow_datasets/ --prefix MLPF_test_ --nvalid 1 --gpus 0 --train --test --make-plots --conv-type gnn_lsh --export-onnx
