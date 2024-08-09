#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PWD=`pwd`

rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi

mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240702_cptruthdef/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100000.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240702_cptruthdef/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100001.root

cd ../../..

#Create the ntuples
rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
for file in `\ls -1 local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root/*.root`; do
	python mlpf/data_cms/postprocessing2.py \
	  --input $file \
	  --outpath local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --num-events 10
done

mkdir -p experiments

tfds build mlpf/heptfds/cms_pf/ttbar --manual_dir ./local_test_data

#test transformer with onnx export
python mlpf/pyg_pipeline.py --config parameters/pytorch/pyg-cms.yaml --dataset cms --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ --num-epochs 2 --nvalid 1 --gpus 0 --train --test --make-plots --conv-type attention \
  --export-onnx --pipeline --dtype float32 --attention-type math --num-convs 1

# test Ray Train training
python mlpf/pyg_pipeline.py --config parameters/pytorch/pyg-cms.yaml --dataset cms --data-dir ${PWD}/tensorflow_datasets/ \
	--prefix MLPF_test_ --num-epochs 2 --nvalid 1 --gpus 0 --train --ray-train --ray-cpus 2 --local --conv-type attention \
	--pipeline --dtype float32 --attention-type math --num-convs 1 --experiments-dir ${PWD}/experiments
