#!/bin/bash
set -e

rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi

mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root
cd local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_2.root
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_3.root

cd ../../..

#Create the ntuples
rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
for file in `\ls -1 local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/root/*.root`; do
	python3 mlpf/data/postprocessing2.py \
	  --input $file \
	  --outpath local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw \
	  --save-normalized-table --events-per-file 5
done

#Set aside some data for validation
mkdir -p local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/val
mv local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_3_0.pkl local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/val/

mkdir -p experiments
rm -Rf experiments/test-*

#Run a simple training on a few events
rm -Rf local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr
python3 mlpf/launcher.py --model-spec parameters/test-cms.yaml --action data

echo "Cloning hep_tfds."
git clone https://github.com/erwulff/hep_tfds.git
echo "Installing hep_tfds."
cd hep_tfds
python3 setup.py install
echo "Building TFRecords files."
tfds build heptfds/cms_pf --manual_dir ../local_test_data/TTbar_14TeV_TuneCUETP8M1_cfi
cd ..
rm -rf hep_tfds
echo "Removed hep_tfds repo."

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/test-cms.yaml -p test-cms-

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -c parameters/test-cms.yaml -t ./experiments/test-cms-*

python3 scripts/test_load_tfmodel.py ./experiments/test-cms-*/model_frozen/frozen_graph.pb

python3 mlpf/pipeline.py train -c parameters/test-cms-v2.yaml -p test-cms-v2-
python3 mlpf/pipeline.py evaluate -c parameters/test-cms-v2.yaml -t ./experiments/test-cms-v2-*

rm -Rf local_test_data