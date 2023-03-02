#!/bin/bash
set -e

# download and process the datasets under particleflow/data/clic_edm4hep/
rm -Rf data/delphes/pythia8_ttbar
rm -Rf data/delphes/pythia8_qcd

# make cms directories
mkdir -p data/delphes/pythia8_ttbar/raw
mkdir -p data/delphes/pythia8_qcd/raw

# download some ttbar test data
cd data/delphes/pythia8_ttbar/raw/
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_2.root

# download some qcd test data (TODO: this is ttbar)
cd ../../pythia8_qcd/raw/
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_2.root

# setup directory for processed datafiles
cd ../..
mkdir -p pythia8_ttbar/processed
mkdir -p pythia8_qcd/processed

# process the raw datafiles
cd ../../mlpf/pyg/
echo -----------------------
for sample in ../../data/clic_edm4hep/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data CMS --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------

# run a supervised training of mlpf on DELPHES dataset
cd ../
python pyg_pipeline.py --dataset CMS --prefix MLPF_test
