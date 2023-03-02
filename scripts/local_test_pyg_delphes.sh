#!/bin/bash
set -e

# download and process the datasets under particleflow/data/clic_edm4hep/
rm -Rf data/delphes/pythia8_ttbar
rm -Rf data/delphes/pythia8_qcd

# make delphes directories
mkdir -p data/delphes/pythia8_ttbar/raw
mkdir -p data/delphes/pythia8_qcd/raw

# download some ttbar test data
cd data/delphes/pythia8_ttbar/raw/
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_2.pkl.bz2
bzip2 -d *

# download some qcd test data
cd ../../pythia8_qcd/raw/
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_1.pkl.bz2
wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_2.pkl.bz2
bzip2 -d *

# setup directory for processed datafiles
cd ../..
mkdir -p pythia8_ttbar/processed
mkdir -p pythia8_qcd/processed

# process the raw datafiles
cd ../../mlpf/pyg/
echo -----------------------
for sample in ../../data/clic_edm4hep/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data DELPHES --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------

# run a supervised training of mlpf on DELPHES dataset
cd ../
python pyg_pipeline.py --dataset DELPHES --prefix MLPF_test
