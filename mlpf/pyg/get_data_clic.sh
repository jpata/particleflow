#!/bin/bash

set -e

# retrieve datafiles
mkdir clic_edm4hep_2023_02_27
cd clic_edm4hep_2023_02_27
wget https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/clic_edm4hep_2023_02_27.tgz
tar xf clic_edm4hep_2023_02_27.tgz
rm -rf clic_edm4hep_2023_02_27.tgz
cd ..

# restructure the sample directories to hold parquet files under raw/ and pT files under processed/
for sample in clic_edm4hep_2023_02_27/* ; do
  echo Restructuring $sample sample directory
    mkdir $sample/raw
    mv $sample/*.parquet $sample/raw/
    mkdir $sample/processed
done

# process the raw datafiles
echo -----------------------
for sample in clic_edm4hep_2023_02_27/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data CLIC --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------

# make data/ directory to hold the clic_edm4hep_2023_02_27/ directory of datafiles under particleflow/
mkdir -p ../../data
mv clic_edm4hep_2023_02_27 ../../data/
