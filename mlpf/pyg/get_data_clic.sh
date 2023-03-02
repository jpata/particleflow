#!/bin/bash

set -e

# retrieve directories (also here https://jpata.web.cern.ch/jpata/mlpf/clic/)
rsync -r fmokhtar@lxplus.cern.ch:/eos/user/j/jpata/mlpf/clic_edm4hep ./

# restructure the sample directories to hold parquet files under raw/ and pT files under processed/
for sample in clic_edm4hep/* ; do
  echo Restructuring $sample sample directory
    mkdir $sample/raw
    mv $sample/*.parquet $sample/raw/
    mkdir $sample/processed
done

# make data/ directory to hold the clic_edm4hep/ directory of datafiles under particleflow/
mkdir -p ../../../data

# move the clic_edm4hep/ directory of datafiles there
mv clic_edm4hep ../../../data/
cd ..

# process the raw datafiles
echo -----------------------
for sample in ../../data/clic_edm4hep/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------
