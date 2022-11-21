# #!/bin/bash

# set -e

# retrieve directories (also here https://jpata.web.cern.ch/jpata/mlpf/clic/)
rsync -r fmokhtar@lxplus.cern.ch:/eos/user/j/jpata/www/mlpf/clic ./

# restructure the sample directories to hold parquet files under raw/ and pT files under processed/
for sample in clic/* ; do
  echo Restrcuturing $sample sample directory
    mkdir $sample/raw
    mv $sample/*.parquet $sample/raw/
    mkdir $sample/processed
done

# make data/ directory to hold the clic/ directory of datafiles under particleflow/
mkdir -p ../../../data

# move the clic/ directory of datafiles there
mv clic ../../../data/
cd ..

# process the raw datafiles
echo -----------------------
for sample in ../../data/clic/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data clic --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------