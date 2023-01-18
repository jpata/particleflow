#!/bin/bash
set -e

# download and process the datasets under particleflow/data/clic/
rm -Rf data/clic/gev380ee_pythia6_ttbar_rfull201
rm -Rf data/clic/gev380ee_pythia6_qcd_all_rfull201/

mkdir -p data/clic/gev380ee_pythia6_ttbar_rfull201/raw/
mkdir -p data/clic/gev380ee_pythia6_qcd_all_rfull201/raw/

# download some ttbar test data
cd data/clic/gev380ee_pythia6_ttbar_rfull201/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0001_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0002_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0003_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0004_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_ttbar_rfull201/pythia6_ttbar_0005_pandora.parquet

# download some qcd test data
cd ../../gev380ee_pythia6_qcd_all_rfull201/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_qcd_all_rfull201/pythia6_qcd_all_0001_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_qcd_all_rfull201/pythia6_qcd_all_0002_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_qcd_all_rfull201/pythia6_qcd_all_0006_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_qcd_all_rfull201/pythia6_qcd_all_0007_pandora.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic/gev380ee_pythia6_qcd_all_rfull201/pythia6_qcd_all_0008_pandora.parquet

# setup directory for processed datafiles
cd ../..
mkdir -p gev380ee_pythia6_ttbar_rfull201/processed
mkdir -p gev380ee_pythia6_qcd_all_rfull201/processed

# process the raw datafiles
cd ../../mlpf/pyg_ssl/
echo -----------------------
for sample in ../../data/clic/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------

# run an ssl training of mlpf
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test --prefix_mlpf MLPF_test --train_mlpf --ssl
