#!/bin/bash
set -e

# download and process the datasets under particleflow/data/clic_edm4hep/
rm -Rf data/clic_edm4hep/p8_ee_tt_ecm365
rm -Rf data/clic_edm4hep/p8_ee_qcd_ecm365/

mkdir -p data/clic_edm4hep/p8_ee_tt_ecm365/raw/
mkdir -p data/clic_edm4hep/p8_ee_qcd_ecm365/raw/

# download some ttbar test data
cd data/clic_edm4hep/p8_ee_tt_ecm365/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_1.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_10.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_100.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_tt_ecm365/reco_p8_ee_tt_ecm365_1000.parquet

# download some qcd test data
cd ../../p8_ee_qcd_ecm365/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_qcd_ecm365/reco_p8_ee_qcd_ecm365_1.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_qcd_ecm365/reco_p8_ee_qcd_ecm365_10.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_qcd_ecm365/reco_p8_ee_qcd_ecm365_100.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/p8_ee_qcd_ecm365/reco_p8_ee_qcd_ecm365_1000.parquet

# setup directory for processed datafiles
cd ../..
mkdir -p p8_ee_tt_ecm365/processed
mkdir -p p8_ee_qcd_ecm365/processed

# process the raw datafiles
cd ../../mlpf/pyg_ssl/
echo -----------------------
for sample in ../../data/clic_edm4hep/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --dataset $sample \
    --processed_dir $sample/processed --num-files-merge 100 --num-proc 1
done
echo -----------------------

# run an ssl training of mlpf
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test --prefix_mlpf MLPF_test --train_mlpf --ssl
