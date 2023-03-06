#!/bin/bash
set -e

# download and process the datasets under particleflow/data/clic_edm4hep/
rm -Rf data/clic_edm4hep/p8_ee_tt_ecm380
rm -Rf data/clic_edm4hep/p8_ee_qq_ecm380

mkdir -p data/clic_edm4hep/p8_ee_tt_ecm380/raw/
mkdir -p data/clic_edm4hep/p8_ee_qq_ecm380/raw/

# download some ttbar test data
cd data/clic_edm4hep/p8_ee_tt_ecm380/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_2.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_3.parquet

# download some qq test data
cd ../../p8_ee_qq_ecm380/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100001.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100002.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100003.parquet

# setup directory for processed datafiles
cd ../..
mkdir -p p8_ee_tt_ecm380/processed
mkdir -p p8_ee_qq_ecm380/processed

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
