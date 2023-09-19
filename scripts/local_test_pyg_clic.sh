#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

set -e

# download and process the datasets under particleflow/data/clic_edm4hep/
rm -Rf data/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380
rm -Rf data/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380

mkdir -p data/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/raw/
mkdir -p data/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/raw/

# download some ttbar test data
cd data/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_1.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_2.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_tt_ecm380/reco_p8_ee_tt_ecm380_3.parquet

# download some qcd test data
cd ../../p8_ee_qq_ecm380/raw/
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100001.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100002.parquet
wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100003.parquet

# setup directory for processed datafiles
cd ../..
mkdir -p p8_ee_tt_ecm380/processed
mkdir -p p8_ee_qq_ecm380/processed

# process the raw datafiles
cd ../../mlpf/pyg/
echo -----------------------
for sample in ../../data/clic_edm4hep_2023_02_27/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data CLIC --dataset $sample \
    --processed_dir $sample/processed --num-files-merge=1 --num-proc 1
done
echo -----------------------

# run a supervised training of mlpf on CLIC dataset
cd ../
env
python pyg_pipeline.py --dataset CLIC --data_path ../data/clic_edm4hep_2023_02_27/ --prefix MLPF_test --overwrite --n_train=1 --n_valid=1 --n_test=1
