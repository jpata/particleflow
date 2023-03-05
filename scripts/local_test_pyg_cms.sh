# #!/bin/bash
# set -e

# # download and process the datasets under particleflow/data/cms/
# rm -Rf data/cms/TTbar_14TeV_TuneCUETP8M1_cfi
# rm -Rf data/cms/pythia8_qcd

# # make cms directories
# mkdir -p data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/raw
# mkdir -p data/cms/QCDForPF_14TeV_TuneCUETP8M1_cfi/raw

# # download some ttbar test data
# cd data/cms/TTbar_14TeV_TuneCUETP8M1_cfi/raw/
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_1.pkl.bz2
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_2.pkl.bz2
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_3.pkl.bz2

# # download some qcd test data
# cd ../../QCDForPF_14TeV_TuneCUETP8M1_cfi/raw/
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/QCDForPF_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_2001.pkl.bz2
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/QCDForPF_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_2002.pkl.bz2
# wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/QCDForPF_14TeV_TuneCUETP8M1_cfi/raw/pfntuple_2003.pkl.bz2

# # setup directory for processed datafiles
# cd ../..
# mkdir -p TTbar_14TeV_TuneCUETP8M1_cfi/processed
# mkdir -p QCDForPF_14TeV_TuneCUETP8M1_cfi/processed

# process the raw datafiles
# cd ../../mlpf/pyg/
cd mlpf/pyg/
echo -----------------------
for sample in ../../data/cms/* ; do
  echo Processing $sample sample
  python3 PFGraphDataset.py --data CMS --dataset $sample \
    --processed_dir $sample/processed --num-proc 1
done
echo -----------------------

# run a supervised training of mlpf on CMS dataset
cd ../
# cd mlpf/
python pyg_pipeline.py --dataset CMS --prefix MLPF_test --overwrite --n_train=1 --n_valid=1 --n_test=1
