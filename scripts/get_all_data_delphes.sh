# this script assumes you git cloned the repo and are inside the particleflow/scripts directory
# you can run the script using ./get_all_data_delphes.sh

#!/bin/bash
set -e

rm -Rf test_tmp_delphes
mkdir test_tmp_delphes
cd test_tmp_delphes

mkdir -p experiments

mkdir -p data/pythia8_ttbar
mkdir -p data/pythia8_ttbar/raw
mkdir -p data/pythia8_ttbar/processed

mkdir -p data/pythia8_qcd
mkdir -p data/pythia8_qcd/raw
mkdir -p data/pythia8_qcd/processed

# now get the ttbar data for training/testing
cd data/pythia8_ttbar/raw/

for j in {0..9}
do
  for i in {0..49}
  do
    wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_"$j"_"$i".pkl.bz2
  done
done

bzip2 -d *

# now get the qcd data for extra validation
cd ../../pythia8_qcd/raw/

for i in {0..49}
do
    wget --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_"$i".pkl.bz2
done

bzip2 -d *

# be in test_tmp_delphes when you process the files.. so the next cd tries to ensure that..
cd ../../../

#generate pytorch data files from pkl files
python3 ../particleflow/mlpf/pytorch/graph_data_delphes.py --dataset data/pythia8_ttbar \
  --processed_dir data/pythia8_ttbar/processed --num-files-merge 1 --num-proc 1

#generate pytorch data files from pkl files
python3 ../particleflow/mlpf/pytorch/graph_data_delphes.py --dataset data/pythia8_qcd \
  --processed_dir data/pythia8_qcd/processed --num-files-merge 1 --num-proc 1
