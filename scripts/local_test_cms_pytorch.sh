#!/bin/bash
# set -e
#
# rm -Rf test_tmp
# mkdir test_tmp
cd test_tmp

# mkdir -p experiments
# mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi
# cd data/TTbar_14TeV_TuneCUETP8M1_cfi
#
# #download the root input file
# wget --no-check-certificate https://jpata.web.cern.ch/jpata/mlpf/cms/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_1.root
# cd ../..
#
# mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
# mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/processed
#
# #generate pickle data files from root
# python3 ../../mlpf/data/postprocessing2.py --input data/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_1.root \
#   --events-per-file 1 --outpath data/TTbar_14TeV_TuneCUETP8M1_cfi/raw --save-normalized-table

#generate pytorch data files
python3 ../../mlpf/pytorch_cms/graph_data_delphes.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --processed_dir data/TTbar_14TeV_TuneCUETP8M1_cfi/processed --num-files-merge 1 --num-proc 1
#
# #run the pytorch training
# echo Beginning the training..
# python3 pipeline_cms.py \
#   --n_epochs=10 --n_train=1 --n_valid=1 --n_test=1 --batch_size=4 \
#   --dataset='../../test_tmp/data/TTbar_14TeV_TuneCUETP8M1_cfi' \
#   --outpath='../../test_tmp/experiments'
