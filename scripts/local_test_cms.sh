#!/bin/bash
set -e

rm -Rf test_tmp
mkdir test_tmp
cd test_tmp

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi
cd data/TTbar_14TeV_TuneCUETP8M1_cfi

#download the root input file
wget --no-check-certificate https://login-1.hep.caltech.edu/~jpata/particleflow/2020-07/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_1.root
cd ../..

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/processed

#generate pickle data files from root
python3 ../mlpf/data/postprocessing2.py --input data/TTbar_14TeV_TuneCUETP8M1_cfi/pfntuple_1.root \
  --events-per-file 1 --outpath data/TTbar_14TeV_TuneCUETP8M1_cfi/raw --save-normalized-table

#generate pytorch data files
python3 ../mlpf/pytorch/graph_data_cms.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --processed_dir data/TTbar_14TeV_TuneCUETP8M1_cfi/processed --num-files-merge 1 --num-proc 1

#run the pytorch training
COMET_API_KEY="bla" python3 ../mlpf/pytorch/train_end2end_cms.py \
  --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi --space_dim 2 --n_train 3 \
  --n_val 2 --model PFNet7 --convlayer gravnet-radius --convlayer2 sgconv \
  --lr 0.0001 --hidden_dim 32 --n_epochs 2 --l1 1.0 --l2 0.001 --target cand \
  --batch_size 1 --dropout 0.2 --disable_comet

# #generate dataframe with predictions from the pytorch model
python3 ../mlpf/pytorch/eval_end2end_cms.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --path data/PFNet* --model PFNet7 --start 3 --stop 5 --epoch 1

export OUTFILE=`find data -name df.pkl.bz2 | head -n1`
du $OUTFILE
python3 ../mlpf/plotting/plots.py --pkl $OUTFILE --target cand
