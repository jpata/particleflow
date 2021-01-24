#!/bin/bash
set -e

rm -Rf test_tmp_delphes
mkdir test_tmp_delphes
cd test_tmp_delphes

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/processed

cd data/TTbar_14TeV_TuneCUETP8M1_cfi/raw

#download the pickle data files




cd test_tmp_delphes

#generate pytorch data files from pkl files
python3 ../mlpf/pytorch/graph_data_delphes.py --dataset data/delphes_cfi \
  --processed_dir data/delphes_cfi/processed --num-files-merge 1 --num-proc 1

#run the pytorch training
COMET_API_KEY="bla" python3 ../mlpf/pytorch/train_end2end_delphes.py \
  --dataset data/delphes_cfi --space_dim 2 --n_train 2 \
  --n_val 1 --model PFNet7 --convlayer gravnet-radius --convlayer2 "none" \
  --lr 0.0001 --hidden_dim 32 --n_epochs 2 --l1 1.0 --l2 0.001 --target cand \
  --batch_size 1 --dropout 0.2 --disable_comet
#
# #generate dataframe with predictions from the pytorch model
# python3 ../mlpf/pytorch/eval_end2end.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi \
#   --path data/PFNet* --model PFNet7 --start 3 --stop 5 --epoch 1
#
# export OUTFILE=`find data -name df.pkl.bz2 | head -n1`
# du $OUTFILE
# python3 ../mlpf/plotting/plots.py --pkl $OUTFILE --target cand
