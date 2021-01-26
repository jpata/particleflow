# #!/bin/bash
# set -e
#
# rm -Rf test_tmp_delphes
# mkdir test_tmp_delphes
# cd test_tmp_delphes
#
# mkdir -p data/delphes_cfi
# mkdir -p data/delphes_cfi/raw
# mkdir -p data/delphes_cfi/processed
#
# cd data/delphes_cfi/raw
#
# #download some pickle data files (for this test we download 3 pkl files and allocate 2 for train and 1 for valid)
# wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_0.pkl.bz2
# wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_1.pkl.bz2
# wget --no-check-certificate -nc https://zenodo.org/record/4452283/files/tev14_pythia8_ttbar_0_10.pkl.bz2
#
# #decompress them
# bzip2 -d tev14_pythia8_ttbar_0_0.pkl.bz2
# bzip2 -d tev14_pythia8_ttbar_0_1.pkl.bz2
# bzip2 -d tev14_pythia8_ttbar_0_10.pkl.bz2
#
# cd ../../..
#
# #generate pytorch data files from pkl files
# python3 ../mlpf/pytorch/graph_data_delphes.py --dataset data/delphes_cfi \
#   --processed_dir data/delphes_cfi/processed --num-files-merge 1 --num-proc 1
cd test_tmp_delphes
# # before training a model, first get rid of any previous models stored
# rm -Rf data/PFNet*
#
# #run the pytorch training
# COMET_API_KEY="bla" python3 ../mlpf/pytorch/train_end2end_delphes.py \
#   --dataset data/delphes_cfi --space_dim 2 --n_train 1 \
#   --n_val 1 --model PFNet7 --convlayer gravnet-knn --convlayer2 "none" \
#   --lr 0.0001 --hidden_dim 32 --n_epochs 3 --l1 1.0 --l2 0.001 --target gen \
#   --batch_size 1 --dropout 0.2 --disable_comet

#predict on some test data and make plots
python3 ../mlpf/pytorch/eval_end2end_delphes.py --dataset data/delphes_cfi \
  --path data/PFNet* --model PFNet7 --start 1 --stop 2 --epoch 1 --target gen
