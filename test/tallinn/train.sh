#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4

IMG=/home/joosep/HEP-KBFI/singularity/base.simg
cd ~/particleflow

#TF training
#singularity exec --nv ~/particleflow/mlpf python3 test/tf_model.py --datapath data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr2 --target cand --ntrain 10000 --ntest 10000

#Pytorch training
singularity exec -B /home --nv $IMG \
  python3 test/train_end2end.py \
  --dataset /home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 400 --n_val 100 \
  --model PFNet7 --convlayer gravnet-radius --lr 0.005 \
  --hidden_dim 32 --n_epochs 100 \
  --l1 1000.0 --l2 100.0 --l3 1000.0 --space_dim 2 --nearest 5 --convlayer2 sgconv \
  --target cand --batch_size 1 --activation leaky_relu \
  --dropout 0.0 --encoding_dim 256 --optimizer adamw --radius 0.01 --input-encoding 0
