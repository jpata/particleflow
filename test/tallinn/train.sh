#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 4

cd ~/particleflow

#TF training
#singularity exec --nv ~/particleflow/mlpf python3 test/tf_model.py --datapath data/TTbar_14TeV_TuneCUETP8M1_cfi/tfr2 --target cand --ntrain 10000 --ntest 10000

#Pytorch training
singularity exec -B /scratch -B /home --nv /home/software/singularity/base.simg:2020-09-04 \
  python3 test/train_end2end.py \
  --dataset /home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 4000 --n_val 1000 \
  --model PFNet7 --convlayer gravnet-radius --lr 0.0001 \
  --hidden_dim 256 --n_epochs 100 \
  --l1 1000.0 --l2 0.0 --space_dim 3 --nearest 5 --convlayer2 sgconv \
  --target cand --batch_size 4 --activation leaky_relu \
  --dropout 0.0 --encoding_dim 256 --optimizer adamw --radius 1.0
