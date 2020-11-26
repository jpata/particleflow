#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=8G

IMG=/home/software/singularity/base.simg:latest
cd ~/particleflow

#TF training
singularity exec --nv $IMG python3 mlpf/tensorflow/tf_model.py \
  --datapath ./data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --target cand --ntrain 70000 --ntest 20000 --convlayer ghconv \
  --lr 1e-5 --nepochs 100 --num-neighbors 10 \
  --num-hidden-id-enc 1 --num-hidden-id-dec 2 \
  --num-hidden-reg-enc 1 --num-hidden-reg-dec 2 \
  --bin-size 100 --hidden-dim-id 256 --hidden-dim-reg 256 \
  --batch-size 5 --distance-dim 256 \
  --dropout 0.0 \
  --num-convs-id 3 --num-convs-reg 3 --load experiments/run_13/weights.27-*.hdf5

#Pytorch  training
#singularity exec -B /home --nv $IMG \
#  python3 test/train_end2end.py \
#  --dataset /home/joosep/particleflow/data/TTbar_14TeV_TuneCUETP8M1_cfi \
#  --n_train 400 --n_val 100 \
#  --model PFNet7 --convlayer gravnet-radius --lr 0.005 \
#  --hidden_dim 32 --n_epochs 100 \
#  --l1 1000.0 --l2 100.0 --l3 1000.0 --space_dim 2 --nearest 5 --convlayer2 sgconv \
#  --target cand --batch_size 1 --activation leaky_relu \
#  --dropout 0.0 --encoding_dim 256 --optimizer adamw --radius 0.01 --input-encoding 0
