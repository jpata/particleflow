#!/bin/bash

#PyTorch model training - SOTA
singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/train_end2end.py \
  --dataset /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 10 --n_test 10 \
  --model PFNet5 --lr 0.0001 \
  --hidden_dim 128 --n_epochs 10 \
  --l1 1.0 --l2 0.001 --l3 0.0 \
  --target gen \
  --dropout 0.2

#Keras model trainings - experimental
#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_data.py

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 15000 --ntest 5000 --nepochs 20 --lr 0.00001 --nplot 1 --name run_01

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 15000 --ntest 5000 --nepochs 20 --lr 0.00001 --nplot 1 --custom-training-loop --name run_02

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 2500 --ntest 500 --nepochs 20 --lr 0.00001 --nplot 2 --name run_03

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 2500 --ntest 500 --nepochs 20 --lr 0.00001 --nplot 2 --name run_04
