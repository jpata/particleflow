#!/bin/bash

### PyTorch model training - SOTA

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/train_end2end.py \
#  --dataset /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
#  --n_train 500 --n_test 100 \
#  --model PFNet7 --convlayer gravnet-radius --lr 0.0001 \
#  --hidden_dim 512 --n_epochs 50 \
#  --l1 1.0 --l2 0.001 --l3 0.0 \
#  --target gen \
#  --dropout 0.2

###Keras model trainings - experimental

CUDA_VISIBLE_DEVICES=0,1 singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/tf_model.py --target cand --ntrain 400000 --ntest 100000 --nepochs 100 --lr 0.00001 \
  --nhidden 512 --distance-dim 512 --num-conv 1 --weights inverse
