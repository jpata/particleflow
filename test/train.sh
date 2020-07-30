#!/bin/bash

### PyTorch model training - WIP
singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/mlpf.simg \
  python3 test/train_end2end.py \
  --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 4000 --n_val 1000 \
  --model PFNet7 --convlayer gravnet-radius --lr 0.0001 \
  --hidden_dim 256 --n_epochs 50 \
  --l1 1.0 --l2 0.001 --l3 0.0 \
  --target cand --batch_size 1 \
  --dropout 0.2

###Keras model trainings - current SOTA
singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/tf_model.py --target cand --ntrain 10000 --ntest 10000 --nepochs 100 --lr 0.00001 \
  --datapath /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --nhidden 256 --distance-dim 256 --num-conv 4 --weights inverse --lr-decay 0 --convlayer ghconv
