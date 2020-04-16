#!/bin/bash

singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/train_end2end.py \
  --dataset /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 10 --n_test 10 \
  --model PFNet7 --lr 0.0001 \
  --hidden_dim 512 --n_epochs 100 \
  --l1 1.0 --l2 0.001 --l3 0.0 \
  --target gen --convlayer gravnet-radius \
  --dropout 0.2

