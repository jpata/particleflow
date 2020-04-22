#!/bin/bash

### PyTorch model training - SOTA

singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
  python3 test/train_end2end.py \
  --dataset /storage/group/gpu/bigdata/particleflow/TTbar_14TeV_TuneCUETP8M1_cfi \
  --n_train 10 --n_test 10 \
  --model PFNet5 --lr 0.0001 \
  --hidden_dim 128 --n_epochs 10 \
  --l1 1.0 --l2 0.001 --l3 0.0 \
  --target gen \
  --dropout 0.2

###Keras model trainings - experimental

#prepare TFRecords, already done in advance
#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_data.py --target cand

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_data.py --target gen

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 5000 --ntest 1000 --nepochs 100 --lr 0.00001 --nplot 0 --name run_01

#Distributed training
#CUDA_VISIBLE_DEVICES=2,3 singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 1000 --ntest 100 --nepochs 50 --lr 0.001 --nplot 5

#CUDA_VISIBLE_DEVICES=1,2,3,4 singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 5000 --ntest 1000 --nepochs 100 --lr 0.00001 --nplot 100 --load experiments/run_17/weights.12-9.90.hdf5

#singularity exec -B /storage --nv ~jpata/gpuservers/singularity/images/pytorch.simg \
#  python3 test/tf_model.py --target cand --ntrain 5000 --ntest 1000 --nepochs 0 --lr 0.00001 --nplot 0 --load experiments/run_17/weights.08-9.95.hdf5
