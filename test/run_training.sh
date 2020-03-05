#!/bin/bash

singularity exec --nv -B /storage ~jpata/gpuservers/singularity/images/pytorch.simg python3 \
    test/train_end2end.py --model PFNet6 --n_train 200 \
    --batch_size 2 --n_epoch 100 --lr 0.0001 --hidden_dim 512
