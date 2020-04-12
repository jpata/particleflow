#!/bin/bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5 singularity exec --nv ~/gpuservers/singularity/images/pytorch.simg python3 test/train_end2end.py --dataset data/TTbar_14TeV_TuneCUETP8M1_cfi --n_train 1000 --n_test 100 --model PFNet7 --lr 0.001 --hidden_dim 512 --n_epochs 100 --l2 0.01 --l3 0 --target gen --batch_size 5
