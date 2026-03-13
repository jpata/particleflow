#!/bin/bash
unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py \
    --data-dir $TFDS_DATA_DIR \
    --config parameters/pytorch/pyg-cms.yaml \
    train \
    --gpus 8 \
    --gpu-batch-multiplier 4 \
    --num-workers 1 \
    --prefetch-factor 1 \
    --conv-type attention \
    --dtype bfloat16 \
    --optimizer adamw \
    --lr 0.001 \
    --num-steps 500000 \
    --val-freq 10000 \
    --checkpoint-freq 100
